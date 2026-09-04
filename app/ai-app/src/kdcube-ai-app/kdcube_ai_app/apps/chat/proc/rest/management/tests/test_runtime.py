from __future__ import annotations

from types import SimpleNamespace

import pytest
from kdcube_ai_app.apps.chat.proc.rest.management.runtime import (
    KDCubeManagementRuntime,
)
from kdcube_ai_app.apps.chat.proc.rest.management.service import (
    ManagementApplicationNotFound,
)
from kdcube_ai_app.infra.plugin.bundle_loader import (
    APIEndpointSpec,
    BundleInterfaceManifest,
    CronJobSpec,
    MCPEndpointSpec,
    OnJobSpec,
    OnMessageSpec,
    UIWidgetSpec,
)
from starlette.requests import Request

APPLICATION_ID = "workspace@1-0"


def _request() -> Request:
    app = SimpleNamespace(state=SimpleNamespace(redis_async=object()))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("runtime.example", 443),
            "app": app,
        }
    )


def _registry() -> SimpleNamespace:
    return SimpleNamespace(
        bundles={
            APPLICATION_ID: SimpleNamespace(
                path="/private/source/path",
                module="entrypoint",
                singleton=False,
            )
        },
        default_bundle_id=APPLICATION_ID,
    )


@pytest.mark.asyncio
async def test_surface_discovery_returns_declared_relative_routes(monkeypatch) -> None:
    from kdcube_ai_app.apps.chat.proc.rest.integrations import integrations
    from kdcube_ai_app.apps.chat.proc.rest.management import runtime as runtime_module

    async def _load_registry(*_args, **_kwargs):
        return _registry()

    manifest = BundleInterfaceManifest(
        bundle_id=APPLICATION_ID,
        api_endpoints=(
            APIEndpointSpec(
                method_name="status",
                alias="status",
                http_method="GET",
                route="public",
            ),
        ),
        mcp_endpoints=(
            MCPEndpointSpec(
                method_name="tools",
                alias="workspace",
                route="operations",
                transport="streamable-http",
            ),
        ),
        ui_widgets=(
            UIWidgetSpec(
                method_name="workspace_widget",
                alias="workspace",
                icon={},
            ),
        ),
        scheduled_jobs=(CronJobSpec(method_name="daily", alias="daily"),),
        on_job=OnJobSpec(method_name="on_job"),
        on_message=OnMessageSpec(method_name="on_message"),
    )
    monkeypatch.setattr(runtime_module, "load_registry", _load_registry)
    monkeypatch.setattr(
        runtime_module,
        "load_bundle_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        integrations,
        "_authoritative_bundle_props",
        lambda **_kwargs: {},
    )

    result = await KDCubeManagementRuntime(
        _request(),
        tenant="tenant-a",
        project="project-a",
    ).application_surfaces(APPLICATION_ID)

    base = (
        "/api/integrations/bundles/tenant-a/project-a/"
        "workspace@1-0"
    )
    assert result == {
        "application_id": APPLICATION_ID,
        "surfaces": {
            "api": [
                {
                    "alias": "status",
                    "method": "GET",
                    "path": f"{base}/public/status",
                }
            ],
            "mcp": [
                {
                    "alias": "workspace",
                    "transport": "streamable-http",
                    "path": f"{base}/mcp/workspace",
                }
            ],
            "widgets": [
                {
                    "alias": "workspace",
                    "path": f"{base}/widgets/workspace",
                }
            ],
            "jobs": [{"alias": "daily"}, {"alias": "on_job"}],
            "messaging": [{"kind": "on_message"}],
        },
    }
    assert "/private/source/path" not in str(result)


@pytest.mark.asyncio
async def test_reload_passes_one_exact_declared_application(monkeypatch) -> None:
    from kdcube_ai_app.apps.chat.proc.rest.integrations import integrations
    from kdcube_ai_app.apps.chat.proc.rest.management import runtime as runtime_module

    async def _load_registry(*_args, **_kwargs):
        return _registry()

    calls = []

    async def _reload(request, session, payload):
        calls.append((request, session, payload))
        return {"changed_bundle_ids": [payload.bundle_id]}

    monkeypatch.setattr(runtime_module, "load_registry", _load_registry)
    monkeypatch.setattr(integrations, "_do_reload_bundles_from_authority", _reload)
    monkeypatch.setattr(
        runtime_module.application_readiness_registry,
        "snapshot",
        lambda **_kwargs: SimpleNamespace(desired_generation="generation-7"),
    )
    request = _request()
    runtime = KDCubeManagementRuntime(
        request,
        tenant="tenant-a",
        project="project-a",
    )

    result = await runtime.reload_application(
        APPLICATION_ID,
        caller_profile="caller-profile-1",
    )

    assert result == {
        "application_id": APPLICATION_ID,
        "state": "completed",
        "changed_application_ids": [APPLICATION_ID],
        "generation": "generation-7",
    }
    assert len(calls) == 1
    assert calls[0][0] is request
    assert calls[0][1].user_id == "caller-profile-1"
    assert calls[0][2].tenant == "tenant-a"
    assert calls[0][2].project == "project-a"
    assert calls[0][2].bundle_id == APPLICATION_ID


@pytest.mark.asyncio
async def test_reload_rejects_an_undeclared_application_before_internal_reload(
    monkeypatch,
) -> None:
    from kdcube_ai_app.apps.chat.proc.rest.integrations import integrations
    from kdcube_ai_app.apps.chat.proc.rest.management import runtime as runtime_module

    async def _load_registry(*_args, **_kwargs):
        return _registry()

    async def _must_not_reload(*_args, **_kwargs):
        raise AssertionError("internal reload must not run")

    monkeypatch.setattr(runtime_module, "load_registry", _load_registry)
    monkeypatch.setattr(
        integrations,
        "_do_reload_bundles_from_authority",
        _must_not_reload,
    )
    runtime = KDCubeManagementRuntime(
        _request(),
        tenant="tenant-a",
        project="project-a",
    )

    with pytest.raises(ManagementApplicationNotFound):
        await runtime.reload_application(
            "other@1-0",
            caller_profile="caller-profile-1",
        )
