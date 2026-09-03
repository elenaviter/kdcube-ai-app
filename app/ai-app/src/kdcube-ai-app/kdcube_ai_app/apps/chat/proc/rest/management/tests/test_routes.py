from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kdcube_ai_app.apps.chat.proc.rest.management import routes
from kdcube_ai_app.apps.chat.proc.rest.management.config import (
    DelegatedManagementConfig,
)
from kdcube_ai_app.apps.chat.proc.rest.management.contracts import (
    RELOAD_OPERATION,
    management_resource,
)
from kdcube_ai_app.apps.chat.proc.rest.management.service import ManagementResponse


TENANT = "tenant-a"
PROJECT = "project-a"
RESOURCE = management_resource(TENANT, PROJECT)


def _config(*, enabled: bool = True) -> DelegatedManagementConfig:
    return DelegatedManagementConfig(
        enabled=enabled,
        tenant=TENANT,
        project=PROJECT,
        connection_hub_app_id="connection-hub@1-0",
        service_id="kdcube-management",
        service_secret_ref="connections.management.signing_secret",
        admission_url="http://connection-hub.test/admission",
    )


def _client(monkeypatch, *, enabled: bool = True) -> TestClient:
    config = _config(enabled=enabled)
    monkeypatch.setattr(
        routes,
        "_configuration",
        lambda: (SimpleNamespace(INSTANCE_ID="proc-test"), config, RESOURCE),
    )
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/integrations")
    return TestClient(app)


def test_metadata_publishes_resource_authorization_server_and_operations(
    monkeypatch,
) -> None:
    response = _client(monkeypatch).get(
        "/api/integrations/management/v1/.well-known/oauth-protected-resource"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resource"] == RESOURCE
    assert payload["authorization_servers"] == [
        "https://testserver/api/integrations/bundles/tenant-a/project-a/"
        "connection-hub@1-0/public/oauth"
    ]
    assert payload["bearer_methods_supported"] == ["header"]
    assert set(payload["kdcube_management_operations"]) == {
        "kdcube.management.deployment.inspect",
        "kdcube.management.application.surfaces.read",
        RELOAD_OPERATION,
    }


def test_metadata_uses_public_https_origin_behind_tunnel(monkeypatch) -> None:
    response = _client(monkeypatch).get(
        "/api/integrations/management/v1/.well-known/oauth-protected-resource",
        headers={
            "Host": "runtime.example.test",
            "X-Forwarded-Host": "public.example.test",
            "X-Forwarded-Proto": "http",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["authorization_servers"] == [
        "https://public.example.test/api/integrations/bundles/tenant-a/project-a/"
        "connection-hub@1-0/public/oauth"
    ]


def test_metadata_preserves_explicit_local_http_origin(monkeypatch) -> None:
    response = _client(monkeypatch).get(
        "/api/integrations/management/v1/.well-known/oauth-protected-resource",
        headers={"Host": "localhost:5173", "X-Forwarded-Proto": "http"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["authorization_servers"] == [
        "http://localhost:5173/api/integrations/bundles/tenant-a/project-a/"
        "connection-hub@1-0/public/oauth"
    ]


def test_missing_bearer_points_to_protected_resource_metadata(monkeypatch) -> None:
    response = _client(monkeypatch).get(
        "/api/integrations/management/v1/deployment",
        headers={"Idempotency-Key": "inspect-1"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "delegated_bearer_missing"
    assert response.headers["www-authenticate"] == (
        'Bearer resource_metadata="https://testserver/api/integrations/'
        'management/v1/.well-known/oauth-protected-resource"'
    )


def test_invalid_idempotency_and_request_shapes_fail_before_service(monkeypatch) -> None:
    client = _client(monkeypatch)
    headers = {"Authorization": "Bearer opaque-token"}

    invalid_key = client.get(
        "/api/integrations/management/v1/deployment",
        headers={**headers, "Idempotency-Key": "contains space"},
    )
    inspect_body = client.request(
        "GET",
        "/api/integrations/management/v1/deployment",
        headers={**headers, "Idempotency-Key": "inspect-1"},
        content=b"{}",
    )
    invalid_app = client.post(
        "/api/integrations/management/v1/applications/%2A/reload",
        headers={**headers, "Idempotency-Key": "reload-1"},
        json={},
    )
    invalid_reload = client.post(
        "/api/integrations/management/v1/applications/app-a@1-0/reload",
        headers={**headers, "Idempotency-Key": "reload-1"},
        json={"all": True},
    )

    assert invalid_key.status_code == 400
    assert invalid_key.json()["error"]["code"] == "idempotency_key_invalid"
    assert inspect_body.status_code == 400
    assert inspect_body.json()["error"]["code"] == "request_body_not_allowed"
    assert invalid_app.status_code == 400
    assert invalid_app.json()["error"]["code"] == "application_id_invalid"
    assert invalid_reload.status_code == 400
    assert invalid_reload.json()["error"]["code"] == "reload_request_invalid"


def test_reload_route_passes_one_exact_application_and_empty_body(
    monkeypatch,
) -> None:
    client = _client(monkeypatch)
    calls: list[dict[str, Any]] = []

    class _Service:
        async def execute(self, **kwargs: Any) -> ManagementResponse:
            calls.append(kwargs)
            return ManagementResponse(
                200,
                {
                    "ok": True,
                    "operation": kwargs["operation"],
                    "application_id": kwargs["application_id"],
                },
            )

    async def _service(_request):
        return _Service(), RESOURCE

    monkeypatch.setattr(routes, "_service", _service)
    response = client.post(
        "/api/integrations/management/v1/applications/app-a@1-0/reload",
        headers={
            "Authorization": "Bearer opaque-token",
            "Idempotency-Key": "reload-1",
        },
        json={},
    )

    assert response.status_code == 200
    assert calls == [
        {
            "operation": RELOAD_OPERATION,
            "delegated_bearer": "opaque-token",
            "invocation_id": "reload-1",
            "application_id": "app-a@1-0",
            "body": {},
        }
    ]
