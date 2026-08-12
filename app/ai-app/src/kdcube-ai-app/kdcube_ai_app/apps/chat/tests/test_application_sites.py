from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from starlette.requests import Request

from kdcube_ai_app.apps.chat.proc.rest.integrations import integrations
from kdcube_ai_app.apps.chat.sdk.solutions.sites import (
    ApplicationSite,
    ApplicationSiteTarget,
    compile_application_site_catalog,
)


_SITE_TARGET = ApplicationSiteTarget(
    path="/applications/website@1",
    module=None,
    singleton=False,
)


def _request(*, host: str = "runtime.example.com") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [(b"host", host.encode("utf-8"))],
            "scheme": "https",
            "server": (host, 443),
            "client": ("127.0.0.1", 12345),
            "http_version": "1.1",
        }
    )


@pytest.mark.asyncio
async def test_site_alias_delegates_to_standard_static_serving(monkeypatch) -> None:
    sites = [ApplicationSite("website@1", "docs", False, ("docs.example.com",), _SITE_TARGET)]
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=sites,
    )
    captured = {}

    async def _catalog(_request):
        return catalog

    async def _serve_static_asset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)
    monkeypatch.setattr(integrations, "serve_static_asset", _serve_static_asset)

    response = await integrations._serve_application_site(
        request=_request(),
        site_alias="docs",
        path="guide/getting-started",
    )

    assert response.status_code == 200
    assert captured["tenant"] == "tenant-a"
    assert captured["project"] == "project-a"
    assert captured["bundle_id"] == "website@1"
    assert captured["path"] == "guide/getting-started"
    assert captured["base_href"] == "/sites/docs/"
    assert captured["html_context"]["application_id"] == "website@1"
    assert captured["html_context"]["catalog_revision"] == catalog.revision
    assert captured["resolved_spec"].id == "website@1"
    assert captured["resolved_spec"].path == _SITE_TARGET.path


@pytest.mark.asyncio
async def test_root_selects_site_by_forwarded_host(monkeypatch) -> None:
    sites = [ApplicationSite("website@1", "docs", False, ("docs.example.com",), _SITE_TARGET)]
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=sites,
    )
    captured = {}

    async def _catalog(_request):
        return catalog

    async def _serve_static_asset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status_code=200)

    request = _request(host="proxy.internal")
    request.scope["headers"].append((b"x-forwarded-host", b"docs.example.com"))
    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)
    monkeypatch.setattr(integrations, "serve_static_asset", _serve_static_asset)

    await integrations._serve_application_site(request=request, site_alias="")

    assert captured["bundle_id"] == "website@1"
    assert captured["base_href"] == "/"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_prefix", "expected_location"),
    [
        ("/platform", "/platform/chat"),
        ("/control/ui", "/control/ui/chat"),
    ],
)
async def test_root_without_site_redirects_to_configured_platform(
    monkeypatch,
    route_prefix: str,
    expected_location: str,
) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
    )

    async def _catalog(_request):
        return catalog

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)
    monkeypatch.setattr(
        integrations,
        "get_settings",
        lambda: SimpleNamespace(plain=lambda key: route_prefix if key == "proxy.route_prefix" else None),
    )

    response = await integrations.application_site_landing(request=_request())

    assert isinstance(response, RedirectResponse)
    assert response.status_code == 307
    assert response.headers["location"] == expected_location


@pytest.mark.asyncio
async def test_clean_path_without_site_returns_controlled_404(monkeypatch) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
    )

    async def _catalog(_request):
        return catalog

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)

    with pytest.raises(HTTPException) as exc_info:
        await integrations.application_site_landing_path(
            path="docs/page",
            request=_request(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Application site path not found"


@pytest.mark.asyncio
@pytest.mark.parametrize("route_prefix", [None, ""])
async def test_root_without_site_uses_platform_fallback_when_prefix_is_missing(
    monkeypatch,
    route_prefix,
) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
    )

    async def _catalog(_request):
        return catalog

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)
    monkeypatch.setattr(
        integrations,
        "get_settings",
        lambda: SimpleNamespace(plain=lambda key: route_prefix if key == "proxy.route_prefix" else None),
    )

    response = await integrations.application_site_landing(request=_request())

    assert isinstance(response, RedirectResponse)
    assert response.status_code == 307
    assert response.headers["location"] == "/platform/chat"


@pytest.mark.asyncio
async def test_default_site_serves_root_when_no_host_match(monkeypatch) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[ApplicationSite("site@1", "default", True, (), _SITE_TARGET)],
    )
    captured = {}

    async def _catalog(_request):
        return catalog

    async def _serve_static_asset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)
    monkeypatch.setattr(integrations, "serve_static_asset", _serve_static_asset)

    response = await integrations.application_site_landing(request=_request(host="unknown.example.test"))

    assert response.status_code == 200
    assert captured["bundle_id"] == "site@1"
    assert captured["path"] == "index.html"
    assert captured["base_href"] == "/"


@pytest.mark.asyncio
async def test_wildcard_host_selects_site_for_nested_asset(monkeypatch) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[
            ApplicationSite(
                "site@1",
                "site",
                False,
                ("*.example.test",),
                _SITE_TARGET,
            )
        ],
    )
    captured = {}

    async def _catalog(_request):
        return catalog

    async def _serve_static_asset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)
    monkeypatch.setattr(integrations, "serve_static_asset", _serve_static_asset)

    response = await integrations.application_site_landing_path(
        path="assets/site.css",
        request=_request(host="docs.example.test"),
    )

    assert response.status_code == 200
    assert captured["bundle_id"] == "site@1"
    assert captured["path"] == "assets/site.css"
    assert captured["base_href"] == "/"


@pytest.mark.asyncio
async def test_unknown_alias_returns_controlled_404(monkeypatch) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[ApplicationSite("site@1", "known", False, (), _SITE_TARGET)],
    )

    async def _catalog(_request):
        return catalog

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)

    with pytest.raises(HTTPException) as exc_info:
        await integrations.application_site_root(site_alias="missing", request=_request())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Application site 'missing' not found"


@pytest.mark.asyncio
async def test_missing_site_owner_returns_structured_503(monkeypatch) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[ApplicationSite("site@1", "site", True, ())],
    )

    async def _catalog(_request):
        return catalog

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)

    with pytest.raises(HTTPException) as exc_info:
        await integrations.application_site_landing(request=_request())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Application site owner 'site@1' is unavailable"


@pytest.mark.asyncio
async def test_hot_catalog_lookup_does_not_access_request_redis(monkeypatch) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[ApplicationSite("website@1", "docs", True, ())],
    )
    monkeypatch.setattr(
        integrations.application_site_catalog_runtime,
        "snapshot",
        lambda: catalog,
    )
    monkeypatch.setattr(
        integrations,
        "_get_app_redis",
        lambda _request: pytest.fail("site request attempted to read Redis"),
    )

    restored = await integrations._application_site_catalog(_request())

    assert restored is catalog


@pytest.mark.asyncio
async def test_host_root_path_forwards_requested_file(monkeypatch) -> None:
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[ApplicationSite("website@1", "docs", False, ("docs.example.com",), _SITE_TARGET)],
    )
    captured = {}

    async def _catalog(_request):
        return catalog

    async def _serve_static_asset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(integrations, "_application_site_catalog", _catalog)
    monkeypatch.setattr(integrations, "serve_static_asset", _serve_static_asset)

    response = await integrations.application_site_landing_path(
        path="guide/index.html",
        request=_request(host="docs.example.com"),
    )

    assert response.status_code == 200
    assert captured["path"] == "guide/index.html"
    assert captured["base_href"] == "/"
