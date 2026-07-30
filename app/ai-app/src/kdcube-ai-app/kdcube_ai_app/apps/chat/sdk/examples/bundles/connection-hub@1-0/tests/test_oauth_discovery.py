from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.requests import Request

from kdcube_ai_app.apps.chat.sdk.runtime.dynamic_module_loader import (
    load_dynamic_module_for_path,
)


def _load_entrypoint_module():
    bundle_root = Path(__file__).resolve().parents[1]
    _module_name, module = load_dynamic_module_for_path(bundle_root / "entrypoint.py")
    return module


def _request(*, method: str = "GET") -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "https",
            "path": (
                "/api/integrations/bundles/tenant-a/project-a/"
                "connection-hub@1-0/public/oauth/"
                ".well-known/oauth-authorization-server"
            ),
            "raw_path": b"",
            "query_string": b"",
            "headers": [(b"host", b"runtime.example.test")],
            "client": ("127.0.0.1", 12345),
            "server": ("runtime.example.test", 443),
        }
    )


@pytest.mark.asyncio
async def test_connection_hub_discovery_advertises_enabled_client_registration_modes():
    module = _load_entrypoint_module()
    entrypoint = module.ConnectionHubEntrypoint.__new__(module.ConnectionHubEntrypoint)
    entrypoint.bundle_props = {
        "connections": {
            "delegated_credentials": {
                "oauth": {
                    "enabled": True,
                    "dynamic_client_registration": {"enabled": False},
                    "client_id_metadata_documents": {"enabled": True},
                }
            }
        }
    }
    entrypoint.runtime_identity = lambda: {"tenant": "tenant-a", "project": "project-a"}

    response = await entrypoint.oauth_get(
        request=_request(),
        path_tail=".well-known/oauth-authorization-server",
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["client_id_metadata_document_supported"] is True
    assert "registration_endpoint" not in payload
    assert payload["revocation_endpoint"] == (
        "https://runtime.example.test/api/integrations/bundles/tenant-a/project-a/"
        "connection-hub@1-0/public/oauth/revoke"
    )


@pytest.mark.asyncio
async def test_connection_hub_dispatches_the_advertised_revocation_route(monkeypatch):
    module = _load_entrypoint_module()
    entrypoint = module.ConnectionHubEntrypoint.__new__(module.ConnectionHubEntrypoint)
    entrypoint.bundle_props = {
        "connections": {
            "delegated_credentials": {"oauth": {"enabled": True}}
        }
    }
    entrypoint.runtime_identity = lambda: {"tenant": "tenant-a", "project": "project-a"}

    async def _revoke(request):
        return module.JSONResponse({"routed": True})

    monkeypatch.setattr(module, "oauth_revoke", _revoke)
    response = await entrypoint.oauth_post(
        request=_request(method="POST"),
        path_tail="revoke",
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"routed": True}
