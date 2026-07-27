from __future__ import annotations

import inspect
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    ConnectedAccountCredential,
)
from kdcube_ai_app.apps.chat.sdk.runtime.dynamic_module_loader import (
    load_dynamic_module_for_path,
)
from kdcube_ai_app.infra.plugin.bundle_loader import BUNDLE_VENV_ATTR


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = BUNDLE_ROOT / "services" / "productivity" / "google_sheets.py"


def _load_module():
    _name, module = load_dynamic_module_for_path(MODULE_PATH)
    return module


def test_google_sheets_venv_boundary_is_async_and_bundle_owned() -> None:
    module = _load_module()
    operation = module._execute_google_sheets_in_venv

    assert inspect.iscoroutinefunction(operation)
    declaration = getattr(operation, BUNDLE_VENV_ATTR)
    assert declaration["requirements"] == "requirements.txt"
    assert declaration["timeout_seconds"] == 120
    assert (BUNDLE_ROOT / declaration["requirements"]).is_file()


async def test_google_sheets_service_keeps_bearer_inside_boundary(monkeypatch) -> None:
    module = _load_module()
    credential = ConnectedAccountCredential(
        ok=True,
        access_token="provider-secret-token",
        account_id="google-account-1",
        provider_id="google",
        connector_app_id="gmail",
        claim="sheets:read",
        tool_name="productivity.sheets.read",
    )
    captured = {}

    async def fake_credential(**_kwargs):
        return credential

    async def fake_venv(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "error": None,
            "ret": {"ranges": [{"range": "Data!A1:B2", "values": [["A", 1]]}]},
        }

    service = module.GoogleSheetsService()
    monkeypatch.setattr(service, "_credential", fake_credential)
    monkeypatch.setattr(module, "_execute_google_sheets_in_venv", fake_venv)

    result = await service.execute(
        operation="read",
        claim="sheets:read",
        tool_name="productivity.sheets.read",
        payload={"spreadsheet_ref": "sheet-123", "ranges": ["Data!A1:B2"]},
    )

    assert result["ok"] is True
    assert result["ret"]["account_id"] == "google-account-1"
    assert captured["access_token"] == "provider-secret-token"
    assert "provider-secret-token" not in str(result)


async def test_google_sheets_provider_auth_failure_uses_connected_account_retry(
    monkeypatch,
) -> None:
    module = _load_module()
    credential = ConnectedAccountCredential(
        ok=True,
        access_token="expired-provider-token",
        account_id="google-account-1",
        provider_id="google",
        connector_app_id="gmail",
        claim="sheets:read",
        tool_name="productivity.sheets.describe",
    )

    async def fake_credential(**_kwargs):
        return credential

    async def fake_venv(**_kwargs):
        return {
            "ok": False,
            "error": {
                "code": "google_sheets_authorization_failed",
                "message": "Google rejected the credential.",
                "provider_status": 401,
            },
            "ret": None,
        }

    service = module.GoogleSheetsService()
    monkeypatch.setattr(service, "_credential", fake_credential)
    monkeypatch.setattr(module, "_execute_google_sheets_in_venv", fake_venv)

    result = await service._execute_once(
        operation="describe",
        claim="sheets:read",
        tool_name="productivity.sheets.describe",
        payload={"spreadsheet_ref": "sheet-123"},
        account_id="",
        where="google_sheets.describe",
    )

    marker = result["__connected_account_auth_failure__"]
    assert marker["credential"] == credential
    assert "expired-provider-token" not in marker["message"]


async def test_google_sheets_multi_claim_resolution_stays_on_one_account(
    monkeypatch,
) -> None:
    module = _load_module()
    calls = []

    async def fake_resolve(_source, **kwargs):
        calls.append(dict(kwargs))
        return ConnectedAccountCredential(
            ok=True,
            access_token=f"token-for-{kwargs['claim']}",
            account_id="google-account-1",
            provider_id="google",
            connector_app_id="gmail",
            claim=kwargs["claim"],
            tool_name=kwargs["tool_name"],
        )

    monkeypatch.setattr(module, "resolve_connected_account_claim", fake_resolve)
    monkeypatch.setattr(module, "resolve_connector_app_id", lambda _provider: "gmail")

    credential = await module.GoogleSheetsService()._credential(
        claim=("sheets:read", "sheets:write"),
        tool_name="named_services.sheets.object.upsert",
        account_id="",
    )

    assert credential.claim == "sheets:write"
    assert [call["claim"] for call in calls] == ["sheets:read", "sheets:write"]
    assert [call["account_id"] for call in calls] == ["", "google-account-1"]
