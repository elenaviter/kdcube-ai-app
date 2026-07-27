from __future__ import annotations

import inspect
import json
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


async def test_google_sheets_partial_create_auth_failure_is_not_retried(
    monkeypatch,
) -> None:
    module = _load_module()
    credential = ConnectedAccountCredential(
        ok=True,
        access_token="provider-secret-token",
        account_id="google-account-1",
        provider_id="google",
        connector_app_id="gmail",
        claim="sheets:write",
        tool_name="productivity.sheets.create",
    )

    async def fake_credential(**_kwargs):
        return credential

    async def fake_venv(**_kwargs):
        return {
            "ok": False,
            "error": {
                "code": "google_sheets_authorization_failed",
                "message": "Request had insufficient authentication scopes.",
                "provider_status": 403,
                "provider_code": "PERMISSION_DENIED",
                "provider_reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
                "retryable": False,
                "stage": "open_created_spreadsheet",
                "outcome_unknown": False,
                "partial_result": {
                    "resource_created": True,
                    "spreadsheet_id": "created-partial",
                    "completed_stages": ["create_file"],
                },
            },
        }

    service = module.GoogleSheetsService()
    monkeypatch.setattr(service, "_credential", fake_credential)
    monkeypatch.setattr(module, "_execute_google_sheets_in_venv", fake_venv)

    result = await service._execute_once(
        operation="create_spreadsheet",
        claim="sheets:write",
        tool_name="productivity.sheets.create",
        payload={"title": "KDCube Sheets Test"},
        account_id="",
        where="google_sheets.create_spreadsheet",
    )

    assert "__connected_account_auth_failure__" not in result
    assert result["error"]["code"] == "google_sheets_authorization_failed"
    assert result["ret"]["reconnect_required"] is True
    assert result["ret"]["partial_result"]["spreadsheet_id"] == (
        "created-partial"
    )


async def test_google_sheets_provider_failure_is_logged_and_promoted_without_diagnostics(
    monkeypatch,
    caplog,
) -> None:
    module = _load_module()
    credential = ConnectedAccountCredential(
        ok=True,
        access_token="provider-secret-token",
        account_id="google-account-1",
        provider_id="google",
        connector_app_id="gmail",
        claim="sheets:write",
        tool_name="productivity.sheets.create",
    )

    async def fake_credential(**_kwargs):
        return credential

    async def fake_venv(**_kwargs):
        return {
            "ok": False,
            "error": {
                "code": "google_sheets_provider_configuration_error",
                "message": "Google Sheets API is disabled for this project.",
                "provider_status": 403,
                "provider_code": "PERMISSION_DENIED",
                "provider_reason": "SERVICE_DISABLED",
                "retryable": False,
                "stage": "open_created_spreadsheet",
                "outcome_unknown": False,
                "partial_result": {
                    "resource_created": True,
                    "spreadsheet_id": "created-partial",
                    "web_url": (
                        "https://docs.google.com/spreadsheets/d/created-partial/edit"
                    ),
                    "completed_stages": ["create_file"],
                },
                "_diagnostics": {
                    "exception_chain": [
                        {"type": "PermissionError", "message": ""},
                        {"type": "APIError", "message": "provider request failed"},
                    ]
                },
            },
        }

    service = module.GoogleSheetsService()
    monkeypatch.setattr(service, "_credential", fake_credential)
    monkeypatch.setattr(module, "_execute_google_sheets_in_venv", fake_venv)
    caplog.set_level("ERROR", logger=module.LOGGER.name)

    result = await service._execute_once(
        operation="create_spreadsheet",
        claim="sheets:write",
        tool_name="productivity.sheets.create",
        payload={"title": "KDCube Sheets Test"},
        account_id="",
        where="google_sheets.create_spreadsheet",
    )

    assert result["error"] == {
        "code": "google_sheets_provider_configuration_error",
        "message": "Google Sheets API is disabled for this project.",
        "where": "google_sheets.create_spreadsheet",
        "managed": True,
    }
    assert result["ret"] == {
        "outcome_unknown": False,
        "provider_status": 403,
        "provider_code": "PERMISSION_DENIED",
        "provider_reason": "SERVICE_DISABLED",
        "provider": "google",
        "operation": "create_spreadsheet",
        "category": "provider_configuration_error",
        "retryable": False,
        "stage": "open_created_spreadsheet",
        "partial_result": {
            "resource_created": True,
            "spreadsheet_id": "created-partial",
            "web_url": "https://docs.google.com/spreadsheets/d/created-partial/edit",
            "completed_stages": ["create_file"],
        },
    }
    assert "_diagnostics" not in str(result)
    assert "SERVICE_DISABLED" in caplog.text
    assert "APIError" in caplog.text
    assert "provider-secret-token" not in caplog.text


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


async def test_signed_snapshot_re_resolves_user_credential_and_streams_all_values(
    monkeypatch,
) -> None:
    module = _load_module()
    resolutions = []

    async def fake_access_token(_entrypoint, **kwargs):
        resolutions.append(dict(kwargs))
        return "provider-secret-token", None

    async def fake_venv(**kwargs):
        operation = kwargs["operation"]
        if operation == "describe":
            return {
                "ok": True,
                "ret": {
                    "spreadsheet_id": "sheet-123",
                    "title": "KDCube Sheets MCP Test",
                    "tabs": [
                        {
                            "sheet_id": 7,
                            "title": "Data",
                            "sheet_type": "GRID",
                            "row_count": 2,
                            "column_count": 2,
                        }
                    ],
                },
            }
        assert operation == "read"
        return {
            "ok": True,
            "ret": {
                "spreadsheet_id": "sheet-123",
                "ranges": [
                    {
                        "range": "Data!A1:B2",
                        "values": [["Name", "Value"], ["full", 42]],
                    }
                ],
                "range_count": 1,
                "row_count": 2,
                "cell_count": 4,
            },
        }

    monkeypatch.setattr(
        module,
        "resolve_connected_account_access_token",
        fake_access_token,
    )
    monkeypatch.setattr(module, "_execute_google_sheets_in_venv", fake_venv)

    result = await module.fetch_google_sheets_snapshot(
        object(),
        user_id="user-1",
        tenant="demo",
        project="project",
        object_ref="sheets:google:account-1:spreadsheet:sheet-123",
        bundle_id="kdcube-services@1-0",
    )

    assert result["ok"] is True
    chunks = [chunk async for chunk in result["chunks"]]
    body = b"".join(chunks)
    assert chunks
    assert body.index(b'"materialization"') < body.index(b'"values"')
    snapshot = json.loads(body)
    assert snapshot["values"]["ranges"][0]["values"] == [
        ["Name", "Value"],
        ["full", 42],
    ]
    assert [row["claim"] for row in resolutions] == ["sheets:read"]
    assert all(row["user_id"] == "user-1" for row in resolutions)
    assert "provider-secret-token" not in body.decode("utf-8")
