"""Connection Hub backed Google Sheets service for the productivity MCP door."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    ConnectedAccountCredential,
    connected_account_auth_failure,
    resolve_connected_account_claim,
    run_with_connected_account_retry,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.connector_app_resolution import (
    resolve_connector_app_id,
)
from kdcube_ai_app.infra.plugin.bundle_loader import venv


LOGGER = logging.getLogger("kdcube.services.productivity.google_sheets")

SHEETS_PROVIDER_ID = "google"
SHEETS_READ_CLAIM = "sheets:read"
SHEETS_WRITE_CLAIM = "sheets:write"

_SERVICE: Any = None


def bind_service(service: Any) -> None:
    global _SERVICE
    _SERVICE = service


@venv(requirements="requirements.txt", timeout_seconds=120)
async def _execute_google_sheets_in_venv(
    *,
    operation: str,
    access_token: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from kdcube_ai_app.apps.chat.sdk.integrations.google.sheets_proxy import (
        execute_google_sheets_operation,
    )

    return execute_google_sheets_operation(
        operation=operation,
        access_token=access_token,
        payload=payload,
    )


def _error_result(
    *, code: str, message: str, where: str, ret: Any = None
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": str(code or "google_sheets_error"),
            "message": str(message or "Google Sheets operation failed."),
            "where": where,
            "managed": True,
        },
        "ret": ret,
    }


def _provider_status(error: Mapping[str, Any]) -> int:
    try:
        return int(error.get("provider_status") or 0)
    except (TypeError, ValueError):
        return 0


class GoogleSheetsService:
    async def _credential(
        self,
        *,
        claim: str | Sequence[str],
        tool_name: str,
        account_id: str,
    ) -> ConnectedAccountCredential:
        claims = (
            [str(item).strip() for item in claim if str(item or "").strip()]
            if not isinstance(claim, str)
            else [claim.strip()]
        )
        if not claims:
            raise ValueError("At least one Google Sheets claim is required.")
        selected_account_id = str(account_id or "").strip()
        credential: ConnectedAccountCredential | None = None
        for required_claim in claims:
            credential = await resolve_connected_account_claim(
                globals(),
                provider_id=SHEETS_PROVIDER_ID,
                connector_app_id=resolve_connector_app_id(SHEETS_PROVIDER_ID),
                claim=required_claim,
                account_id=selected_account_id,
                tool_name=tool_name,
            )
            if not credential.ok:
                return credential
            selected_account_id = credential.account_id or selected_account_id
        assert credential is not None
        return credential

    async def execute(
        self,
        *,
        operation: str,
        claim: str | Sequence[str],
        tool_name: str,
        payload: Mapping[str, Any] | None = None,
        account_id: str = "",
    ) -> dict[str, Any]:
        where = f"google_sheets.{operation}"
        return await run_with_connected_account_retry(
            globals(),
            where=where,
            run=lambda: self._execute_once(
                operation=operation,
                claim=claim,
                tool_name=tool_name,
                payload=dict(payload or {}),
                account_id=str(account_id or "").strip(),
                where=where,
            ),
        )

    async def _execute_once(
        self,
        *,
        operation: str,
        claim: str | Sequence[str],
        tool_name: str,
        payload: dict[str, Any],
        account_id: str,
        where: str,
    ) -> dict[str, Any]:
        credential = await self._credential(
            claim=claim,
            tool_name=tool_name,
            account_id=account_id,
        )
        if not credential.ok:
            return credential.error_envelope(where=where)
        if not credential.access_token:
            return _error_result(
                code="credential_missing_access_token",
                message="The connected Google credential has no access token.",
                where=where,
            )
        try:
            result = await _execute_google_sheets_in_venv(
                operation=operation,
                access_token=credential.access_token,
                payload=payload,
            )
        except Exception as exc:
            LOGGER.error(
                "Google Sheets venv operation failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return _error_result(
                code="google_sheets_runtime_error",
                message="The Google Sheets dependency runtime failed.",
                where=where,
            )
        if not isinstance(result, Mapping):
            return _error_result(
                code="google_sheets_invalid_result",
                message="The Google Sheets provider returned an invalid result.",
                where=where,
            )
        error = result.get("error")
        error_map = dict(error or {}) if isinstance(error, Mapping) else {}
        provider_status = _provider_status(error_map)
        if provider_status in {401, 403}:
            return connected_account_auth_failure(
                credential,
                str(
                    error_map.get("message")
                    or "Google rejected the connected credential."
                ),
            )
        if not bool(result.get("ok")):
            return _error_result(
                code=str(error_map.get("code") or "google_sheets_provider_error"),
                message=str(
                    error_map.get("message") or "Google Sheets operation failed."
                ),
                where=where,
                ret={
                    "outcome_unknown": bool(error_map.get("outcome_unknown")),
                    "provider_status": provider_status,
                },
            )
        ret = result.get("ret")
        normalized = dict(ret or {}) if isinstance(ret, Mapping) else {"value": ret}
        normalized["account_id"] = credential.account_id
        return {"ok": True, "error": None, "ret": normalized}


__all__ = [
    "GoogleSheetsService",
    "SHEETS_PROVIDER_ID",
    "SHEETS_READ_CLAIM",
    "SHEETS_WRITE_CLAIM",
    "bind_service",
]
