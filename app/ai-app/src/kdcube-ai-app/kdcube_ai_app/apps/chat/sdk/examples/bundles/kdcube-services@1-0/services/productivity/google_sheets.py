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
from kdcube_ai_app.apps.chat.sdk.integrations.file_delivery import (
    resolve_connected_account_access_token,
)
from kdcube_ai_app.apps.chat.sdk.integrations.sheets.named_service import (
    SHEETS_NAMESPACE,
    make_sheets_named_service_provider,
    parse_sheets_ref,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connector_app_resolution import (
    resolve_connector_app_id,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceContext,
    NamedServiceRequest,
    NamedServiceStreamResult,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    OBJECT_GET,
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
        return await self._execute_with_access_token(
            operation=operation,
            access_token=credential.access_token,
            payload=payload,
            account_id=credential.account_id,
            where=where,
            credential=credential,
        )

    async def _execute_with_access_token(
        self,
        *,
        operation: str,
        access_token: str,
        payload: Mapping[str, Any],
        account_id: str,
        where: str,
        credential: ConnectedAccountCredential | None,
    ) -> dict[str, Any]:
        try:
            result = await _execute_google_sheets_in_venv(
                operation=operation,
                access_token=access_token,
                payload=dict(payload or {}),
            )
        except Exception as exc:
            LOGGER.exception(
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
        error_code = str(
            error_map.get("code") or "google_sheets_provider_error"
        )
        if not bool(result.get("ok")):
            LOGGER.error(
                "Google Sheets provider operation failed "
                "operation=%s code=%s status=%s provider_code=%s "
                "provider_reason=%s stage=%s retryable=%s "
                "outcome_unknown=%s partial_result=%r diagnostics=%r",
                operation,
                error_code,
                provider_status,
                error_map.get("provider_code") or "",
                error_map.get("provider_reason") or "",
                error_map.get("stage") or operation,
                bool(error_map.get("retryable")),
                bool(error_map.get("outcome_unknown")),
                error_map.get("partial_result") or {},
                error_map.get("_diagnostics") or {},
            )
        ret = {
            "provider": str(error_map.get("provider") or "google"),
            "operation": str(error_map.get("operation") or operation),
            "category": str(
                error_map.get("category")
                or error_code.removeprefix("google_sheets_")
                or "provider_error"
            ),
            "outcome_unknown": bool(error_map.get("outcome_unknown")),
            "provider_status": provider_status,
            "provider_code": str(error_map.get("provider_code") or ""),
            "provider_reason": str(error_map.get("provider_reason") or ""),
            "retryable": bool(error_map.get("retryable")),
            "stage": str(error_map.get("stage") or operation),
        }
        partial_result = error_map.get("partial_result")
        if isinstance(partial_result, Mapping) and partial_result:
            ret["partial_result"] = dict(partial_result)
        if error_code == "google_sheets_authorization_failed":
            message = str(
                error_map.get("message")
                or "Google rejected the connected credential."
            )
            if "partial_result" in ret:
                ret["reconnect_required"] = True
                return _error_result(
                    code=error_code,
                    message=message,
                    where=where,
                    ret=ret,
                )
            if credential is not None:
                return connected_account_auth_failure(credential, message)
            return _error_result(
                code="google_sheets_authorization_failed",
                message=message,
                where=where,
                ret=ret,
            )
        if not bool(result.get("ok")):
            return _error_result(
                code=error_code,
                message=str(
                    error_map.get("message") or "Google Sheets operation failed."
                ),
                where=where,
                ret=ret,
            )
        ret = result.get("ret")
        normalized = dict(ret or {}) if isinstance(ret, Mapping) else {"value": ret}
        normalized["account_id"] = str(account_id or "").strip()
        return {"ok": True, "error": None, "ret": normalized}


async def fetch_google_sheets_snapshot(
    entrypoint: Any,
    *,
    user_id: str,
    tenant: str,
    project: str,
    object_ref: str,
    bundle_id: str,
) -> dict[str, Any]:
    """Resolve a complete Sheets snapshot for one signed object URL."""
    try:
        parsed = parse_sheets_ref(object_ref)
    except ValueError as exc:
        return {
            "ok": False,
            "status": 400,
            "error": {"code": "invalid_sheets_ref", "message": str(exc)},
        }
    access_token, failure = await resolve_connected_account_access_token(
        entrypoint,
        user_id=str(user_id or "").strip(),
        tenant=str(tenant or "").strip(),
        project=str(project or "").strip(),
        provider_id=SHEETS_PROVIDER_ID,
        connector_app_id=resolve_connector_app_id(SHEETS_PROVIDER_ID),
        claim=SHEETS_READ_CLAIM,
        account_id=str(parsed.get("account_id") or "").strip(),
    )
    if failure is not None:
        return dict(failure)

    service = GoogleSheetsService()

    async def _execute(**kwargs: Any) -> dict[str, Any]:
        return await service._execute_with_access_token(
            operation=str(kwargs.get("operation") or ""),
            access_token=access_token,
            payload=dict(kwargs.get("payload") or {}),
            account_id=str(parsed.get("account_id") or "").strip(),
            where=f"google_sheets.{kwargs.get('operation') or ''}",
            credential=None,
        )

    provider = make_sheets_named_service_provider(
        execute_operation=_execute,
        bundle_id=bundle_id,
    )
    result = await provider.dispatch(
        NamedServiceContext(
            tenant=tenant,
            project=project,
            user_id=user_id,
        ),
        NamedServiceRequest(
            operation=OBJECT_GET,
            namespace=SHEETS_NAMESPACE,
            object_ref=object_ref,
            response_mode="stream",
            context={"source": "signed_download", "materialize": True},
        ),
    )
    if not isinstance(result, NamedServiceStreamResult):
        response = result
        error = response.error.to_dict() if response.error is not None else {}
        return {
            "ok": False,
            "status": int(response.status or 500),
            "error": error or {
                "code": "sheets_snapshot_unavailable",
                "message": "The spreadsheet snapshot could not be produced.",
            },
        }
    return {
        "ok": True,
        "chunks": result.chunks,
        "filename": result.filename or "spreadsheet.sheets.json",
        "mime_type": result.media_type or "application/json",
        "headers": dict(result.headers or {}),
        "status": int(result.status_code or 200),
    }


__all__ = [
    "GoogleSheetsService",
    "SHEETS_PROVIDER_ID",
    "SHEETS_READ_CLAIM",
    "SHEETS_WRITE_CLAIM",
    "bind_service",
    "fetch_google_sheets_snapshot",
]
