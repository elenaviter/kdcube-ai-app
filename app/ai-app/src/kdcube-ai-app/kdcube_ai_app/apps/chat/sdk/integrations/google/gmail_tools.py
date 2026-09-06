# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Gmail tools backed by Connection Hub connected accounts."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import pathlib
import re
from typing import Annotated, Any

import httpx
import semantic_kernel as sk

try:
    from semantic_kernel.functions import kernel_function
except Exception:
    from semantic_kernel.utils.function_decorator import kernel_function

from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    ConnectedAccountCredential,
    connected_account_auth_failure,
    resolve_connected_account_claim,
    run_with_connected_account_retry,
)
from kdcube_ai_app.apps.chat.sdk.integrations.email.delivery import (
    build_email_message,
    split_email_addresses,
)
from kdcube_ai_app.apps.chat.sdk.integrations.provider_errors import (
    ProviderFailure,
    log_provider_failure,
    provider_failure_from_exception,
    provider_failure_from_response,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import artifact_outdir_for, resolve_artifact_path
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_NAMESPACE_FILES,
    build_physical_artifact_path,
    physical_path_to_logical_path,
    split_logical_artifact_ref,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connector_app_resolution import resolve_connector_app_id


GMAIL_PROVIDER_ID = "google"
GMAIL_READ_CLAIM = "gmail:read"
GMAIL_SEND_CLAIM = "gmail:send"
# Drafts are NOT covered by gmail.send: users.drafts.* needs gmail.compose.
# The claim exists so an automation can prepare mail a person reviews and
# sends themselves, without ever holding send authority.
GMAIL_COMPOSE_CLAIM = "gmail:compose"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_BODY_CHARS = 24000
MAX_DOWNLOAD_ATTACHMENTS = 20
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
# Inline (base64-in-envelope) reads stay small: they travel inside one MCP response.
INLINE_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024

_SERVICE = None
_INTEGRATIONS: dict[str, Any] = {}
LOGGER = logging.getLogger(__name__)


def bind_service(svc: Any) -> None:
    global _SERVICE
    _SERVICE = svc


def bind_integrations(integrations: dict[str, Any] | None) -> None:
    global _INTEGRATIONS
    _INTEGRATIONS = dict(integrations or {})


def _ok_ret_result(ret: Any) -> dict[str, Any]:
    return {"ok": True, "error": None, "ret": ret}


def _error_result(*, code: str, message: str, where: str, ret: Any = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "where": where,
            "managed": True,
        },
        "ret": ret,
    }


def _gmail_failure(
    response: httpx.Response,
    *,
    operation: str,
    fallback: str,
    where: str,
    stage: str = "",
    mutating: bool = False,
    log: bool = True,
) -> ProviderFailure:
    failure = provider_failure_from_response(
        response,
        provider=GMAIL_PROVIDER_ID,
        service="gmail",
        operation=operation,
        fallback=fallback,
        stage=stage,
        mutating=mutating,
    )
    if log:
        log_provider_failure(LOGGER, failure, where=where)
    return failure


def _auth_failure_message(failure: ProviderFailure) -> str:
    reason = failure.provider_reason or failure.provider_code
    if reason and reason.lower() not in failure.message.lower():
        return f"{failure.message} [{reason}]"
    return failure.message


async def _run_provider_call(
    *,
    where: str,
    operation: str,
    run: Any,
    mutating: bool = False,
) -> dict[str, Any]:
    try:
        return await run_with_connected_account_retry(
            globals(),
            where=where,
            run=run,
        )
    except httpx.HTTPError as exc:
        failure = provider_failure_from_exception(
            exc,
            provider=GMAIL_PROVIDER_ID,
            service="gmail",
            operation=operation,
            fallback="Gmail did not return a response.",
            mutating=mutating,
        )
        log_provider_failure(LOGGER, failure, where=where, exc=exc)
        return failure.error_result(where=where)


# Live provider rejections return connected_account_auth_failure(credential,
# message) markers; run_with_connected_account_retry force-refreshes once,
# re-runs the tool body, and only then emits the reconnect envelope.


def _header_map(message: dict[str, Any]) -> dict[str, str]:
    payload = message.get("payload") if isinstance(message, dict) else None
    headers = payload.get("headers") if isinstance(payload, dict) else None
    out: dict[str, str] = {}
    for item in headers or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        if name:
            out[name] = str(item.get("value") or "").strip()
    return out


def _safe_segment(raw: str, *, fallback: str = "item") -> str:
    value = re.sub(r"[^A-Za-z0-9._@ -]+", "_", str(raw or "")).strip(" ._")
    return value or fallback


def _safe_filename(raw: str, *, fallback: str = "attachment.bin") -> str:
    name = pathlib.PurePosixPath(str(raw or "")).name
    return _safe_segment(name, fallback=fallback)


def _jsonish_list(value: Any) -> list[Any]:
    if value in ("", None):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    try:
        parsed = json.loads(str(value))
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _string_list(value: Any) -> list[str]:
    if value in ("", None):
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parsed = _jsonish_list(raw)
        if parsed:
            return [str(item or "").strip() for item in parsed if str(item or "").strip()]
        return [item.strip() for item in re.split(r"[,;\n]+", raw) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    return []


def _decode_b64url(raw: str) -> bytes:
    value = str(raw or "").strip()
    if not value:
        return b""
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _decode_gmail_text(raw: str) -> str:
    try:
        return _decode_b64url(raw).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _walk_parts(part: dict[str, Any] | None):
    if not isinstance(part, dict):
        return
    yield part
    for child in part.get("parts") or []:
        if isinstance(child, dict):
            yield from _walk_parts(child)


def _part_headers(part: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in part.get("headers") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        if name:
            out[name] = str(item.get("value") or "").strip()
    return out


def _body_data(part: dict[str, Any]) -> str:
    body = part.get("body") if isinstance(part, dict) else None
    return str((body or {}).get("data") or "") if isinstance(body, dict) else ""


def _attachment_id(part: dict[str, Any]) -> str:
    body = part.get("body") if isinstance(part, dict) else None
    return str((body or {}).get("attachmentId") or "").strip() if isinstance(body, dict) else ""


def _attachment_size(part: dict[str, Any]) -> int:
    body = part.get("body") if isinstance(part, dict) else None
    try:
        return int((body or {}).get("size") or 0) if isinstance(body, dict) else 0
    except Exception:
        return 0


def _extract_message_content(message: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("payload") if isinstance(message, dict) else None
    headers = _header_map(message)
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict[str, Any]] = []
    inline_attachments: list[dict[str, Any]] = []

    for part in _walk_parts(payload if isinstance(payload, dict) else None) or []:
        mime_type = str(part.get("mimeType") or "").strip()
        filename = str(part.get("filename") or "").strip()
        part_headers = _part_headers(part)
        content_disposition = part_headers.get("content-disposition", "").lower()
        attachment_id = _attachment_id(part)
        data = _body_data(part)
        is_inline = "inline" in content_disposition and "attachment" not in content_disposition
        if filename and attachment_id:
            row = {
                "attachment_id": attachment_id,
                "part_id": str(part.get("partId") or ""),
                "filename": filename,
                "mime_type": mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
                "size_bytes": _attachment_size(part),
                "inline": is_inline,
                "content_id": part_headers.get("content-id", "").strip("<>"),
            }
            (inline_attachments if is_inline else attachments).append(row)
            continue
        if data and mime_type == "text/plain":
            text_parts.append(_decode_gmail_text(data))
        elif data and mime_type == "text/html":
            html_parts.append(_decode_gmail_text(data))

    return {
        "id": str(message.get("id") or ""),
        "thread_id": str(message.get("threadId") or ""),
        "label_ids": list(message.get("labelIds") or []),
        "snippet": str(message.get("snippet") or ""),
        "headers": {
            "subject": headers.get("subject", ""),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "cc": headers.get("cc", ""),
            "date": headers.get("date", ""),
            "message_id": headers.get("message-id", ""),
        },
        "body_text": "\n\n".join(part for part in text_parts if part).strip(),
        "body_html": "\n\n".join(part for part in html_parts if part).strip(),
        "attachments": attachments,
        "inline_attachments": inline_attachments,
        "attachment_count": len(attachments),
        "inline_attachment_count": len(inline_attachments),
    }


def _current_artifact_context() -> tuple[pathlib.Path | None, str]:
    from kdcube_ai_app.apps.chat.sdk.runtime import run_ctx
    from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import get_current_user_identity

    outdir_raw = str(run_ctx.OUTDIR_CV.get("") or "").strip()
    turn_id = str((get_current_user_identity() or {}).get("turn_id") or "").strip()
    if not outdir_raw or not turn_id:
        return None, turn_id
    return artifact_outdir_for(pathlib.Path(outdir_raw), create=True), turn_id


def _resolve_input_artifact(path_value: str, artifact_root: pathlib.Path) -> pathlib.Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    if raw.startswith("conv:fi:"):
        _conversation_id, turn_id, namespace, rel = split_logical_artifact_ref(raw)
        if turn_id and namespace and rel:
            physical = build_physical_artifact_path(turn_id=turn_id, namespace=namespace, relpath=rel)
            return resolve_artifact_path(artifact_root, physical)
        return None
    candidate = pathlib.Path(raw)
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
            resolved.relative_to(artifact_root.resolve())
        except Exception:
            return None
        return resolved if resolved.exists() and resolved.is_file() else None
    return resolve_artifact_path(artifact_root, raw)


def _load_local_attachments(attachment_paths: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _string_list(attachment_paths)
    if not paths:
        return [], []
    artifact_root, _turn_id = _current_artifact_context()
    if artifact_root is None:
        return [], [{
            "code": "artifact_workspace_unavailable",
            "message": "Current ReAct artifact workspace is unavailable; cannot attach local files.",
        }]
    attachments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in paths:
        resolved = _resolve_input_artifact(item, artifact_root)
        if resolved is None or not resolved.exists() or not resolved.is_file():
            errors.append({"code": "attachment_not_found", "path": item, "message": "Attachment path was not found."})
            continue
        data = resolved.read_bytes()
        filename = _safe_filename(resolved.name)
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        attachments.append({"filename": filename, "mime_type": mime_type, "data": data, "source_path": item})
    return attachments, errors


def _load_inline_attachments(value: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attachments carried IN the call: [{filename, content_base64, mime_type?}].

    The `attachment_paths` lane resolves KDCube workspace refs, which only
    exist inside a conversation turn. Automation callers (delegated bearers on
    the productivity door) have no workspace, so drafts accept the bytes
    directly, bounded by the same per-attachment cap."""
    items = _jsonish_list(value)
    attachments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append({
                "code": "attachment_invalid",
                "index": index,
                "message": "Each inline attachment must be an object with filename and content_base64.",
            })
            continue
        filename = _safe_filename(str(item.get("filename") or ""))
        encoded = str(item.get("content_base64") or "").strip()
        if not encoded:
            errors.append({
                "code": "attachment_content_required",
                "index": index,
                "message": f"{filename}: content_base64 is required.",
            })
            continue
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception:
            errors.append({
                "code": "attachment_invalid_base64",
                "index": index,
                "message": f"{filename}: content_base64 is not valid base64.",
            })
            continue
        if len(data) > MAX_ATTACHMENT_BYTES:
            errors.append({
                "code": "attachment_too_large",
                "index": index,
                "message": f"{filename}: {len(data)} bytes exceeds the {MAX_ATTACHMENT_BYTES} byte cap.",
            })
            continue
        mime_type = str(item.get("mime_type") or "").strip() or (
            mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        attachments.append({
            "filename": filename,
            "mime_type": mime_type,
            "data": data,
            "source_path": f"inline:{index}",
        })
    return attachments, errors


async def _get_gmail_message(
    client: httpx.AsyncClient,
    token: str,
    message_id: str,
) -> tuple[dict[str, Any] | None, str, ProviderFailure | None]:
    response = await client.get(
        f"{GMAIL_API}/messages/{message_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"format": "full"},
    )
    if response.status_code >= 400:
        failure = _gmail_failure(
            response,
            operation="get_message",
            fallback="Failed to fetch Gmail message.",
            where="gmail.read_gmail_message",
        )
        return None, failure.message, failure
    try:
        data = response.json()
    except Exception:
        return None, "Gmail message response was not valid JSON.", None
    return data if isinstance(data, dict) else {}, "", None


async def _fetch_gmail_attachment(
    client: httpx.AsyncClient,
    token: str,
    *,
    message_id: str,
    attachment_id: str,
    max_bytes: int | None = MAX_ATTACHMENT_BYTES,
) -> tuple[bytes | None, str, ProviderFailure | None]:
    response = await client.get(
        f"{GMAIL_API}/messages/{message_id}/attachments/{attachment_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if response.status_code >= 400:
        failure = _gmail_failure(
            response,
            operation="get_attachment",
            fallback="Failed to fetch Gmail attachment.",
            where="gmail.download_gmail_attachments",
        )
        return None, failure.message, failure
    try:
        payload = response.json()
        data = _decode_b64url(str((payload or {}).get("data") or ""))
    except Exception:
        return None, "Gmail attachment payload could not be decoded.", None
    if max_bytes is not None and len(data) > max_bytes:
        return None, f"Attachment is larger than the configured limit of {max_bytes} bytes.", None
    return data, "", None


async def _download_gmail_attachments_for_message(
    client: httpx.AsyncClient,
    token: str,
    *,
    message: dict[str, Any],
    credential: ConnectedAccountCredential,
    attachment_ids: list[str] | None = None,
    include_inline: bool = False,
    max_attachments: int = MAX_DOWNLOAD_ATTACHMENTS,
    max_bytes_per_attachment: int = MAX_ATTACHMENT_BYTES,
    visibility: str = "external",
) -> dict[str, Any]:
    artifact_root, turn_id = _current_artifact_context()
    if artifact_root is None or not turn_id:
        return _error_result(
            code="artifact_workspace_unavailable",
            message="Current ReAct turn id or artifact workspace is unavailable.",
            where="gmail.download_gmail_attachments",
        )
    parsed = _extract_message_content(message)
    selected_ids = set(attachment_ids or [])
    rows = list(parsed.get("attachments") or [])
    if include_inline:
        rows.extend(parsed.get("inline_attachments") or [])
    if selected_ids:
        rows = [item for item in rows if str(item.get("attachment_id") or "") in selected_ids]
    rows = rows[: max(1, min(int(max_attachments or MAX_DOWNLOAD_ATTACHMENTS), MAX_DOWNLOAD_ATTACHMENTS))]

    files: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    message_id = str(parsed.get("id") or message.get("id") or "")
    account_key = _safe_segment(str(credential.account_id or "gmail"), fallback="gmail")
    message_key = _safe_segment(message_id, fallback="message")
    for row in rows:
        attachment_id = str(row.get("attachment_id") or "").strip()
        filename = _safe_filename(str(row.get("filename") or "attachment.bin"))
        if not attachment_id:
            continue
        data, err, provider_failure = await _fetch_gmail_attachment(
            client,
            token,
            message_id=message_id,
            attachment_id=attachment_id,
            max_bytes=max_bytes_per_attachment,
        )
        if provider_failure and provider_failure.credential_failure:
            return connected_account_auth_failure(
                credential,
                _auth_failure_message(provider_failure),
            )
        if err or data is None:
            error_payload: dict[str, Any] = {
                "code": (
                    provider_failure.normalized_code
                    if provider_failure
                    else "gmail_attachment_fetch_failed"
                ),
                "message": err or "Attachment fetch failed.",
            }
            if provider_failure:
                error_payload["provider"] = provider_failure.client_ret()
            errors.append({
                "attachment_id": attachment_id,
                "filename": filename,
                "error": error_payload,
            })
            continue
        rel = pathlib.PurePosixPath("gmail-attachments") / account_key / message_key / filename
        physical = build_physical_artifact_path(turn_id=turn_id, namespace=ARTIFACT_NAMESPACE_FILES, relpath=rel.as_posix())
        target = resolve_artifact_path(artifact_root, physical, prefer_existing=False)
        if target.exists():
            stem = pathlib.PurePosixPath(filename).stem or "attachment"
            suffix = pathlib.PurePosixPath(filename).suffix
            rel = pathlib.PurePosixPath("gmail-attachments") / account_key / message_key / f"{stem}-{len(files) + 1}{suffix}"
            physical = build_physical_artifact_path(turn_id=turn_id, namespace=ARTIFACT_NAMESPACE_FILES, relpath=rel.as_posix())
            target = resolve_artifact_path(artifact_root, physical, prefer_existing=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        logical = physical_path_to_logical_path(physical)
        mime_type = str(row.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream")
        files.append(
            {
                "type": "file",
                "kind": "file",
                "source_type": "file",
                "visibility": visibility,
                "artifact_path": logical,
                "logical_path": logical,
                "path": physical,
                "physical_path": physical,
                "filename": filename,
                "mime": mime_type,
                "mime_type": mime_type,
                "size": len(data),
                "size_bytes": len(data),
                "description": f"Gmail attachment from {parsed.get('headers', {}).get('subject') or message_id}",
                "source": {
                    "provider": "gmail",
                    "account_id": credential.account_id,
                    "message_id": message_id,
                    "thread_id": parsed.get("thread_id") or "",
                    "attachment_id": attachment_id,
                    "from": parsed.get("headers", {}).get("from", ""),
                    "to": parsed.get("headers", {}).get("to", ""),
                    "subject": parsed.get("headers", {}).get("subject", ""),
                    "date": parsed.get("headers", {}).get("date", ""),
                },
            }
        )
    return {
        "ok": True,
        "artifact_type": "files",
        "error": None,
        "ret": {
            # The declared-file marker must survive `ret` unwrapping so the
            # ReAct runtime hosts these files as conversation artifacts.
            "artifact_type": "files",
            "message_id": message_id,
            "thread_id": parsed.get("thread_id") or "",
            "account_id": credential.account_id,
            "file_count": len(files),
            "files": files,
            "errors": errors,
            "usage": {
                "read": "Use returned logical_path values with react.read.",
                "send": "Pass returned logical_path or physical_path values to gmail.send_gmail attachment_paths.",
                "deliver": "External files are rehosted as normal conversation artifacts by the ReAct runtime.",
            },
        },
        "files": files,
        "file_count": len(files),
        "errors": errors,
    }


class GmailTools:
    async def _credential(
        self,
        *,
        claim: str,
        tool_name: str,
        account_id: str = "",
    ) -> ConnectedAccountCredential:
        return await resolve_connected_account_claim(
            globals(),
            provider_id=GMAIL_PROVIDER_ID,
            connector_app_id=resolve_connector_app_id(GMAIL_PROVIDER_ID),
            claim=claim,
            account_id=account_id,
            tool_name=tool_name,
        )

    async def _profile_email(
        self,
        client: httpx.AsyncClient,
        token: str,
    ) -> tuple[str, ProviderFailure | None]:
        response = await client.get(
            f"{GMAIL_API}/profile",
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code >= 400:
            return "", _gmail_failure(
                response,
                operation="get_profile",
                fallback="Failed to resolve the Gmail profile.",
                where="gmail.get_profile",
            )
        try:
            data = response.json()
        except Exception:
            return "", None
        return (
            str(data.get("emailAddress") or "").strip()
            if isinstance(data, dict)
            else "",
            None,
        )

    @kernel_function(
        name="search_gmail",
        description=(
            "Search the current user's connected Gmail account. "
            "Requires the user to connect Gmail with the gmail:read claim in Connection Hub. "
            "Returns {ok, error, ret}; ret contains message ids, subjects, senders, dates, snippets, and thread ids."
        ),
    )
    async def search_gmail(
        self,
        query: Annotated[str, "Gmail search query, for example 'from:alice@example.com newer_than:7d'."] = "",
        max_results: Annotated[int, "Maximum messages to return, 1-10.", {"min": 1, "max": 10}] = 5,
        cursor: Annotated[str, "Optional Gmail pagination cursor returned by the previous search."] = "",
        account_id: Annotated[str, "Optional connected account id when the user has several Gmail accounts."] = "",
    ) -> Annotated[dict, "Envelope: {ok, error, ret}."]:
        return await _run_provider_call(
            where="gmail.search_gmail",
            operation="search",
            run=lambda: self._search_gmail(
                query=query,
                max_results=max_results,
                cursor=cursor,
                account_id=account_id,
            ),
        )

    async def _search_gmail(
        self,
        *,
        query: str,
        max_results: int,
        cursor: str,
        account_id: str,
    ) -> dict[str, Any]:
        credential = await self._credential(claim=GMAIL_READ_CLAIM, account_id=account_id, tool_name="gmail.search_gmail")
        if not credential.ok:
            return credential.error_envelope(where="gmail.search_gmail")
        if not credential.access_token:
            return _error_result(
                code="credential_missing_access_token",
                message="Connected Gmail credential has no access token.",
                where="gmail.search_gmail",
            )

        limit = max(1, min(int(max_results or 5), 10))
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {"q": query or "", "maxResults": limit}
            if str(cursor or "").strip():
                params["pageToken"] = str(cursor).strip()
            list_response = await client.get(
                f"{GMAIL_API}/messages",
                headers={"Authorization": f"Bearer {credential.access_token}"},
                params=params,
            )
            if list_response.status_code >= 400:
                failure = _gmail_failure(
                    list_response,
                    operation="search",
                    fallback="Gmail search failed.",
                    where="gmail.search_gmail",
                )
                if failure.credential_failure:
                    return connected_account_auth_failure(
                        credential,
                        _auth_failure_message(failure),
                    )
                return failure.error_result(where="gmail.search_gmail")
            try:
                list_data = list_response.json()
            except Exception:
                list_data = {}
            messages = list_data.get("messages") if isinstance(list_data, dict) else []
            rows: list[dict[str, Any]] = []
            for item in messages or []:
                message_id = str((item or {}).get("id") or "").strip()
                if not message_id:
                    continue
                detail = await client.get(
                    f"{GMAIL_API}/messages/{message_id}",
                    headers={"Authorization": f"Bearer {credential.access_token}"},
                    params=[
                        ("format", "metadata"),
                        ("metadataHeaders", "Subject"),
                        ("metadataHeaders", "From"),
                        ("metadataHeaders", "Date"),
                    ],
                )
                if detail.status_code >= 400:
                    failure = _gmail_failure(
                        detail,
                        operation="get_message_metadata",
                        fallback="Failed to fetch Gmail message metadata.",
                        where="gmail.search_gmail",
                    )
                    if failure.credential_failure:
                        return connected_account_auth_failure(
                            credential,
                            _auth_failure_message(failure),
                        )
                    rows.append(
                        {
                            "id": message_id,
                            "error": {
                                "code": failure.normalized_code,
                                "message": failure.message,
                                "provider": failure.client_ret(),
                            },
                        }
                    )
                    continue
                try:
                    detail_data = detail.json()
                except Exception:
                    detail_data = {}
                headers = _header_map(detail_data)
                rows.append(
                    {
                        "id": message_id,
                        "thread_id": str(detail_data.get("threadId") or ""),
                        "subject": headers.get("subject", ""),
                        "from": headers.get("from", ""),
                        "date": headers.get("date", ""),
                        "snippet": str(detail_data.get("snippet") or ""),
                    }
                )
        return _ok_ret_result({
            "messages": rows,
            "count": len(rows),
            "next_cursor": str(
                (list_data.get("nextPageToken") or "")
                if isinstance(list_data, dict)
                else ""
            ),
            "account_id": credential.account_id,
        })

    @kernel_function(
        name="read_gmail_message",
        description=(
            "Read one Gmail message body and attachment metadata from the current user's connected Gmail account. "
            "Requires gmail:read. Use search_gmail first to get the message id. "
            "Returns {ok, error, ret}; ret contains headers, body text/html, and attachment ids."
        ),
    )
    async def read_gmail_message(
        self,
        message_id: Annotated[str, "Gmail message id returned by search_gmail."],
        include_html: Annotated[bool, "Include HTML body in the result."] = False,
        max_body_chars: Annotated[int, "Maximum body characters to return, 1000-24000.", {"min": 1000, "max": MAX_BODY_CHARS}] = 12000,
        account_id: Annotated[str, "Optional connected account id when the user has several Gmail accounts."] = "",
    ) -> Annotated[dict, "Envelope: {ok, error, ret}."]:
        return await _run_provider_call(
            where="gmail.read_gmail_message",
            operation="get_message",
            run=lambda: self._read_gmail_message(
                message_id=message_id,
                include_html=include_html,
                max_body_chars=max_body_chars,
                account_id=account_id,
            ),
        )

    async def _read_gmail_message(
        self,
        *,
        message_id: str,
        include_html: bool,
        max_body_chars: int,
        account_id: str,
    ) -> dict[str, Any]:
        msg_id = str(message_id or "").strip()
        if not msg_id:
            return _error_result(
                code="message_id_required",
                message="message_id is required.",
                where="gmail.read_gmail_message",
            )
        credential = await self._credential(claim=GMAIL_READ_CLAIM, account_id=account_id, tool_name="gmail.read_gmail_message")
        if not credential.ok:
            return credential.error_envelope(where="gmail.read_gmail_message")
        if not credential.access_token:
            return _error_result(
                code="credential_missing_access_token",
                message="Connected Gmail credential has no access token.",
                where="gmail.read_gmail_message",
            )
        limit = max(1000, min(int(max_body_chars or 12000), MAX_BODY_CHARS))
        async with httpx.AsyncClient(timeout=30.0) as client:
            message, err, provider_failure = await _get_gmail_message(
                client, credential.access_token, msg_id
            )
        if err or message is None:
            if provider_failure and provider_failure.credential_failure:
                return connected_account_auth_failure(
                    credential,
                    _auth_failure_message(provider_failure),
                )
            if provider_failure:
                return provider_failure.error_result(
                    where="gmail.read_gmail_message"
                )
            return _error_result(
                code="gmail_provider_protocol_error",
                message=err or "Gmail message fetch failed.",
                where="gmail.read_gmail_message",
            )
        parsed = _extract_message_content(message)
        body_text = str(parsed.get("body_text") or "")
        body_html = str(parsed.get("body_html") or "")
        ret = {
            "id": parsed.get("id") or msg_id,
            "thread_id": parsed.get("thread_id") or "",
            "headers": parsed.get("headers") or {},
            "snippet": parsed.get("snippet") or "",
            "body_text": body_text[:limit],
            "body_text_truncated": len(body_text) > limit,
            "attachments": parsed.get("attachments") or [],
            "inline_attachments": parsed.get("inline_attachments") or [],
            "attachment_count": parsed.get("attachment_count") or 0,
            "inline_attachment_count": parsed.get("inline_attachment_count") or 0,
            "account_id": credential.account_id,
            "usage": {
                "download": "Call gmail.download_gmail_attachments with this message id to materialize attachments as KDCube files.",
                "forward": "Call gmail.forward_gmail_message to forward this email.",
            },
        }
        if include_html:
            ret["body_html"] = body_html[:limit]
            ret["body_html_truncated"] = len(body_html) > limit
        return _ok_ret_result(ret)

    async def read_gmail_attachment(
        self,
        *,
        message_id: str,
        attachment_id: str = "",
        part_id: str = "",
        max_bytes: int = INLINE_ATTACHMENT_MAX_BYTES,
        account_id: str = "",
    ) -> dict[str, Any]:
        """One attachment's bytes, base64 in the envelope. For surfaces that run
        outside a ReAct turn (the productivity MCP door) and therefore have no
        artifact workspace to materialize into. Bounded by ``max_bytes`` so a
        JSON-RPC response stays small; ``part_id`` is the stable selector
        (Gmail attachment ids rotate per fetch)."""
        where = "gmail.read_gmail_attachment"
        msg_id = str(message_id or "").strip()
        if not msg_id:
            return _error_result(code="message_id_required", message="message_id is required.", where=where)
        want_part = str(part_id or "").strip()
        want_id = str(attachment_id or "").strip()
        if not want_part and not want_id:
            return _error_result(code="attachment_selector_required", message="part_id or attachment_id is required.", where=where)
        limit = max(1, min(int(max_bytes or INLINE_ATTACHMENT_MAX_BYTES), INLINE_ATTACHMENT_MAX_BYTES))
        credential = await self._credential(claim=GMAIL_READ_CLAIM, account_id=account_id, tool_name=where)
        if not credential.ok:
            return credential.error_envelope(where=where)
        if not credential.access_token:
            return _error_result(code="credential_missing_access_token", message="Connected Gmail credential has no access token.", where=where)
        async with httpx.AsyncClient(timeout=60.0) as client:
            message, err, provider_failure = await _get_gmail_message(client, credential.access_token, msg_id)
            if err or message is None:
                if provider_failure and provider_failure.credential_failure:
                    return connected_account_auth_failure(credential, _auth_failure_message(provider_failure))
                if provider_failure:
                    return provider_failure.error_result(where=where)
                return _error_result(code="gmail_provider_protocol_error", message=err or "Gmail message fetch failed.", where=where)
            parsed = _extract_message_content(message)
            rows = [*(parsed.get("attachments") or []), *(parsed.get("inline_attachments") or [])]
            row = next((r for r in rows if want_part and str(r.get("part_id") or "") == want_part), None) \
                or next((r for r in rows if want_id and str(r.get("attachment_id") or "") == want_id), None)
            if row is None:
                return _error_result(code="gmail_attachment_not_found", message="Attachment was not found on this message.", where=where)
            declared = int(row.get("size_bytes") or 0)
            if declared > limit:
                return _error_result(
                    code="gmail_attachment_too_large",
                    message=f"Attachment is {declared} bytes, above the {limit} byte inline read limit.",
                    where=where,
                    ret={"size_bytes": declared, "max_bytes": limit},
                )
            data, err, provider_failure = await _fetch_gmail_attachment(
                client, credential.access_token, message_id=msg_id,
                attachment_id=str(row.get("attachment_id") or ""), max_bytes=limit,
            )
        if provider_failure and provider_failure.credential_failure:
            return connected_account_auth_failure(credential, _auth_failure_message(provider_failure))
        if err or data is None:
            if provider_failure:
                return provider_failure.error_result(where=where)
            return _error_result(code="gmail_attachment_fetch_failed", message=err or "Attachment fetch failed.", where=where)
        filename = _safe_filename(str(row.get("filename") or "attachment.bin"))
        return _ok_ret_result({
            "message_id": msg_id,
            "part_id": str(row.get("part_id") or ""),
            "filename": filename,
            "mime_type": str(row.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"),
            "size_bytes": len(data),
            "content_base64": base64.b64encode(data).decode("ascii"),
            "account_id": credential.account_id,
        })

    @kernel_function(
        name="download_gmail_attachments",
        description=(
            "Download attachments from one Gmail message into the current KDCube/ReAct artifact workspace. "
            "Requires gmail:read. Use read_gmail_message first to inspect attachment ids. "
            "Returns {ok, error, ret} plus artifact_type=files so downloaded attachments can be read, delivered, or reattached."
        ),
    )
    async def download_gmail_attachments(
        self,
        message_id: Annotated[str, "Gmail message id returned by search_gmail/read_gmail_message."],
        attachment_ids: Annotated[str, "Optional comma/newline/JSON list of attachment ids. Empty downloads all non-inline attachments."] = "",
        include_inline: Annotated[bool, "Also download inline attachments such as embedded images."] = False,
        max_attachments: Annotated[int, "Maximum attachments to download, 1-20.", {"min": 1, "max": MAX_DOWNLOAD_ATTACHMENTS}] = 10,
        max_bytes_per_attachment: Annotated[int, "Maximum bytes per attachment.", {"min": 1, "max": MAX_ATTACHMENT_BYTES}] = MAX_ATTACHMENT_BYTES,
        visibility: Annotated[str, "external for deliverable artifacts, internal for analysis-only artifacts."] = "external",
        account_id: Annotated[str, "Optional connected account id when the user has several Gmail accounts."] = "",
    ) -> Annotated[dict, "Envelope with artifact_type=files and ret.files."]:
        return await _run_provider_call(
            where="gmail.download_gmail_attachments",
            operation="download_attachments",
            run=lambda: self._download_gmail_attachments(
                message_id=message_id,
                attachment_ids=attachment_ids,
                include_inline=include_inline,
                max_attachments=max_attachments,
                max_bytes_per_attachment=max_bytes_per_attachment,
                visibility=visibility,
                account_id=account_id,
            ),
        )

    async def _download_gmail_attachments(
        self,
        *,
        message_id: str,
        attachment_ids: str,
        include_inline: bool,
        max_attachments: int,
        max_bytes_per_attachment: int,
        visibility: str,
        account_id: str,
    ) -> dict[str, Any]:
        msg_id = str(message_id or "").strip()
        if not msg_id:
            return _error_result(
                code="message_id_required",
                message="message_id is required.",
                where="gmail.download_gmail_attachments",
            )
        visibility_norm = str(visibility or "external").strip().lower()
        if visibility_norm not in {"external", "internal"}:
            return _error_result(
                code="invalid_visibility",
                message="visibility must be 'external' or 'internal'.",
                where="gmail.download_gmail_attachments",
            )
        credential = await self._credential(claim=GMAIL_READ_CLAIM, account_id=account_id, tool_name="gmail.download_gmail_attachments")
        if not credential.ok:
            return credential.error_envelope(where="gmail.download_gmail_attachments")
        if not credential.access_token:
            return _error_result(
                code="credential_missing_access_token",
                message="Connected Gmail credential has no access token.",
                where="gmail.download_gmail_attachments",
            )
        async with httpx.AsyncClient(timeout=60.0) as client:
            message, err, provider_failure = await _get_gmail_message(
                client, credential.access_token, msg_id
            )
            if err or message is None:
                if provider_failure and provider_failure.credential_failure:
                    return connected_account_auth_failure(
                        credential,
                        _auth_failure_message(provider_failure),
                    )
                if provider_failure:
                    return provider_failure.error_result(
                        where="gmail.download_gmail_attachments"
                    )
                return _error_result(
                    code="gmail_provider_protocol_error",
                    message=err or "Gmail message fetch failed.",
                    where="gmail.download_gmail_attachments",
                )
            return await _download_gmail_attachments_for_message(
                client,
                credential.access_token,
                message=message,
                credential=credential,
                attachment_ids=_string_list(attachment_ids),
                include_inline=include_inline,
                max_attachments=max_attachments,
                max_bytes_per_attachment=max_bytes_per_attachment,
                visibility=visibility_norm,
            )

    @kernel_function(
        name="send_gmail",
        description=(
            "Send an email through the current user's connected Gmail account. "
            "Requires the user to connect Gmail with the gmail:send claim in Connection Hub. "
            "Can attach KDCube artifact/file paths. Returns {ok, error, ret}; ret contains the Gmail message id and thread id."
        ),
    )
    async def send_gmail(
        self,
        to: Annotated[str, "Comma, semicolon, or newline separated recipient email addresses."],
        subject: Annotated[str, "Email subject."] = "KDCube message",
        body_markdown: Annotated[str, "Markdown body to send as text and HTML."] = "",
        cc: Annotated[str, "Optional comma, semicolon, or newline separated cc recipients."] = "",
        bcc: Annotated[str, "Optional comma, semicolon, or newline separated bcc recipients."] = "",
        body_html: Annotated[str, "Optional complete HTML body. Leave empty when using body_markdown."] = "",
        attachment_paths: Annotated[str, "Optional comma/newline/JSON list of KDCube logical_path or physical_path file refs to attach."] = "",
        account_id: Annotated[str, "Optional connected account id when the user has several Gmail accounts."] = "",
    ) -> Annotated[dict, "Envelope: {ok, error, ret}."]:
        return await _run_provider_call(
            where="gmail.send_gmail",
            operation="send_message",
            mutating=True,
            run=lambda: self._send_gmail(
                to=to,
                subject=subject,
                body_markdown=body_markdown,
                cc=cc,
                bcc=bcc,
                body_html=body_html,
                attachment_paths=attachment_paths,
                account_id=account_id,
            ),
        )

    async def _send_gmail(
        self,
        *,
        to: str,
        subject: str,
        body_markdown: str,
        cc: str,
        bcc: str,
        body_html: str,
        attachment_paths: str,
        account_id: str,
    ) -> dict[str, Any]:
        recipients = split_email_addresses(to)
        if not recipients:
            return _error_result(
                code="recipient_required",
                message="At least one recipient email address is required.",
                where="gmail.send_gmail",
            )
        credential = await self._credential(claim=GMAIL_SEND_CLAIM, account_id=account_id, tool_name="gmail.send_gmail")
        if not credential.ok:
            return credential.error_envelope(where="gmail.send_gmail")
        if not credential.access_token:
            return _error_result(
                code="credential_missing_access_token",
                message="Connected Gmail credential has no access token.",
                where="gmail.send_gmail",
            )
        attachments, attachment_errors = _load_local_attachments(attachment_paths)
        if attachment_errors:
            return _error_result(
                code="attachment_load_failed",
                message="One or more requested attachments could not be loaded.",
                where="gmail.send_gmail",
                ret={"attachment_errors": attachment_errors},
            )

        async with httpx.AsyncClient(timeout=30.0) as client:
            sender_email, provider_failure = await self._profile_email(
                client, credential.access_token
            )
            if not sender_email:
                if provider_failure and provider_failure.credential_failure:
                    return connected_account_auth_failure(
                        credential,
                        _auth_failure_message(provider_failure),
                    )
                if provider_failure:
                    return provider_failure.error_result(where="gmail.send_gmail")
                return _error_result(
                    code="gmail_profile_unavailable",
                    message="Could not resolve the connected Gmail sender address.",
                    where="gmail.send_gmail",
                )
            message = build_email_message(
                sender_email=sender_email,
                recipients=recipients,
                cc=split_email_addresses(cc),
                bcc=split_email_addresses(bcc),
                subject=subject,
                body_text=body_markdown,
                body_html=body_html,
                attachments=attachments,
            )
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
            response = await client.post(
                f"{GMAIL_API}/messages/send",
                headers={"Authorization": f"Bearer {credential.access_token}"},
                json={"raw": raw},
            )
            if response.status_code >= 400:
                failure = _gmail_failure(
                    response,
                    operation="send_message",
                    fallback="Gmail send failed.",
                    where="gmail.send_gmail",
                    mutating=True,
                )
                if failure.credential_failure:
                    return connected_account_auth_failure(
                        credential,
                        _auth_failure_message(failure),
                    )
                return failure.error_result(where="gmail.send_gmail")
            try:
                data = response.json()
            except Exception:
                data = {}
        return _ok_ret_result(
            {
                "id": str(data.get("id") or ""),
                "thread_id": str(data.get("threadId") or ""),
                "sender": sender_email,
                "account_id": credential.account_id,
                "attachment_count": len(attachments),
                "attachments": [
                    {
                        "filename": item.get("filename"),
                        "mime_type": item.get("mime_type"),
                        "source_path": item.get("source_path"),
                    }
                    for item in attachments
                ],
            }
        )

    @kernel_function(
        name="create_gmail_draft",
        description=(
            "Create a DRAFT email in the current user's connected Gmail account without sending it. "
            "The person reviews and sends the draft themselves; this tool never sends. "
            "Requires the gmail:compose claim in Connection Hub (drafts are not covered by gmail:send). "
            "Attachments: KDCube artifact/file paths (conversation lane) and/or inline base64 entries "
            "(automation lane). Returns {ok, error, ret}; ret carries the draft id and message id."
        ),
    )
    async def create_gmail_draft(
        self,
        to: Annotated[str, "Comma, semicolon, or newline separated recipient email addresses. May be empty for a recipientless draft."] = "",
        subject: Annotated[str, "Email subject."] = "",
        body_markdown: Annotated[str, "Markdown body stored as text and HTML."] = "",
        cc: Annotated[str, "Optional comma, semicolon, or newline separated cc recipients."] = "",
        bcc: Annotated[str, "Optional comma, semicolon, or newline separated bcc recipients."] = "",
        body_html: Annotated[str, "Optional complete HTML body. Leave empty when using body_markdown."] = "",
        attachment_paths: Annotated[str, "Optional comma/newline/JSON list of KDCube logical_path or physical_path file refs to attach (conversation lane only)."] = "",
        attachments_base64: Annotated[str, "Optional JSON list of {filename, content_base64, mime_type?} inline attachments (the lane for automation callers with no KDCube workspace)."] = "",
        account_id: Annotated[str, "Optional connected account id when the user has several Gmail accounts."] = "",
    ) -> Annotated[dict, "Envelope: {ok, error, ret}."]:
        return await _run_provider_call(
            where="gmail.create_gmail_draft",
            operation="create_draft",
            mutating=True,
            run=lambda: self._create_gmail_draft(
                to=to,
                subject=subject,
                body_markdown=body_markdown,
                cc=cc,
                bcc=bcc,
                body_html=body_html,
                attachment_paths=attachment_paths,
                attachments_base64=attachments_base64,
                account_id=account_id,
            ),
        )

    async def _create_gmail_draft(
        self,
        *,
        to: str,
        subject: str,
        body_markdown: str,
        cc: str,
        bcc: str,
        body_html: str,
        attachment_paths: str,
        attachments_base64: str,
        account_id: str,
    ) -> dict[str, Any]:
        # A draft may legitimately have no recipient yet; unlike send, nothing
        # is required except a connected account with the compose claim.
        recipients = split_email_addresses(to)
        credential = await self._credential(
            claim=GMAIL_COMPOSE_CLAIM, account_id=account_id,
            tool_name="gmail.create_gmail_draft",
        )
        if not credential.ok:
            return credential.error_envelope(where="gmail.create_gmail_draft")
        if not credential.access_token:
            return _error_result(
                code="credential_missing_access_token",
                message="Connected Gmail credential has no access token.",
                where="gmail.create_gmail_draft",
            )
        attachments: list[dict[str, Any]] = []
        attachment_errors: list[dict[str, Any]] = []
        if str(attachment_paths or "").strip():
            loaded, errors = _load_local_attachments(attachment_paths)
            attachments.extend(loaded)
            attachment_errors.extend(errors)
        inline, inline_errors = _load_inline_attachments(attachments_base64)
        attachments.extend(inline)
        attachment_errors.extend(inline_errors)
        if attachment_errors:
            return _error_result(
                code="attachment_load_failed",
                message="One or more requested attachments could not be loaded.",
                where="gmail.create_gmail_draft",
                ret={"attachment_errors": attachment_errors},
            )

        async with httpx.AsyncClient(timeout=30.0) as client:
            sender_email, provider_failure = await self._profile_email(
                client, credential.access_token
            )
            if not sender_email:
                if provider_failure and provider_failure.credential_failure:
                    return connected_account_auth_failure(
                        credential,
                        _auth_failure_message(provider_failure),
                    )
                if provider_failure:
                    return provider_failure.error_result(where="gmail.create_gmail_draft")
                return _error_result(
                    code="gmail_profile_unavailable",
                    message="Could not resolve the connected Gmail sender address.",
                    where="gmail.create_gmail_draft",
                )
            message = build_email_message(
                sender_email=sender_email,
                recipients=recipients,
                cc=split_email_addresses(cc),
                bcc=split_email_addresses(bcc),
                subject=subject,
                body_text=body_markdown,
                body_html=body_html,
                attachments=attachments,
            )
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
            response = await client.post(
                f"{GMAIL_API}/drafts",
                headers={"Authorization": f"Bearer {credential.access_token}"},
                json={"message": {"raw": raw}},
            )
            if response.status_code >= 400:
                failure = _gmail_failure(
                    response,
                    operation="create_draft",
                    fallback="Gmail draft creation failed.",
                    where="gmail.create_gmail_draft",
                    mutating=True,
                )
                if failure.credential_failure:
                    return connected_account_auth_failure(
                        credential,
                        _auth_failure_message(failure),
                    )
                return failure.error_result(where="gmail.create_gmail_draft")
            try:
                data = response.json()
            except Exception:
                data = {}
        draft_message = data.get("message") if isinstance(data.get("message"), dict) else {}
        return _ok_ret_result(
            {
                "draft_id": str(data.get("id") or ""),
                "message_id": str(draft_message.get("id") or ""),
                "thread_id": str(draft_message.get("threadId") or ""),
                "sender": sender_email,
                "account_id": credential.account_id,
                "recipients": recipients,
                "subject": subject,
                "attachment_count": len(attachments),
                "attachments": [
                    {
                        "filename": item.get("filename"),
                        "mime_type": item.get("mime_type"),
                        "source_path": item.get("source_path"),
                    }
                    for item in attachments
                ],
                "drafts_url": "https://mail.google.com/mail/u/0/#drafts",
            }
        )

    @kernel_function(
        name="forward_gmail_message",
        description=(
            "Forward one Gmail message to recipients through the current user's connected Gmail account. "
            "Requires gmail:read and gmail:send. Can optionally include original Gmail attachments and extra KDCube file refs."
        ),
    )
    async def forward_gmail_message(
        self,
        message_id: Annotated[str, "Gmail message id returned by search_gmail/read_gmail_message."],
        to: Annotated[str, "Comma, semicolon, or newline separated recipient email addresses."],
        note_markdown: Annotated[str, "Optional note to add above the forwarded message."] = "",
        cc: Annotated[str, "Optional comma, semicolon, or newline separated cc recipients."] = "",
        bcc: Annotated[str, "Optional comma, semicolon, or newline separated bcc recipients."] = "",
        include_original_attachments: Annotated[bool, "Attach original non-inline Gmail attachments to the forwarded email."] = False,
        attachment_paths: Annotated[str, "Optional comma/newline/JSON list of additional KDCube file refs to attach."] = "",
        account_id: Annotated[str, "Optional connected account id when the user has several Gmail accounts."] = "",
    ) -> Annotated[dict, "Envelope: {ok, error, ret}."]:
        return await _run_provider_call(
            where="gmail.forward_gmail_message",
            operation="forward_message",
            mutating=True,
            run=lambda: self._forward_gmail_message(
                message_id=message_id,
                to=to,
                note_markdown=note_markdown,
                cc=cc,
                bcc=bcc,
                include_original_attachments=include_original_attachments,
                attachment_paths=attachment_paths,
                account_id=account_id,
            ),
        )

    async def _forward_gmail_message(
        self,
        *,
        message_id: str,
        to: str,
        note_markdown: str,
        cc: str,
        bcc: str,
        include_original_attachments: bool,
        attachment_paths: str,
        account_id: str,
    ) -> dict[str, Any]:
        msg_id = str(message_id or "").strip()
        recipients = split_email_addresses(to)
        if not msg_id:
            return _error_result(
                code="message_id_required",
                message="message_id is required.",
                where="gmail.forward_gmail_message",
            )
        if not recipients:
            return _error_result(
                code="recipient_required",
                message="At least one recipient email address is required.",
                where="gmail.forward_gmail_message",
            )
        read_credential = await self._credential(
            claim=GMAIL_READ_CLAIM,
            account_id=account_id,
            tool_name="gmail.forward_gmail_message",
        )
        if not read_credential.ok:
            return read_credential.error_envelope(where="gmail.forward_gmail_message")
        if not read_credential.access_token:
            return _error_result(
                code="credential_missing_access_token",
                message="Connected Gmail read credential has no access token.",
                where="gmail.forward_gmail_message",
            )
        send_credential = await self._credential(
            claim=GMAIL_SEND_CLAIM,
            account_id=account_id,
            tool_name="gmail.forward_gmail_message",
        )
        if not send_credential.ok:
            return send_credential.error_envelope(where="gmail.forward_gmail_message")
        token = send_credential.access_token or read_credential.access_token
        if not token:
            return _error_result(
                code="credential_missing_access_token",
                message="Connected Gmail credential has no access token.",
                where="gmail.forward_gmail_message",
            )
        local_attachments, attachment_errors = _load_local_attachments(attachment_paths)
        if attachment_errors:
            return _error_result(
                code="attachment_load_failed",
                message="One or more requested attachments could not be loaded.",
                where="gmail.forward_gmail_message",
                ret={"attachment_errors": attachment_errors},
            )

        async with httpx.AsyncClient(timeout=60.0) as client:
            message, err, provider_failure = await _get_gmail_message(
                client, read_credential.access_token, msg_id
            )
            if err or message is None:
                if provider_failure and provider_failure.credential_failure:
                    return connected_account_auth_failure(
                        read_credential,
                        _auth_failure_message(provider_failure),
                    )
                if provider_failure:
                    return provider_failure.error_result(
                        where="gmail.forward_gmail_message"
                    )
                return _error_result(
                    code="gmail_provider_protocol_error",
                    message=err or "Gmail message fetch failed.",
                    where="gmail.forward_gmail_message",
                )
            parsed = _extract_message_content(message)
            headers = parsed.get("headers") or {}
            sender_email, profile_failure = await self._profile_email(client, token)
            if not sender_email:
                if profile_failure and profile_failure.credential_failure:
                    return connected_account_auth_failure(
                        send_credential,
                        _auth_failure_message(profile_failure),
                    )
                if profile_failure:
                    return profile_failure.error_result(
                        where="gmail.forward_gmail_message"
                    )
                return _error_result(
                    code="gmail_profile_unavailable",
                    message="Could not resolve the connected Gmail sender address.",
                    where="gmail.forward_gmail_message",
                )

            original_attachments: list[dict[str, Any]] = []
            original_attachment_errors: list[dict[str, Any]] = []
            if include_original_attachments:
                for row in list(parsed.get("attachments") or [])[:MAX_DOWNLOAD_ATTACHMENTS]:
                    data, fetch_err, provider_failure = await _fetch_gmail_attachment(
                        client,
                        read_credential.access_token,
                        message_id=msg_id,
                        attachment_id=str(row.get("attachment_id") or ""),
                    )
                    if provider_failure and provider_failure.credential_failure:
                        return connected_account_auth_failure(
                            read_credential,
                            _auth_failure_message(provider_failure),
                        )
                    if fetch_err or data is None:
                        error_payload: dict[str, Any] = {
                            "message": fetch_err or "Attachment fetch failed."
                        }
                        if provider_failure:
                            error_payload.update(
                                {
                                    "code": provider_failure.normalized_code,
                                    "provider": provider_failure.client_ret(),
                                }
                            )
                        original_attachment_errors.append(
                            {
                                "attachment_id": row.get("attachment_id"),
                                "filename": row.get("filename"),
                                "error": error_payload,
                            }
                        )
                        continue
                    filename = _safe_filename(str(row.get("filename") or "attachment.bin"))
                    original_attachments.append(
                        {
                            "filename": filename,
                            "mime_type": str(row.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"),
                            "data": data,
                            "source_path": f"gmail:{msg_id}:{row.get('attachment_id')}",
                        }
                    )

            subject = str(headers.get("subject") or "").strip()
            if not subject.lower().startswith(("fwd:", "fw:")):
                subject = f"Fwd: {subject or 'Gmail message'}"
            original_body = str(parsed.get("body_text") or parsed.get("snippet") or "")
            note = str(note_markdown or "").strip()
            body_parts: list[str] = []
            if note:
                body_parts.append(note)
                body_parts.append("")
            body_parts.extend(
                [
                    "---------- Forwarded message ---------",
                    f"From: {headers.get('from') or ''}",
                    f"Date: {headers.get('date') or ''}",
                    f"Subject: {headers.get('subject') or ''}",
                    f"To: {headers.get('to') or ''}",
                    "",
                    original_body,
                ]
            )
            outgoing = build_email_message(
                sender_email=sender_email,
                recipients=recipients,
                cc=split_email_addresses(cc),
                bcc=split_email_addresses(bcc),
                subject=subject,
                body_text="\n".join(body_parts),
                attachments=[*local_attachments, *original_attachments],
            )
            raw = base64.urlsafe_b64encode(outgoing.as_bytes()).decode("ascii").rstrip("=")
            response = await client.post(
                f"{GMAIL_API}/messages/send",
                headers={"Authorization": f"Bearer {token}"},
                json={"raw": raw},
            )
            if response.status_code >= 400:
                failure = _gmail_failure(
                    response,
                    operation="forward_message",
                    fallback="Gmail forward failed.",
                    where="gmail.forward_gmail_message",
                    mutating=True,
                )
                if failure.credential_failure:
                    return connected_account_auth_failure(
                        send_credential,
                        _auth_failure_message(failure),
                    )
                result = failure.error_result(where="gmail.forward_gmail_message")
                result["ret"]["original_attachment_errors"] = original_attachment_errors
                return result
            try:
                data = response.json()
            except Exception:
                data = {}
        return _ok_ret_result(
            {
                "id": str(data.get("id") or ""),
                "thread_id": str(data.get("threadId") or ""),
                "source_message_id": msg_id,
                "sender": sender_email,
                "account_id": send_credential.account_id or read_credential.account_id,
                "attachment_count": len(local_attachments) + len(original_attachments),
                "original_attachment_errors": original_attachment_errors,
            }
        )


kernel = sk.Kernel()
tools = GmailTools()
kernel.add_plugin(tools, "gmail")


__all__ = [
    "GMAIL_PROVIDER_ID",
    "GMAIL_READ_CLAIM",
    "GMAIL_SEND_CLAIM",
    "GMAIL_COMPOSE_CLAIM",
    "GmailTools",
    "kernel",
    "tools",
]
