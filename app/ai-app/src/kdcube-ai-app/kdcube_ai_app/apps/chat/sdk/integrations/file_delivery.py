# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Turn-free byte fetch and credential resolution for integration delivery.

Chat tools materialize provider binaries into the ReAct turn's artifact
workspace. Transports without a turn — the named-services MCP surface and the
public signed-download routes — need the bytes directly. This module fetches
them through the same Connection Hub facade the tools use
(``DelegatedToKdcubeClient.ensure_claim``): no workspace, no credential
ownership, no broker re-implementation.

Provider transport helpers are reused from the integration tool modules
(``gmail_tools``/Slack Web API shapes) so there is exactly one implementation
of each provider call path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from typing import Any, AsyncIterator

import httpx

from kdcube_ai_app.apps.chat.sdk.integrations.google.gmail_tools import (
    GMAIL_PROVIDER_ID,
    GMAIL_READ_CLAIM,
    _extract_message_content,
    _fetch_gmail_attachment,
    _get_gmail_message,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube import (
    DelegatedToKdcubeClient,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.models import (
    ClaimResolution,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.connector_app_resolution import resolve_connector_app_id

LOGGER = logging.getLogger("kdcube.sdk.integrations.file_delivery")

MAX_DELIVERY_BYTES = 25 * 1024 * 1024
MAIL_MESSAGE_SNAPSHOT_SCHEMA = "kdcube.mail.message.snapshot.v1"
MAIL_MESSAGE_SNAPSHOT_MEDIA_TYPE = (
    "application/vnd.kdcube.mail.message.snapshot+json;version=1"
)


async def _json_chunks(
    value: Any,
    *,
    chunk_bytes: int = 64 * 1024,
) -> AsyncIterator[bytes]:
    """Encode JSON incrementally and yield to the proc event loop."""

    encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
    pending = bytearray()
    for piece in encoder.iterencode(value):
        encoded = piece.encode("utf-8")
        offset = 0
        if pending:
            take = min(chunk_bytes - len(pending), len(encoded))
            pending.extend(encoded[:take])
            offset = take
            if len(pending) == chunk_bytes:
                yield bytes(pending)
                pending.clear()
                await asyncio.sleep(0)
        while len(encoded) - offset >= chunk_bytes:
            yield encoded[offset : offset + chunk_bytes]
            offset += chunk_bytes
            await asyncio.sleep(0)
        pending.extend(encoded[offset:])
    if pending:
        yield bytes(pending)


def _failure(*, code: str, message: str, status: int = 400, resolution: ClaimResolution | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "error": {"code": code, "message": message},
        "status": status,
    }
    if resolution is not None:
        out["resolution"] = resolution.to_dict(include_credential=False)
    return out


async def resolve_connected_account_access_token(
    entrypoint: Any,
    *,
    user_id: str,
    tenant: str,
    project: str,
    provider_id: str,
    connector_app_id: str,
    claim: str,
    account_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve one provider access token for the download identity.

    Returns ``(token, None)`` or ``("", failure_dict)``. Consent failures keep
    the broker's resolution fields so callers can answer with the contract."""
    client = await DelegatedToKdcubeClient.from_connection_hub(
        entrypoint,
        user_id=user_id,
        tenant=tenant,
        project=project,
    )
    # The calling agent's per-provider account binding (if any) restricts which
    # connected account may satisfy this claim. Unset / non-agent (e.g. a
    # user-initiated download) → None → no restriction (unchanged).
    from kdcube_ai_app.apps.chat.sdk.solutions.connections.agent_account_scope import (
        account_claim_scope_for,
    )
    resolution = await client.ensure_claim(
        provider_id=provider_id,
        connector_app_id=connector_app_id,
        claim=claim,
        account_id=account_id or None,
        account_claim_scope=account_claim_scope_for(provider_id),
    )
    if not resolution.ok or resolution.credential is None:
        return "", _failure(
            code="needs_connected_account_consent",
            message=resolution.message or "The connected account cannot authorize this download.",
            status=403,
            resolution=resolution,
        )
    raw = dict(resolution.credential.credential or {})
    token = str(raw.get("access_token") or raw.get("token") or "").strip()
    if not token:
        return "", _failure(
            code="credential_unusable",
            message="The connected account credential does not carry a usable access token.",
            status=403,
            resolution=resolution,
        )
    return token, None


async def fetch_mail_attachment(
    entrypoint: Any,
    *,
    user_id: str,
    tenant: str,
    project: str,
    account_id: str,
    message_id: str,
    attachment_id: str,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Fetch one Gmail attachment for complete out-of-band delivery.

    Gmail returns attachment data as base64 JSON, so this adapter must decode
    one provider response in memory. It deliberately applies no KDCube model-
    context cap; the ordinary in-turn download tool keeps its bounded limit.
    """
    token, failure = await resolve_connected_account_access_token(
        entrypoint,
        user_id=user_id,
        tenant=tenant,
        project=project,
        provider_id=GMAIL_PROVIDER_ID,
        connector_app_id=resolve_connector_app_id(GMAIL_PROVIDER_ID),
        claim=GMAIL_READ_CLAIM,
        account_id=account_id,
    )
    if failure is not None:
        return failure
    async with httpx.AsyncClient(timeout=60.0) as client:
        message, error, auth_failed = await _get_gmail_message(client, token, message_id)
        if message is None:
            return _failure(
                code="gmail_message_unavailable",
                message=error or "Failed to fetch the Gmail message.",
                status=403 if auth_failed else 404,
            )
        parsed = _extract_message_content(message)
        rows = [*(parsed.get("attachments") or []), *(parsed.get("inline_attachments") or [])]
        # Gmail attachment ids rotate on every messages.get, so refs carry the
        # stable part id; match that first, then a same-fetch attachment id.
        row = next(
            (item for item in rows if str(item.get("part_id") or "") == attachment_id),
            None,
        ) or next(
            (item for item in rows if str(item.get("attachment_id") or "") == attachment_id),
            None,
        )
        if row is None:
            return _failure(
                code="gmail_attachment_not_found",
                message="The message does not carry the requested attachment.",
                status=404,
            )
        data, error, auth_failed = await _fetch_gmail_attachment(
            client,
            token,
            message_id=message_id,
            attachment_id=str(row.get("attachment_id") or ""),
            max_bytes=max_bytes,
        )
    if data is None:
        return _failure(
            code="gmail_attachment_fetch_failed",
            message=error or "Failed to fetch the Gmail attachment.",
            status=403 if auth_failed else 502,
        )
    filename = str(row.get("filename") or "attachment.bin")
    mime = str(row.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream")
    return {
        "ok": True,
        "data": data,
        "filename": filename,
        "mime_type": mime,
        "size_bytes": len(data),
        "account_id": account_id,
        "message_id": message_id,
        "attachment_id": attachment_id,
    }


async def fetch_mail_message_snapshot(
    entrypoint: Any,
    *,
    user_id: str,
    tenant: str,
    project: str,
    account_id: str,
    message_id: str,
    object_ref: str,
) -> dict[str, Any]:
    """Stream one complete normalized Gmail message as a JSON snapshot.

    The ordinary named-service ``object.get`` response stays bounded for
    prompt safety. This path is the complete-data escape hatch used by signed
    external downloads and ``react.pull`` materialization.
    """

    token, failure = await resolve_connected_account_access_token(
        entrypoint,
        user_id=user_id,
        tenant=tenant,
        project=project,
        provider_id=GMAIL_PROVIDER_ID,
        connector_app_id=resolve_connector_app_id(GMAIL_PROVIDER_ID),
        claim=GMAIL_READ_CLAIM,
        account_id=account_id,
    )
    if failure is not None:
        return failure
    async with httpx.AsyncClient(timeout=60.0) as client:
        message, error, auth_failed = await _get_gmail_message(
            client, token, message_id
        )
    if message is None:
        return _failure(
            code="gmail_message_unavailable",
            message=error or "Failed to fetch the Gmail message.",
            status=403 if auth_failed else 404,
        )
    parsed = _extract_message_content(message)
    normalized = {
        "id": str(parsed.get("id") or message_id),
        "thread_id": str(parsed.get("thread_id") or ""),
        "headers": dict(parsed.get("headers") or {}),
        "snippet": str(parsed.get("snippet") or ""),
        "body_text": str(parsed.get("body_text") or ""),
        "body_html": str(parsed.get("body_html") or ""),
        "attachments": list(parsed.get("attachments") or []),
        "inline_attachments": list(parsed.get("inline_attachments") or []),
        "attachment_count": int(parsed.get("attachment_count") or 0),
        "inline_attachment_count": int(
            parsed.get("inline_attachment_count") or 0
        ),
        "account_id": account_id,
    }
    snapshot = {
        "schema": MAIL_MESSAGE_SNAPSHOT_SCHEMA,
        "object_ref": object_ref,
        "object_kind": "mail.message",
        "message": normalized,
        "materialization": {
            "complete_body": True,
            "body_text_chars": len(normalized["body_text"]),
            "body_html_chars": len(normalized["body_html"]),
            "attachment_bytes_included": False,
            "attachment_delivery": (
                "Attachment metadata is included. Fetch each mail attachment "
                "ref separately so its bytes retain independent authorization "
                "and delivery semantics."
            ),
        },
    }
    return {
        "ok": True,
        "chunks": _json_chunks(snapshot),
        "filename": f"gmail-{message_id}.message.json",
        "mime_type": MAIL_MESSAGE_SNAPSHOT_MEDIA_TYPE,
        "status": 200,
    }


async def fetch_slack_file(
    entrypoint: Any,
    *,
    user_id: str,
    tenant: str,
    project: str,
    account_id: str,
    file_id: str,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Stream one Slack file without imposing a KDCube context-size cap.

    ``max_bytes`` remains accepted for source compatibility but is not used by
    this out-of-band delivery path. Provider/deployment transport limits still
    apply; KDCube does not turn a large but downloadable Slack file into a
    terminal named-service result.
    """
    del max_bytes
    # Keep this import at the call boundary. The Slack package exports its
    # named-service provider, which itself imports this delivery module.
    from kdcube_ai_app.apps.chat.sdk.integrations.slack.tools import (
        SLACK_API,
        SLACK_FILES_READ_CLAIM,
        SLACK_PROVIDER_ID,
    )

    token, failure = await resolve_connected_account_access_token(
        entrypoint,
        user_id=user_id,
        tenant=tenant,
        project=project,
        provider_id=SLACK_PROVIDER_ID,
        connector_app_id=resolve_connector_app_id(SLACK_PROVIDER_ID),
        claim=SLACK_FILES_READ_CLAIM,
        account_id=account_id,
    )
    if failure is not None:
        return failure
    async with httpx.AsyncClient(timeout=60.0) as client:
        info = await client.get(
            f"{SLACK_API}/files.info",
            headers={"Authorization": f"Bearer {token}"},
            params={"file": file_id},
        )
        try:
            payload = info.json()
        except Exception:
            payload = {}
        if info.status_code >= 400 or not (isinstance(payload, dict) and payload.get("ok")):
            detail = str((payload or {}).get("error") or f"HTTP {info.status_code}")
            auth_failed = detail in {"invalid_auth", "not_authed", "token_revoked", "account_inactive"}
            return _failure(
                code="slack_file_info_failed",
                message=f"Slack files.info failed: {detail}.",
                status=403 if auth_failed else 502,
            )
        file_obj = payload.get("file") if isinstance(payload.get("file"), dict) else {}
        download_url = str(file_obj.get("url_private_download") or file_obj.get("url_private") or "").strip()
        if not download_url:
            return _failure(
                code="slack_file_not_downloadable",
                message="Slack file does not expose a private download URL for this token.",
                status=404,
            )
    download_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    )
    stream_context = download_client.stream(
        "GET",
        download_url,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        response = await stream_context.__aenter__()
    except Exception as exc:
        await download_client.aclose()
        LOGGER.warning(
            "Slack file stream could not be opened: account_id=%s file_id=%s error=%s",
            account_id,
            file_id,
            exc,
        )
        return _failure(
            code="slack_file_download_failed",
            message="Slack file download could not be started.",
            status=502,
        )
    if response.status_code >= 400:
        status_code = response.status_code
        await stream_context.__aexit__(None, None, None)
        await download_client.aclose()
        return _failure(
            code="slack_file_download_failed",
            message=f"Slack file download failed with HTTP {status_code}.",
            status=403 if status_code in {401, 403} else 502,
        )

    async def _chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            await stream_context.__aexit__(None, None, None)
            await download_client.aclose()

    filename = str(file_obj.get("name") or file_obj.get("title") or f"{file_id}.bin")
    mime = str(file_obj.get("mimetype") or mimetypes.guess_type(filename)[0] or "application/octet-stream")
    return {
        "ok": True,
        "chunks": _chunks(),
        "filename": filename,
        "mime_type": mime,
        "size_bytes": int(file_obj.get("size") or 0),
        "account_id": account_id,
        "file_id": file_id,
    }


__all__ = [
    "MAX_DELIVERY_BYTES",
    "MAIL_MESSAGE_SNAPSHOT_MEDIA_TYPE",
    "MAIL_MESSAGE_SNAPSHOT_SCHEMA",
    "fetch_mail_attachment",
    "fetch_mail_message_snapshot",
    "fetch_slack_file",
    "resolve_connected_account_access_token",
]
