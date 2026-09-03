# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""iCloud Mail behind the mail verbs, with the same envelopes Gmail answers.

The IMAP/SMTP adapter (``email.icloud``) predates the Connection Hub account
model and expects a token ``store``; here the hub's connected-account
credential (username + app-specific password, resolved per claim) is handed
to it through a tiny shim, so the mail realm can route a call to an iCloud
account exactly as it routes one to Gmail. Results are shaped like
``google.gmail_tools`` results (``{ok, error, ret}`` with the same field
names) so callers never branch on provider after routing.

Drafts: iCloud has no drafts API; a draft is an IMAP ``APPEND`` into the
Drafts mailbox, which is why it rides the ``email:send`` claim (a write to the
mailbox) rather than a compose claim of its own.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import imaplib
import logging
import re
import time
from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    resolve_connected_account_claim,
)
from kdcube_ai_app.apps.chat.sdk.integrations.email.delivery import build_email_message, split_email_addresses
from kdcube_ai_app.apps.chat.sdk.integrations.email.icloud import (
    _connect_imap,
    _imap_credentials,
    fetch_icloud_message,
    fetch_icloud_messages,
    send_icloud_message,
)

ICLOUD_PROVIDER_ID = "icloud_mail"
ICLOUD_CONNECTOR_APP_ID = "app_password"
ICLOUD_READ_CLAIM = "email:read"
ICLOUD_SEND_CLAIM = "email:send"
ICLOUD_DRAFTS_MAILBOX = "Drafts"
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

_SERVICE: Any = None
_INTEGRATIONS: dict[str, Any] = {}
LOGGER = logging.getLogger(__name__)

_NEWER_THAN_RE = re.compile(r"\bnewer_than:(\d+)([dwmy])\b", re.IGNORECASE)
_OLDER_THAN_RE = re.compile(r"\bolder_than:(\d+)([dwmy])\b", re.IGNORECASE)
_UNIT_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def bind_service(svc: Any) -> None:
    global _SERVICE
    _SERVICE = svc


def bind_integrations(integrations: Mapping[str, Any] | None) -> None:
    global _INTEGRATIONS
    _INTEGRATIONS = dict(integrations or {})


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ok(ret: Any) -> dict[str, Any]:
    return {"ok": True, "error": None, "ret": ret}


def _error(*, code: str, message: str, where: str, ret: Any = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {"code": code, "message": message, "where": where},
        "ret": ret if ret is not None else {},
    }


class _HubTokenStore:
    """The adapter's ``store`` contract over one hub credential."""

    def __init__(self, raw_credential: Mapping[str, Any]):
        self._raw = dict(raw_credential or {})

    async def get_tokens_async(self, account_id: str) -> dict[str, Any]:
        return {
            "username": _clean(self._raw.get("username") or self._raw.get("email")),
            "password": _clean(
                self._raw.get("app_password")
                or self._raw.get("password")
                or self._raw.get("access_token")
            ),
        }


def translate_gmail_query(query: str) -> tuple[str, str, str]:
    """(imap_query, since, before): the Gmail relative-date operators the
    agents already speak, rewritten into the absolute dates IMAP understands.
    from:/to:/subject:/before:/after: pass through, the adapter parses them."""
    text = _clean(query)
    since = before = ""
    today = dt.date.today()

    def _days(match: re.Match[str]) -> int:
        return int(match.group(1)) * _UNIT_DAYS[match.group(2).lower()]

    newer = _NEWER_THAN_RE.search(text)
    if newer:
        since = (today - dt.timedelta(days=_days(newer))).isoformat()
        text = _NEWER_THAN_RE.sub("", text)
    older = _OLDER_THAN_RE.search(text)
    if older:
        before = (today - dt.timedelta(days=_days(older))).isoformat()
        text = _OLDER_THAN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(), since, before


def _row_from_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": _clean(row.get("message_id")),
        "thread_id": _clean(row.get("thread_id")),
        "subject": _clean(row.get("subject")),
        "from": _clean(row.get("from")),
        "to": _clean(row.get("to")),
        "date": _clean(row.get("date")),
        "snippet": _clean(row.get("snippet")),
        "mailbox": (row.get("label_ids") or ["INBOX"])[0],
        "has_attachments": bool(row.get("has_attachments")),
    }


class IcloudMailTools:
    """Provider transport for iCloud accounts, called by the mail realm."""

    async def _credential(self, *, claim: str, account_id: str, tool_name: str):
        return await resolve_connected_account_claim(
            globals(),
            provider_id=ICLOUD_PROVIDER_ID,
            connector_app_id=ICLOUD_CONNECTOR_APP_ID,
            claim=claim,
            account_id=account_id,
            tool_name=tool_name,
        )

    @staticmethod
    def _adapter_inputs(credential: Any) -> tuple[_HubTokenStore, dict[str, Any]]:
        raw = dict(getattr(credential, "raw_credential", None) or {})
        account = {
            "account_id": _clean(getattr(credential, "account_id", "")),
            "email": _clean(raw.get("email") or raw.get("username")),
        }
        return _HubTokenStore(raw), account

    async def search(
        self, *, query: str = "", max_results: int = 5, account_id: str = ""
    ) -> dict[str, Any]:
        where = "icloud.search"
        credential = await self._credential(
            claim=ICLOUD_READ_CLAIM, account_id=account_id, tool_name=where
        )
        if not credential.ok:
            return credential.error_envelope(where=where)
        store, account = self._adapter_inputs(credential)
        imap_query, since, before = translate_gmail_query(query)
        result = await fetch_icloud_messages(
            store=store,
            account=account,
            unread_only=False,
            limit=max(1, min(int(max_results or 5), 50)),
            query=imap_query,
            since=since,
            before=before,
        )
        if not result.get("ok"):
            error = dict(result.get("error") or {})
            return _error(
                code=_clean(error.get("code")) or "icloud_search_failed",
                message=_clean(error.get("message")) or "iCloud search failed.",
                where=where,
                ret={"provider": "icloud", **{k: v for k, v in error.items() if k not in ("code", "message")}},
            )
        rows = [_row_from_summary(row) for row in (result.get("messages") or [])]
        return _ok(
            {
                "messages": rows,
                "count": len(rows),
                "next_cursor": "",
                "account_id": credential.account_id,
                "provider": "icloud",
            }
        )

    async def read_message(
        self, *, message_id: str, include_html: bool = False, account_id: str = ""
    ) -> dict[str, Any]:
        where = "icloud.read_message"
        credential = await self._credential(
            claim=ICLOUD_READ_CLAIM, account_id=account_id, tool_name=where
        )
        if not credential.ok:
            return credential.error_envelope(where=where)
        store, account = self._adapter_inputs(credential)
        result = await fetch_icloud_message(
            store=store, account=account, message_id=message_id, body_limit=24000
        )
        if not result.get("ok"):
            error = dict(result.get("error") or {})
            return _error(
                code=_clean(error.get("code")) or "icloud_read_failed",
                message=_clean(error.get("message")) or "iCloud message read failed.",
                where=where,
                ret={"provider": "icloud"},
            )
        message = dict(result.get("message") or {})
        return _ok(
            {
                "message": {
                    "id": _clean(message.get("message_id")),
                    "thread_id": "",
                    "label_ids": list(message.get("label_ids") or []),
                    "snippet": _clean(message.get("snippet")),
                    "headers": {
                        "subject": _clean(message.get("subject")),
                        "from": _clean(message.get("from")),
                        "to": _clean(message.get("to")),
                        "cc": _clean(message.get("cc")),
                        "date": _clean(message.get("date")),
                        "message_id": _clean(message.get("provider_message_id")),
                    },
                    "body_text": _clean(message.get("body")),
                    # IMAP body extraction is plain-text; iCloud HTML is not
                    # surfaced here, and saying so beats an empty field that
                    # looks like a missing part.
                    "body_html": "",
                    "body_html_available": False if include_html else None,
                    "attachments": list(message.get("attachments") or []),
                    "attachment_count": len(message.get("attachments") or []),
                },
                "account_id": credential.account_id,
                "provider": "icloud",
            }
        )

    def _load_inline_attachments(self, value: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from kdcube_ai_app.apps.chat.sdk.integrations.google.gmail_tools import (
            _load_inline_attachments,
        )

        return _load_inline_attachments(value)

    async def create_draft(
        self,
        *,
        to: str = "",
        subject: str = "",
        body_markdown: str = "",
        cc: str = "",
        bcc: str = "",
        body_html: str = "",
        attachments_base64: Any = "",
        account_id: str = "",
    ) -> dict[str, Any]:
        where = "icloud.create_draft"
        credential = await self._credential(
            claim=ICLOUD_SEND_CLAIM, account_id=account_id, tool_name=where
        )
        if not credential.ok:
            return credential.error_envelope(where=where)
        attachments, errors = self._load_inline_attachments(attachments_base64)
        if errors:
            return _error(
                code="attachment_load_failed",
                message="One or more requested attachments could not be loaded.",
                where=where,
                ret={"attachment_errors": errors},
            )
        store, account = self._adapter_inputs(credential)
        tokens = await store.get_tokens_async(account["account_id"])
        creds = _imap_credentials(store, account, tokens)
        if not creds.get("ok"):
            error = dict(creds.get("error") or {})
            return _error(
                code=_clean(error.get("code")) or "icloud_account_not_connected",
                message=_clean(error.get("message")) or "iCloud credentials are incomplete.",
                where=where,
            )
        sender = _clean(creds.get("username")) or account["email"]
        message = build_email_message(
            sender_email=sender,
            recipients=split_email_addresses(to),
            cc=split_email_addresses(cc),
            bcc=split_email_addresses(bcc),
            subject=subject,
            body_text=body_markdown,
            body_html=body_html,
            attachments=attachments,
        )
        try:
            appended = await asyncio.to_thread(_append_draft_sync, creds, message.as_bytes())
        except Exception as exc:  # noqa: BLE001 - provider failure surfaces as an envelope
            LOGGER.warning("[icloud.create_draft] append failed: %s", exc)
            return _error(
                code="icloud_draft_failed",
                message=f"iCloud refused the draft: {exc}",
                where=where,
                ret={"provider": "icloud"},
            )
        return _ok(
            {
                "draft_id": appended.get("uid", ""),
                "message_id": _clean(message.get("Message-ID")),
                "mailbox": ICLOUD_DRAFTS_MAILBOX,
                "sender": sender,
                "account_id": credential.account_id,
                "recipients": split_email_addresses(to),
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
                "provider": "icloud",
            }
        )

    async def send(
        self,
        *,
        to: str,
        subject: str = "",
        body_markdown: str = "",
        cc: str = "",
        bcc: str = "",
        body_html: str = "",
        attachments_base64: Any = "",
        account_id: str = "",
    ) -> dict[str, Any]:
        where = "icloud.send"
        recipients = split_email_addresses(to)
        if not recipients:
            return _error(
                code="recipient_required",
                message="At least one recipient email address is required.",
                where=where,
            )
        credential = await self._credential(
            claim=ICLOUD_SEND_CLAIM, account_id=account_id, tool_name=where
        )
        if not credential.ok:
            return credential.error_envelope(where=where)
        attachments, errors = self._load_inline_attachments(attachments_base64)
        if errors:
            return _error(
                code="attachment_load_failed",
                message="One or more requested attachments could not be loaded.",
                where=where,
                ret={"attachment_errors": errors},
            )
        store, account = self._adapter_inputs(credential)
        tokens = await store.get_tokens_async(account["account_id"])
        sender = _clean(tokens.get("username")) or account["email"]
        message = build_email_message(
            sender_email=sender,
            recipients=recipients,
            cc=split_email_addresses(cc),
            bcc=split_email_addresses(bcc),
            subject=subject,
            body_text=body_markdown,
            body_html=body_html,
            attachments=attachments,
        )
        result = await send_icloud_message(store=store, account=account, msg=message)
        if not result.get("ok"):
            error = dict(result.get("error") or {})
            return _error(
                code=_clean(error.get("code")) or "icloud_send_failed",
                message=_clean(error.get("message")) or "iCloud send failed.",
                where=where,
                ret={"provider": "icloud"},
            )
        return _ok(
            {
                "id": _clean(result.get("provider_message_id")),
                "sender": sender,
                "account_id": credential.account_id,
                "attachment_count": len(attachments),
                "provider": "icloud",
            }
        )


def _append_draft_sync(creds: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    """IMAP APPEND into Drafts with the \\Draft flag; returns the new UID when
    the server reports APPENDUID."""
    conn = _connect_imap(creds)
    try:
        typ, data = conn.append(
            ICLOUD_DRAFTS_MAILBOX,
            "\\Draft",
            imaplib.Time2Internaldate(time.time()),
            raw,
        )
        if typ != "OK":
            raise RuntimeError(f"iCloud IMAP APPEND answered {typ}: {data!r}")
        text = b" ".join(part for part in (data or []) if isinstance(part, bytes)).decode("utf-8", "replace")
        match = re.search(r"APPENDUID\s+\d+\s+(\d+)", text)
        return {"uid": match.group(1) if match else "", "raw": text}
    finally:
        try:
            conn.logout()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "ICLOUD_CONNECTOR_APP_ID",
    "ICLOUD_PROVIDER_ID",
    "ICLOUD_READ_CLAIM",
    "ICLOUD_SEND_CLAIM",
    "IcloudMailTools",
    "bind_integrations",
    "bind_service",
    "translate_gmail_query",
]
