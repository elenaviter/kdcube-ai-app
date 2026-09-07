# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Any IMAP/SMTP mailbox behind the mail verbs, with the envelopes Gmail answers.

One transport for every provider instance on Connection Hub's
``email.imap_smtp_app_password`` adapter (iCloud Mail, Yahoo, a company
server): the instance's hosts come from its catalog settings, the account's
username and app-specific password come from the hub credential resolved per
claim. Nothing here names a provider. The older adapter module
(``email.icloud``) holds the IMAP/SMTP mechanics and expects a token
``store``; a tiny shim hands it the hub credential, and the hosts ride the
account mapping it already reads overrides from.

Results are shaped like ``google.gmail_tools`` results (``{ok, error, ret}``
with the same field names) so callers never branch on provider after routing.

Drafts: IMAP has no drafts API; a draft is an ``APPEND`` into the Drafts
mailbox, which is why it rides the ``email:send`` claim (a write to the
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
    fetch_icloud_attachment,
    fetch_icloud_message,
    fetch_icloud_messages,
    send_icloud_draft,
    send_icloud_message,
)

IMAP_SMTP_ADAPTER = "email.imap_smtp_app_password"
EMAIL_READ_CLAIM = "email:read"
EMAIL_SEND_CLAIM = "email:send"
DRAFTS_MAILBOX = "Drafts"
# Host settings a provider instance may carry in its catalog adapter_config;
# absent keys fall back to the adapter module's defaults.
HOST_SETTING_KEYS = ("imap_host", "imap_port", "smtp_host", "smtp_port", "smtp_starttls", "drafts_mailbox")
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
    """The adapter's ``store`` contract over one hub credential.

    ``login`` is the connected account's email from the hub verdict: the
    connect form stores the login as an account attribute, so a credential
    record may hold the app password alone. Without this fallback every
    IMAP/SMTP call on such an account failed with "missing username"."""

    def __init__(self, raw_credential: Mapping[str, Any], *, login: str = ""):
        self._raw = dict(raw_credential or {})
        self._login = _clean(login)

    async def get_tokens_async(self, account_id: str) -> dict[str, Any]:
        return {
            "username": _clean(self._raw.get("username") or self._raw.get("email")) or self._login,
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


class ImapSmtpMailTools:
    """Provider transport for one IMAP/SMTP provider INSTANCE, called by the
    mail realm. ``provider_id``/``connector_app_id`` are the instance's hub
    identity; ``settings`` are its catalog hosts."""

    def __init__(
        self,
        *,
        provider_id: str,
        connector_app_id: str = "app_password",
        settings: Mapping[str, Any] | None = None,
        label: str = "",
    ) -> None:
        self.provider_id = _clean(provider_id)
        self.connector_app_id = _clean(connector_app_id) or "app_password"
        self.settings = {
            key: value for key, value in dict(settings or {}).items() if key in HOST_SETTING_KEYS
        }
        self.label = _clean(label) or self.provider_id
        self.drafts_mailbox = _clean(self.settings.get("drafts_mailbox")) or DRAFTS_MAILBOX

    async def _credential(self, *, claim: str, account_id: str, tool_name: str):
        return await resolve_connected_account_claim(
            globals(),
            provider_id=self.provider_id,
            connector_app_id=self.connector_app_id,
            claim=claim,
            account_id=account_id,
            tool_name=tool_name,
        )

    def _adapter_inputs(self, credential: Any) -> tuple[_HubTokenStore, dict[str, Any]]:
        raw = dict(getattr(credential, "raw_credential", None) or {})
        # The adapter reads host overrides from the account mapping; the
        # instance's catalog settings ride there so no host is hard-coded.
        login = _clean(getattr(credential, "email", ""))
        account = {
            "account_id": _clean(getattr(credential, "account_id", "")),
            "email": _clean(raw.get("email") or raw.get("username")) or login,
            **{key: value for key, value in self.settings.items() if key != "drafts_mailbox"},
        }
        return _HubTokenStore(raw, login=login), account

    def _split_mailbox(self, query: str) -> tuple[str, str]:
        """Gmail's ``in:drafts`` / ``in:sent`` / ``in:<mailbox>`` operator, which
        the agents already speak, selects the IMAP mailbox to search. Drafts
        maps to this instance's configured Drafts mailbox; other names pass
        through as the server's mailbox name; no operator means INBOX."""
        mailbox = ""

        def take(match: "re.Match[str]") -> str:
            nonlocal mailbox
            name = (match.group(1) or match.group(2) or "").strip()
            lowered = name.lower()
            mailbox = self.drafts_mailbox if lowered == "drafts" else ("INBOX" if lowered == "inbox" else name)
            return " "

        rest = re.sub(r'(?i)\bin:(?:"([^"]+)"|(\S+))', take, _clean(query))
        return mailbox, " ".join(rest.split())

    async def search(
        self, *, query: str = "", max_results: int = 5, account_id: str = ""
    ) -> dict[str, Any]:
        where = f"{self.provider_id}.search"
        credential = await self._credential(
            claim=EMAIL_READ_CLAIM, account_id=account_id, tool_name=where
        )
        if not credential.ok:
            return credential.error_envelope(where=where)
        store, account = self._adapter_inputs(credential)
        mailbox, query = self._split_mailbox(query)
        imap_query, since, before = translate_gmail_query(query)
        result = await fetch_icloud_messages(
            store=store,
            account=account,
            unread_only=False,
            limit=max(1, min(int(max_results or 5), 50)),
            query=imap_query,
            since=since,
            before=before,
            mailbox=mailbox,
        )
        if not result.get("ok"):
            error = dict(result.get("error") or {})
            return _error(
                code=_clean(error.get("code")) or "imap_search_failed",
                message=_clean(error.get("message")) or f"{self.label} search failed.",
                where=where,
                ret={"provider": self.provider_id, **{k: v for k, v in error.items() if k not in ("code", "message")}},
            )
        rows = [_row_from_summary(row) for row in (result.get("messages") or [])]
        return _ok(
            {
                "messages": rows,
                "count": len(rows),
                "next_cursor": "",
                "account_id": credential.account_id,
                "provider": self.provider_id,
            }
        )

    async def read_message(
        self, *, message_id: str, include_html: bool = False, account_id: str = ""
    ) -> dict[str, Any]:
        where = f"{self.provider_id}.read_message"
        credential = await self._credential(
            claim=EMAIL_READ_CLAIM, account_id=account_id, tool_name=where
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
                code=_clean(error.get("code")) or "imap_read_failed",
                message=_clean(error.get("message")) or f"{self.label} message read failed.",
                where=where,
                ret={"provider": self.provider_id},
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
                "provider": self.provider_id,
            }
        )

    async def read_attachment(
        self, *, message_id: str, attachment_id: str = "", part_id: str = "",
        max_bytes: int = 10 * 1024 * 1024, account_id: str = "",
    ) -> dict[str, Any]:
        """One attachment's bytes, base64 in the envelope (same shape as the
        Gmail transport's inline read). IMAP part ids are stable per message,
        so ``part_id`` and ``attachment_id`` (``part:<n>``) name the same part."""
        where = f"{self.provider_id}.read_attachment"
        selector = _clean(part_id) or _clean(attachment_id)
        if not selector:
            return _error(code="attachment_selector_required", message="part_id or attachment_id is required.", where=where)
        credential = await self._credential(claim=EMAIL_READ_CLAIM, account_id=account_id, tool_name=where)
        if not credential.ok:
            return credential.error_envelope(where=where)
        store, account = self._adapter_inputs(credential)
        result = await fetch_icloud_attachment(
            store=store, account=account, message_id=message_id,
            attachment_id=selector, max_bytes=max_bytes,
        )
        if not result.get("ok"):
            error = dict(result.get("error") or {})
            return _error(
                code=_clean(error.get("code")) or "imap_attachment_read_failed",
                message=_clean(error.get("message")) or f"{self.label} attachment read failed.",
                where=where,
                ret={"provider": self.provider_id, **{k: error[k] for k in ("size_bytes", "max_bytes") if k in error}},
            )
        meta = dict(result.get("attachment") or {})
        return _ok(
            {
                "message_id": _clean(result.get("message_id")) or message_id,
                "part_id": _clean(meta.get("part_id")),
                "filename": _clean(result.get("filename")),
                "mime_type": _clean(result.get("mime_type")) or "application/octet-stream",
                "size_bytes": int(result.get("size_bytes") or 0),
                "content_base64": str(result.get("base64") or ""),
                "account_id": credential.account_id,
                "provider": self.provider_id,
            }
        )

    async def send_draft(self, *, draft_id: str, account_id: str = "") -> dict[str, Any]:
        """Send a draft that sits in the Drafts mailbox, exactly as stored, and
        remove it. ``draft_id`` is the UID ``create_draft`` returned (or an
        ``imap:<mailbox>:<uid>`` message id). Gated on the send claim."""
        where = f"{self.provider_id}.send_draft"
        if not _clean(draft_id):
            return _error(code="draft_id_required", message="draft_id is required.", where=where)
        credential = await self._credential(claim=EMAIL_SEND_CLAIM, account_id=account_id, tool_name=where)
        if not credential.ok:
            return credential.error_envelope(where=where)
        store, account = self._adapter_inputs(credential)
        result = await send_icloud_draft(store=store, account=account, draft_id=draft_id, mailbox=self.drafts_mailbox)
        if not result.get("ok"):
            error = dict(result.get("error") or {})
            return _error(
                code=_clean(error.get("code")) or "smtp_send_draft_failed",
                message=_clean(error.get("message")) or f"{self.label} draft send failed.",
                where=where,
                ret={"provider": self.provider_id},
            )
        return _ok(
            {
                "id": _clean(result.get("provider_message_id")),
                "draft_id": _clean(draft_id),
                "subject": _clean(result.get("subject")),
                "recipients": list(result.get("recipients") or []),
                "attachment_count": int(result.get("attachment_count") or 0),
                "draft_removed": bool(result.get("draft_removed")),
                "account_id": credential.account_id,
                "provider": self.provider_id,
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
        where = f"{self.provider_id}.create_draft"
        credential = await self._credential(
            claim=EMAIL_SEND_CLAIM, account_id=account_id, tool_name=where
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
                code=_clean(error.get("code")) or "mail_account_not_connected",
                message=_clean(error.get("message")) or f"{self.label} credentials are incomplete.",
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
            appended = await asyncio.to_thread(
                _append_draft_sync, creds, message.as_bytes(), self.drafts_mailbox
            )
        except Exception as exc:  # noqa: BLE001 - provider failure surfaces as an envelope
            LOGGER.warning("[%s.create_draft] append failed: %s", self.provider_id, exc)
            return _error(
                code="imap_draft_failed",
                message=f"{self.label} refused the draft: {exc}",
                where=where,
                ret={"provider": self.provider_id},
            )
        return _ok(
            {
                "draft_id": appended.get("uid", ""),
                "message_id": _clean(message.get("Message-ID")),
                "mailbox": self.drafts_mailbox,
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
                "provider": self.provider_id,
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
        where = f"{self.provider_id}.send"
        recipients = split_email_addresses(to)
        if not recipients:
            return _error(
                code="recipient_required",
                message="At least one recipient email address is required.",
                where=where,
            )
        credential = await self._credential(
            claim=EMAIL_SEND_CLAIM, account_id=account_id, tool_name=where
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
                code=_clean(error.get("code")) or "smtp_send_failed",
                message=_clean(error.get("message")) or f"{self.label} send failed.",
                where=where,
                ret={"provider": self.provider_id},
            )
        return _ok(
            {
                "id": _clean(result.get("provider_message_id")),
                "sender": sender,
                "account_id": credential.account_id,
                "attachment_count": len(attachments),
                "provider": self.provider_id,
            }
        )


def _append_draft_sync(creds: Mapping[str, Any], raw: bytes, mailbox: str = DRAFTS_MAILBOX) -> dict[str, Any]:
    """IMAP APPEND into the Drafts mailbox with the \\Draft flag; returns the
    new UID when the server reports APPENDUID."""
    conn = _connect_imap(creds)
    try:
        typ, data = conn.append(
            mailbox,
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
    "DRAFTS_MAILBOX",
    "EMAIL_READ_CLAIM",
    "EMAIL_SEND_CLAIM",
    "HOST_SETTING_KEYS",
    "IMAP_SMTP_ADAPTER",
    "ImapSmtpMailTools",
    "bind_integrations",
    "bind_service",
    "translate_gmail_query",
]
