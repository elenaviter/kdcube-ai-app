# SPDX-License-Identifier: MIT
"""Any IMAP/SMTP provider instance behind the mail verbs: Gmail-shaped envelopes
over the adapter, hub credentials through the store shim, instance hosts from the
catalog settings, drafts as IMAP APPEND."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field

from kdcube_ai_app.apps.chat.sdk.integrations.email import imap_smtp_tools


@dataclass
class _Credential:
    ok: bool = True
    account_id: str = "icloud_1"
    raw_credential: dict = field(default_factory=lambda: {
        "email": "elena.viter@icloud.com",
        "app_password": "abcd-efgh-ijkl-mnop",
    })

    def error_envelope(self, *, where: str):
        return {"ok": False, "error": {"code": "denied", "where": where}, "ret": {}}


ICLOUD_SETTINGS = {
    "imap_host": "imap.mail.me.com", "imap_port": 993,
    "smtp_host": "smtp.mail.me.com", "smtp_port": 587, "smtp_starttls": True,
}


def _tools():
    return imap_smtp_tools.ImapSmtpMailTools(
        provider_id="icloud_mail", connector_app_id="app_password",
        settings=ICLOUD_SETTINGS, label="iCloud Mail",
    )


def _bind_credential(monkeypatch, credential=None):
    async def _resolve(self, *, claim, account_id, tool_name):
        _resolve.calls.append((claim, account_id, tool_name))
        return credential or _Credential()

    _resolve.calls = []
    monkeypatch.setattr(imap_smtp_tools.ImapSmtpMailTools, "_credential", _resolve)
    return _resolve


def test_translate_gmail_relative_dates_into_imap_since_before():
    text, since, before = imap_smtp_tools.translate_gmail_query(
        "subject:(receipt PYGENML) newer_than:60d older_than:1d from:x@y"
    )
    assert text == "subject:(receipt PYGENML) from:x@y"
    assert since and before and since < before
    assert imap_smtp_tools.translate_gmail_query("") == ("", "", "")


def test_search_hands_hub_credential_to_the_adapter_and_shapes_rows(monkeypatch):
    resolve = _bind_credential(monkeypatch)
    seen = {}

    async def fake_fetch(**kwargs):
        store = kwargs["store"]
        seen["tokens"] = asyncio.get_event_loop().run_until_complete if False else None
        seen["kwargs"] = {k: v for k, v in kwargs.items() if k != "store"}
        seen["creds"] = await store.get_tokens_async("icloud_1")
        return {
            "ok": True,
            "messages": [{
                "message_id": "INBOX:42", "subject": "Hello", "from": "a@b",
                "to": "me", "date": "Thu, 03 Sep 2026", "snippet": "hi",
                "label_ids": ["INBOX"], "has_attachments": False,
            }],
        }

    monkeypatch.setattr(imap_smtp_tools, "fetch_icloud_messages", fake_fetch)
    out = asyncio.run(_tools().search(
        query="from:a@b newer_than:7d", max_results=3, account_id="icloud_1",
    ))
    assert resolve.calls == [("email:read", "icloud_1", "icloud_mail.search")]
    assert seen["creds"] == {"username": "elena.viter@icloud.com", "password": "abcd-efgh-ijkl-mnop"}
    assert seen["kwargs"]["unread_only"] is False
    # The instance's catalog hosts ride the account mapping the adapter reads.
    assert seen["kwargs"]["account"]["imap_host"] == "imap.mail.me.com"
    assert seen["kwargs"]["account"]["smtp_port"] == 587
    assert seen["kwargs"]["query"] == "from:a@b" and seen["kwargs"]["since"]
    assert out["ok"] is True
    assert out["ret"]["provider"] == "icloud_mail"
    assert out["ret"]["messages"][0] == {
        "id": "INBOX:42", "thread_id": "", "subject": "Hello", "from": "a@b",
        "to": "me", "date": "Thu, 03 Sep 2026", "snippet": "hi",
        "mailbox": "INBOX", "has_attachments": False,
    }


def test_search_denial_from_the_hub_is_returned_untouched(monkeypatch):
    _bind_credential(monkeypatch, _Credential(ok=False))
    out = asyncio.run(_tools().search(query="x"))
    assert out["ok"] is False and out["error"]["where"] == "icloud_mail.search"


def test_create_draft_appends_into_drafts_with_attachment(monkeypatch):
    resolve = _bind_credential(monkeypatch)
    captured = {}

    def fake_append(creds, raw, mailbox="Drafts"):
        captured["creds"] = dict(creds)
        captured["raw"] = raw
        captured["mailbox"] = mailbox
        return {"uid": "77", "raw": "APPENDUID 1 77"}

    monkeypatch.setattr(imap_smtp_tools, "_append_draft_sync", fake_append)
    attachments = [{
        "filename": "letter.zip",
        "content_base64": base64.b64encode(b"zip bytes").decode("ascii"),
        "mime_type": "application/zip",
    }]
    out = asyncio.run(_tools().create_draft(
        to="consultant@example.com", subject="Juli 2026", body_markdown="Hallo",
        attachments_base64=__import__("json").dumps(attachments), account_id="icloud_1",
    ))
    # Drafts ride the send claim: an APPEND is a write to the mailbox.
    assert resolve.calls == [("email:send", "icloud_1", "icloud_mail.create_draft")]
    assert captured["creds"]["username"] == "elena.viter@icloud.com"
    assert captured["creds"]["smtp_host"] == "smtp.mail.me.com"
    assert captured["mailbox"] == "Drafts"
    assert b"letter.zip" in captured["raw"] and b"Juli 2026" in captured["raw"]
    assert out["ok"] is True
    assert out["ret"]["draft_id"] == "77"
    assert out["ret"]["mailbox"] == "Drafts"
    assert out["ret"]["attachment_count"] == 1
    assert out["ret"]["recipients"] == ["consultant@example.com"]


def test_create_draft_rejects_bad_attachments_before_touching_imap(monkeypatch):
    _bind_credential(monkeypatch)
    called = []
    monkeypatch.setattr(imap_smtp_tools, "_append_draft_sync", lambda *a, **k: called.append(1))
    out = asyncio.run(_tools().create_draft(
        subject="x", attachments_base64='[{"filename": "a.pdf"}]', account_id="icloud_1",
    ))
    assert out["ok"] is False and out["error"]["code"] == "attachment_load_failed"
    assert not called


def test_login_falls_back_to_the_account_email_when_the_credential_holds_only_the_secret(monkeypatch):
    """The hub connect form stores the login as an account attribute and the
    app password as the credential, so a record can hold the secret alone.
    The verdict's account email is the login then (the live iCloud failure
    "missing username or app-specific password" of 2026-09-07)."""
    credential = _Credential(raw_credential={
        "provider_id": "icloud_mail", "connector_app_id": "app_password",
        "claims": ["email:read", "email:send"], "app_password": "abcd-efgh-ijkl-mnop",
    })
    credential.email = "elena.viter@icloud.com"  # type: ignore[attr-defined]
    _bind_credential(monkeypatch, credential)
    seen = {}

    async def fake_fetch(**kwargs):
        seen["creds"] = await kwargs["store"].get_tokens_async("icloud_1")
        seen["account_email"] = kwargs["account"]["email"]
        return {"ok": True, "messages": []}

    monkeypatch.setattr(imap_smtp_tools, "fetch_icloud_messages", fake_fetch)
    out = asyncio.run(_tools().search(query="newer_than:1d", max_results=1, account_id="icloud_1"))
    assert out["ok"] is True
    assert seen["creds"] == {"username": "elena.viter@icloud.com", "password": "abcd-efgh-ijkl-mnop"}
    assert seen["account_email"] == "elena.viter@icloud.com"

    # a credential that carries its own login keeps it
    credential = _Credential(raw_credential={"username": "other.login", "app_password": "x"})
    credential.email = "elena.viter@icloud.com"  # type: ignore[attr-defined]
    _bind_credential(monkeypatch, credential)
    asyncio.run(_tools().search(query="", max_results=1, account_id="icloud_1"))
    assert seen["creds"]["username"] == "other.login"
