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


def test_send_draft_transmits_the_stored_bytes_then_expunges(monkeypatch):
    """A client's 'send draft' is IMAP fetch -> SMTP -> IMAP expunge. The bytes
    the person reviewed are the bytes that go out; the draft is removed only
    after SMTP accepted them; Bcc is stripped from the transmitted copy."""
    from email.message import EmailMessage
    from kdcube_ai_app.apps.chat.sdk.integrations.email import icloud

    draft = EmailMessage()
    draft["From"] = "elena.viter@icloud.com"; draft["To"] = "consultant@example.de"; draft["Bcc"] = "hidden@example.de"
    draft["Subject"] = "Olena Viter: Jul 2026. 1. Kontoauszug und Hauptbrief"; draft["Message-ID"] = "<draft-1@icloud>"
    draft.set_content("Sehr geehrter Herr ...")
    draft.add_attachment(b"PK\x03\x04zip", maintype="application", subtype="zip", filename="konto.zip")
    raw = draft.as_bytes()
    events: list = []

    class FakeIMAP:
        def __init__(self, *a, **k): pass
        def select(self, mailbox, readonly=False): events.append(("select", mailbox, readonly)); return "OK", [b"3"]
        def uid(self, cmd, *args):
            events.append(("uid", cmd) + args)
            if cmd == "FETCH": return "OK", [b"786 (FLAGS (\\Seen \\Draft))", (b"1 (RFC822 {%d}" % len(raw), raw), b")"]
            return "OK", [b""]
        def expunge(self): events.append(("expunge",)); return "OK", [b"1"]
        def append(self, mailbox, flags, date, raw_bytes): events.append(("append", mailbox, flags, len(raw_bytes))); return "OK", [b"[APPENDUID 1 900]"]
        def close(self): pass
        def logout(self): pass

    class FakeSMTP:
        sent: list = []
        def __init__(self, host, port, timeout=0): events.append(("smtp", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self, context=None): events.append(("starttls",))
        def login(self, user, password): events.append(("login", user, bool(password)))
        def sendmail(self, sender, recipients, data): FakeSMTP.sent.append((sender, list(recipients), data)); return {}

    monkeypatch.setattr(icloud, "_connect_imap", lambda creds: FakeIMAP())
    monkeypatch.setattr(icloud.smtplib, "SMTP", FakeSMTP)
    credential = _Credential(raw_credential={"app_password": "abcd-efgh-ijkl-mnop"})
    credential.email = "elena.viter@icloud.com"  # type: ignore[attr-defined]
    _bind_credential(monkeypatch, credential)

    out = asyncio.run(_tools().send_draft(draft_id="17", account_id="icloud_1"))
    assert out["ok"] is True, out
    assert out["ret"]["draft_removed"] is True and out["ret"]["attachment_count"] == 1
    assert out["ret"]["recipients"] == ["consultant@example.de", "hidden@example.de"]
    sender, recipients, data = FakeSMTP.sent[0]
    assert sender == "elena.viter@icloud.com" and recipients == ["consultant@example.de", "hidden@example.de"]
    assert b"Bcc:" not in data and b"konto.zip" in data and b"<draft-1@icloud>" in data
    # order: select Drafts (writable) -> fetch by UID -> SMTP -> flag deleted -> expunge
    kinds = [e[0] if e[0] != "uid" else e[1] for e in events]
    assert kinds.index("FETCH") < kinds.index("smtp") < kinds.index("append") < kinds.index("STORE") < kinds.index("expunge")
    # the transmitted copy lands in Sent (quoted mailbox name, \Seen), and the result says where
    assert ("append", '"Sent Messages"', "\\Seen", len(data)) in events
    assert out["ret"]["sent_copy_mailbox"] == "Sent Messages"
    assert events[0] == ("select", "Drafts", True)  # read-only fetch
    assert ("select", "Drafts", False) in events  # writable only for the removal
    assert ("uid", "STORE", "17", "+FLAGS", r"(\Deleted)") in events  # one backslash on the wire


def test_send_draft_without_recipient_leaves_the_draft_in_place(monkeypatch):
    from email.message import EmailMessage
    from kdcube_ai_app.apps.chat.sdk.integrations.email import icloud

    draft = EmailMessage(); draft["From"] = "elena.viter@icloud.com"; draft["Subject"] = "no recipient"; draft.set_content("x")
    raw = draft.as_bytes(); events: list = []

    class FakeIMAP:
        def __init__(self, *a, **k): pass
        def select(self, mailbox, readonly=False): return "OK", [b"1"]
        def uid(self, cmd, *args):
            events.append(cmd)
            return ("OK", [(b"1 (RFC822)", raw), b")"]) if cmd == "FETCH" else ("OK", [b""])
        def expunge(self): events.append("expunge"); return "OK", [b""]
        def close(self): pass
        def logout(self): pass

    class ExplodingSMTP:
        def __init__(self, *a, **k): raise AssertionError("SMTP must not be contacted")

    monkeypatch.setattr(icloud, "_connect_imap", lambda creds: FakeIMAP())
    monkeypatch.setattr(icloud.smtplib, "SMTP", ExplodingSMTP)
    credential = _Credential(raw_credential={"app_password": "x"}); credential.email = "elena.viter@icloud.com"  # type: ignore[attr-defined]
    _bind_credential(monkeypatch, credential)
    out = asyncio.run(_tools().send_draft(draft_id="18", account_id="icloud_1"))
    assert out["ok"] is False and out["error"]["code"] == "recipient_required"
    assert "STORE" not in events and "expunge" not in events


def test_search_in_drafts_selects_the_drafts_mailbox(monkeypatch):
    _bind_credential(monkeypatch)
    seen = {}

    async def fake_fetch(**kwargs):
        seen["mailbox"] = kwargs.get("mailbox"); seen["query"] = kwargs.get("query")
        return {"ok": True, "messages": []}

    monkeypatch.setattr(imap_smtp_tools, "fetch_icloud_messages", fake_fetch)
    tools = _tools()
    asyncio.run(tools.search(query="in:drafts subject:Kontoauszug", max_results=5, account_id="icloud_1"))
    assert seen["mailbox"] == "Drafts" and "in:" not in seen["query"] and "Kontoauszug" in seen["query"]
    asyncio.run(tools.search(query="subject:x", max_results=5, account_id="icloud_1"))
    assert seen["mailbox"] == ""  # INBOX by default
    asyncio.run(tools.search(query='in:"Sent Messages" x', max_results=5, account_id="icloud_1"))
    assert seen["mailbox"] == "Sent Messages" and seen["query"] == "x"


def test_imap_mailbox_names_are_quoted_when_needed():
    from kdcube_ai_app.apps.chat.sdk.integrations.email.icloud import imap_mailbox_arg
    assert imap_mailbox_arg("INBOX") == "INBOX"
    assert imap_mailbox_arg("Drafts") == "Drafts"
    assert imap_mailbox_arg("Sent Messages") == '"Sent Messages"'
    assert imap_mailbox_arg('"Sent Messages"') == '"Sent Messages"'
    assert imap_mailbox_arg("[Gmail]/Sent Mail") == '"[Gmail]/Sent Mail"'
    assert imap_mailbox_arg("") == "INBOX"


def test_file_as_sent_appends_to_the_sent_mailbox_as_seen(monkeypatch):
    """Backfilling the Sent record of a message that already went out: same
    builder as a draft, filed in the Sent mailbox with \\Seen, no draft flag."""
    _bind_credential(monkeypatch)
    seen = {}

    def fake_append(creds, raw, mailbox, flags="\\Draft"):
        seen["mailbox"] = mailbox; seen["flags"] = flags; seen["raw"] = raw
        return {"uid": "901", "raw": "[APPENDUID 5 901]"}

    monkeypatch.setattr(imap_smtp_tools, "_append_draft_sync", fake_append)
    out = asyncio.run(_tools().create_draft(
        to="consultant@example.de", subject="Olena Viter: Jul 2026. 1. Kontoauszug und Hauptbrief",
        body_markdown="Sehr geehrter Herr", account_id="icloud_1", file_as_sent=True,
    ))
    assert out["ok"] is True and out["ret"]["mailbox"] == "Sent Messages" and out["ret"]["draft_id"] == "901"
    assert seen["mailbox"] == "Sent Messages" and seen["flags"] == "\\Seen"
    assert b"To: consultant@example.de" in seen["raw"]
    # default stays Drafts with the draft flag
    asyncio.run(_tools().create_draft(to="a@b", subject="s", body_markdown="x", account_id="icloud_1"))
    assert seen["mailbox"] == "Drafts" and seen["flags"] == "\\Draft"
