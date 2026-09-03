# SPDX-License-Identifier: MIT
"""Mail as a realm: members discovered from the hub catalog by adapter family,
accounts listed across them inside the agent binding, and the connect-first
consent naming every member.

The surfaced case: a user with Gmail and iCloud both connected and consented
asked for "the last email from my iCloud"; the tool declared Google only, saw
one Google account, and silently answered from Gmail."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kdcube_ai_app.apps.chat.sdk.integrations.mail import realm

CATALOG = {
    "providers": {
        "google": {
            "adapter": "google.oauth",
            "label": "Google",
            "claims": {"gmail:read": {}, "gmail:send": {}, "gmail:compose": {}, "docs:read": {}},
            "connector_apps": {"gmail": {"enabled": True, "allowed_claims": ["gmail:read", "gmail:send", "gmail:compose", "docs:read"]}},
        },
        "icloud_mail": {
            "adapter": "email.imap_smtp_app_password",
            "label": "iCloud Mail",
            "claims": {"email:read": {}, "email:send": {}},
            "connector_apps": {"app_password": {"enabled": True, "allowed_claims": ["email:read", "email:send"]}},
            "adapter_config": {"imap_host": "imap.mail.me.com", "smtp_host": "smtp.mail.me.com"},
        },
        "nestlogic_mail": {
            "adapter": "email.imap_smtp_app_password",
            "label": "NestLogic Mail",
            "claims": {"email:read": {}, "email:send": {}},
            "connector_apps": {"app_password": {"enabled": True}},
            "adapter_config": {"imap_host": "mail.nestlogic.com"},
        },
        "slack": {"adapter": "slack.oauth_user_token", "claims": {"slack:search": {}}, "connector_apps": {}},
        "disabled_mail": {"adapter": "email.imap_smtp_app_password", "enabled": False, "claims": {"email:read": {}}},
    }
}


@dataclass
class _Account:
    account_id: str
    provider_id: str
    email: str = ""
    display_name: str = ""
    status: str = "connected"
    claims: tuple[str, ...] = ()
    credential_id: str = "cred"

    @property
    def connected(self) -> bool:
        return self.status == "connected" and bool(self.credential_id)


class _Client:
    def __init__(self, rows, catalog=CATALOG):
        self.rows = rows
        self._catalog = catalog
        self.calls: list[str] = []

    async def catalog(self):
        return self._catalog

    async def list_accounts(self, provider_id: str = ""):
        self.calls.append(provider_id)
        return [row for row in self.rows if row.provider_id == provider_id]


def _gmail(**kw) -> _Account:
    base = dict(account_id="google_1", provider_id="google", email="lena@nestlogic.com",
                claims=("gmail:read", "gmail:send", "docs:read"))
    return _Account(**{**base, **kw})


def _icloud(**kw) -> _Account:
    base = dict(account_id="icloud_1", provider_id="icloud_mail", email="elena.viter@icloud.com",
                claims=("email:read", "email:send"))
    return _Account(**{**base, **kw})


def _bind(monkeypatch, rows, *, scope=None):
    client = _Client(rows)

    async def _hub_client():
        return client

    monkeypatch.setattr(realm, "_hub_client", _hub_client)
    monkeypatch.setattr(realm, "_binding_for", lambda provider_id, account_id: scope)
    monkeypatch.setattr(realm, "resolve_connector_app_id", lambda provider_id: "")
    return client


def test_members_are_discovered_by_adapter_family_never_by_name(monkeypatch):
    _bind(monkeypatch, [])
    specs = asyncio.run(realm.discover_mail_providers())
    assert [(s.key, s.transport, s.provider_id) for s in specs] == [
        ("gmail", "gmail", "google"),
        ("icloud_mail", "imap_smtp", "icloud_mail"),
        ("nestlogic_mail", "imap_smtp", "nestlogic_mail"),
    ]
    icloud = specs[1]
    assert icloud.label == "iCloud Mail"
    assert icloud.connector_app_id == "app_password"
    assert icloud.settings["imap_host"] == "imap.mail.me.com"
    assert icloud.requirement("draft") == {
        "provider_id": "icloud_mail", "connector_app_id": "app_password", "claims": ["email:send"],
    }
    assert specs[0].requirement("draft")["claims"] == ["gmail:compose"]


def test_any_of_gate_spans_every_member(monkeypatch):
    _bind(monkeypatch, [])
    specs = asyncio.run(realm.discover_mail_providers())
    gate = realm.mail_requirement(specs, "read")
    assert [alt["provider_id"] for alt in gate["any_of"]] == ["google", "icloud_mail", "nestlogic_mail"]
    assert realm.mail_requirement(specs[:1], "read") == specs[0].requirement("read")


def test_lists_accounts_across_members_and_marks_what_each_may_do(monkeypatch):
    client = _bind(monkeypatch, [_gmail(), _icloud()])
    accounts = asyncio.run(realm.list_mail_accounts())
    assert client.calls == ["google", "icloud_mail", "nestlogic_mail"]
    by_key = {item.provider.key: item for item in accounts}
    assert set(by_key) == {"gmail", "icloud_mail"}
    gmail, icloud = by_key["gmail"], by_key["icloud_mail"]
    assert gmail.allows("read") and gmail.allows("send") and not gmail.allows("draft")
    assert icloud.allows("read") and icloud.allows("send") and icloud.allows("draft")
    assert icloud.public_dict()["provider_label"] == "iCloud Mail"


def test_agent_binding_hides_unbound_accounts_and_narrows_claims(monkeypatch):
    def binding(provider_id, account_id):
        return ("email:read",) if provider_id == "icloud_mail" else ()

    client = _Client([_gmail(), _icloud()])

    async def _hub_client():
        return client

    monkeypatch.setattr(realm, "_hub_client", _hub_client)
    monkeypatch.setattr(realm, "_binding_for", binding)
    monkeypatch.setattr(realm, "resolve_connector_app_id", lambda provider_id: "")
    accounts = asyncio.run(realm.list_mail_accounts())
    assert [item.provider.key for item in accounts] == ["icloud_mail"]
    assert accounts[0].allows("read") and not accounts[0].allows("send")


def test_disconnected_accounts_are_not_offered(monkeypatch):
    _bind(monkeypatch, [_gmail(), _icloud(credential_id="")])
    assert [item.provider.key for item in asyncio.run(realm.list_mail_accounts())] == ["gmail"]


def test_no_hub_scope_means_no_members_and_no_accounts(monkeypatch):
    async def _none():
        return None

    monkeypatch.setattr(realm, "_hub_client", _none)
    assert asyncio.run(realm.discover_mail_providers()) == []
    assert asyncio.run(realm.list_mail_accounts()) == []


def test_nothing_connected_offers_every_member_not_gmail(monkeypatch):
    _bind(monkeypatch, [])
    specs = asyncio.run(realm.discover_mail_providers())
    envelope = realm.connect_required_envelope(
        where="productivity_mail_search", need="read", specs=specs, tenant="demo", project="project",
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "needs_connected_account_consent"
    assert "Gmail or iCloud Mail or NestLogic Mail" in envelope["error"]["message"]
    offered = {(row["provider"], row["claim"]) for row in envelope["ret"]["providers"]}
    assert offered == {("gmail", "gmail:read"), ("icloud_mail", "email:read"), ("nestlogic_mail", "email:read")}
    assert envelope["consent_required"] is True
