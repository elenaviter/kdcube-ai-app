# SPDX-License-Identifier: MIT
"""Mail as a realm: account discovery across providers and the selection rule.

The surfaced case: a user with Gmail and iCloud both connected and consented
asked for "the last email from my iCloud"; the tool declared Google only, saw
one Google account, and silently answered from Gmail. The realm must list
both, ask when both are eligible, and honor an explicit account_id."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from kdcube_ai_app.apps.chat.sdk.integrations.mail import realm


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
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[str] = []

    async def list_accounts(self, provider_id: str = ""):
        self.calls.append(provider_id)
        return [row for row in self.rows if row.provider_id == provider_id]


def _gmail(**kw) -> _Account:
    base = dict(
        account_id="google_1", provider_id="google", email="lena@nestlogic.com",
        claims=("gmail:read", "gmail:send", "docs:read"),
    )
    return _Account(**{**base, **kw})


def _icloud(**kw) -> _Account:
    base = dict(
        account_id="icloud_1", provider_id="icloud_mail", email="elena.viter@icloud.com",
        claims=("email:read", "email:send"),
    )
    return _Account(**{**base, **kw})


def _accounts(monkeypatch, rows, *, scope=None):
    client = _Client(rows)

    async def _hub_client():
        return client

    monkeypatch.setattr(realm, "_hub_client", _hub_client)
    monkeypatch.setattr(realm, "_binding_for", lambda provider_id, account_id: scope)
    return asyncio.run(realm.list_mail_accounts()), client


def test_lists_both_providers_and_marks_what_each_may_do(monkeypatch):
    accounts, client = _accounts(monkeypatch, [_gmail(), _icloud()])
    assert client.calls == ["google", "icloud_mail"]
    by_key = {item.provider.key: item for item in accounts}
    assert set(by_key) == {"gmail", "icloud"}
    gmail, icloud = by_key["gmail"], by_key["icloud"]
    assert gmail.allows("read") and gmail.allows("send")
    assert not gmail.allows("draft")  # gmail:compose not granted on this account
    assert icloud.allows("read") and icloud.allows("send") and icloud.allows("draft")
    assert icloud.public_dict()["provider_label"] == "iCloud Mail"


def test_two_eligible_accounts_ask_instead_of_defaulting_to_gmail(monkeypatch):
    accounts, _ = _accounts(monkeypatch, [_gmail(), _icloud()])
    choice = realm.choose_mail_account(accounts, need="read", where="productivity_mail_search")
    assert choice.account is None
    assert choice.denial is not None
    assert choice.denial["error"]["code"] == "account_required"
    labels = [row["label"] for row in choice.denial["ret"]["candidates"]]
    assert labels == ["lena@nestlogic.com (Gmail)", "elena.viter@icloud.com (iCloud Mail)"]


def test_explicit_account_id_routes_to_that_provider(monkeypatch):
    accounts, _ = _accounts(monkeypatch, [_gmail(), _icloud()])
    choice = realm.choose_mail_account(
        accounts, account_id="icloud_1", need="read", where="productivity_mail_get"
    )
    assert choice.account is not None and choice.account.provider.key == "icloud"
    assert choice.account.provider.requirement("read") == {
        "provider_id": "icloud_mail",
        "connector_app_id": "app_password",
        "claims": ["email:read"],
    }


def test_unknown_account_id_is_refused_with_candidates_never_silently_replaced(monkeypatch):
    accounts, _ = _accounts(monkeypatch, [_gmail(), _icloud()])
    choice = realm.choose_mail_account(
        accounts, account_id="nope", need="read", where="productivity_mail_get"
    )
    assert choice.account is None
    assert choice.denial["error"]["code"] == "account_not_found"
    assert {row["account_id"] for row in choice.denial["ret"]["candidates"]} == {"google_1", "icloud_1"}


def test_one_eligible_account_is_used_without_asking(monkeypatch):
    # Drafting: only iCloud may (gmail:compose was not granted on the Gmail row).
    accounts, _ = _accounts(monkeypatch, [_gmail(), _icloud()])
    choice = realm.choose_mail_account(accounts, need="draft", where="productivity_mail_draft")
    assert choice.denial is None
    assert choice.account is not None and choice.account.provider.key == "icloud"


def test_no_eligible_account_yields_neither_choice_nor_denial(monkeypatch):
    accounts, _ = _accounts(monkeypatch, [_gmail(claims=("docs:read",))])
    choice = realm.choose_mail_account(accounts, need="read", where="productivity_mail_search")
    assert choice.account is None and choice.denial is None


def test_agent_binding_hides_unbound_accounts_and_narrows_claims(monkeypatch):
    # Delegated card binds only the iCloud account, read-only.
    def binding(provider_id, account_id):
        if provider_id == "icloud_mail":
            return ("email:read",)
        return ()

    client = _Client([_gmail(), _icloud()])

    async def _hub_client():
        return client

    monkeypatch.setattr(realm, "_hub_client", _hub_client)
    monkeypatch.setattr(realm, "_binding_for", binding)
    accounts = asyncio.run(realm.list_mail_accounts())
    assert [item.provider.key for item in accounts] == ["icloud"]
    assert accounts[0].allows("read") and not accounts[0].allows("send")


def test_disconnected_accounts_are_not_offered(monkeypatch):
    accounts, _ = _accounts(monkeypatch, [_gmail(), _icloud(credential_id="")])
    assert [item.provider.key for item in accounts] == ["gmail"]


def test_no_hub_scope_means_no_accounts(monkeypatch):
    async def _none():
        return None

    monkeypatch.setattr(realm, "_hub_client", _none)
    assert asyncio.run(realm.list_mail_accounts()) == []


def test_nothing_connected_offers_every_mail_provider_not_gmail():
    envelope = realm.connect_required_envelope(
        where="productivity_mail_search", need="read", tenant="demo", project="project",
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "needs_connected_account_consent"
    assert "Gmail or iCloud Mail" in envelope["error"]["message"]
    offered = {(row["provider"], row["claim"]) for row in envelope["ret"]["providers"]}
    assert offered == {("gmail", "gmail:read"), ("icloud", "email:read")}
    assert envelope["ret"]["reason"] == "connect_required"
    assert envelope["consent_required"] is True
