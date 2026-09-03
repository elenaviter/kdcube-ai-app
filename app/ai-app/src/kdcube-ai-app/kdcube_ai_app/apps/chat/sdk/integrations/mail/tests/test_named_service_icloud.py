# SPDX-License-Identifier: MIT
"""The mail namespace spans providers: iCloud accounts are listed, searched,
read, and drafted beside Gmail, and a cross-provider ambiguity asks."""

from __future__ import annotations

from typing import Any

import pytest

from connection_hub.delegated_to_kdcube.models import ConnectedAccount
from kdcube_ai_app.apps.chat.sdk.integrations.mail.named_service import (
    ACTION_DRAFT,
    ACTION_SEND,
    MAIL_NAMESPACE,
    MailNamedServiceProvider,
)
from kdcube_ai_app.apps.chat.sdk.integrations.mail.realm import MailProviderSpec

GMAIL_SPEC = MailProviderSpec(
    key="gmail", label="Gmail", provider_id="google", connector_app_id="gmail", transport="gmail",
    read_claim="gmail:read", send_claim="gmail:send", draft_claim="gmail:compose",
)
ICLOUD_SPEC = MailProviderSpec(
    key="icloud_mail", label="iCloud Mail", provider_id="icloud_mail", connector_app_id="app_password",
    transport="imap_smtp", read_claim="email:read", send_claim="email:send", draft_claim="email:send",
    settings={"imap_host": "imap.mail.me.com"},
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceRequest,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    NamedServiceContext,
    OBJECT_ACTION,
    OBJECT_GET,
    OBJECT_LIST,
    OBJECT_SEARCH,
)


def _ctx() -> NamedServiceContext:
    return NamedServiceContext(tenant="demo", project="project", user_id="user-1")


def _gmail_account(account_id: str = "acc-g", *claims: str) -> ConnectedAccount:
    return ConnectedAccount(
        account_id=account_id, provider_id="google", connector_app_id="gmail",
        email="lena@nestlogic.com", display_name="Lena", claims=claims or ("gmail:read", "gmail:send"),
        credential_id="cred-g",
    )


def _icloud_account(account_id: str = "acc-i", *claims: str) -> ConnectedAccount:
    return ConnectedAccount(
        account_id=account_id, provider_id="icloud_mail", connector_app_id="app_password",
        email="elena.viter@icloud.com", display_name="", claims=claims or ("email:read", "email:send"),
        credential_id="cred-i",
    )


class _FakeGmail:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search_gmail(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search_gmail", kwargs))
        return {"ok": True, "ret": {"account_id": kwargs["account_id"], "messages": [
            {"id": "g-1", "subject": "From Gmail", "from": "a@b", "date": "d", "snippet": "s"},
        ], "next_cursor": ""}}

    async def create_gmail_draft(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_gmail_draft", kwargs))
        return {"ok": True, "ret": {"draft_id": "gd-1", "account_id": kwargs["account_id"]}}


class _FakeIcloud:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search", kwargs))
        return {"ok": True, "ret": {"account_id": kwargs["account_id"], "provider": "icloud_mail", "messages": [
            {"id": "INBOX:7", "subject": "From iCloud", "from": "c@d", "date": "d", "snippet": "s", "mailbox": "INBOX"},
        ], "next_cursor": ""}}

    async def read_message(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("read_message", kwargs))
        return {"ok": True, "ret": {"account_id": kwargs["account_id"], "message": {
            "id": kwargs["message_id"], "headers": {"subject": "Hi", "from": "c@d", "date": "d"},
            "body_text": "hello body", "attachments": [], "attachment_count": 0,
        }}}

    async def create_draft(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_draft", kwargs))
        return {"ok": True, "ret": {"draft_id": "77", "mailbox": "Drafts", "account_id": kwargs["account_id"]}}

    async def send(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("send", kwargs))
        return {"ok": True, "ret": {"id": "<m@id>", "account_id": kwargs["account_id"]}}


class _Provider(MailNamedServiceProvider):
    def __init__(self, gmail: list[ConnectedAccount], icloud: list[ConnectedAccount]) -> None:
        super().__init__(entrypoint=None, bundle_id="kdcube-services@1-0")
        self.gmail_rows, self.icloud_rows = gmail, icloud
        self._gmail = _FakeGmail()
        self.fake_icloud = _FakeIcloud()
        self._imap_transports["icloud_mail"] = self.fake_icloud  # type: ignore[assignment]

    @property
    def _icloud(self):
        return self.fake_icloud

    async def _realm_specs(self, ctx):
        return [GMAIL_SPEC, ICLOUD_SPEC]

    async def _gmail_accounts(self, ctx, *, claim: str = ""):
        return [a for a in self.gmail_rows if not claim or a.allows(claim)]

    async def _imap_accounts(self, ctx, spec, *, claim: str = ""):
        assert spec.provider_id == "icloud_mail"
        return [a for a in self.icloud_rows if not claim or a.allows(claim)]


@pytest.mark.asyncio
async def test_list_shows_both_providers():
    provider = _Provider([_gmail_account()], [_icloud_account()])
    response = await provider.object_list(_ctx(), NamedServiceRequest(operation=OBJECT_LIST, namespace=MAIL_NAMESPACE))
    assert response.ok
    rows = response.ret["items"]
    assert [(row["provider"], row["email"]) for row in rows] == [
        ("gmail", "lena@nestlogic.com"), ("icloud_mail", "elena.viter@icloud.com"),
    ]
    assert rows[1]["ref"] == "mail:icloud_mail:acc-i"
    assert response.ret["extra"]["providers"] == ["gmail", "icloud_mail"]


@pytest.mark.asyncio
async def test_search_spans_both_providers_and_routes_each_to_its_transport():
    provider = _Provider([_gmail_account()], [_icloud_account()])
    response = await provider.object_search(
        _ctx(), NamedServiceRequest(operation=OBJECT_SEARCH, namespace=MAIL_NAMESPACE, query="hello", limit=10),
    )
    assert response.ok
    refs = [row["ref"] for row in response.ret["items"]]
    assert refs == ["mail:gmail:acc-g:message:g-1", "mail:icloud_mail:acc-i:message:INBOX:7"]
    assert provider._gmail.calls[0][0] == "search_gmail"
    assert provider._icloud.calls[0] == ("search", {"query": "hello", "max_results": 10, "account_id": "acc-i"})
    assert response.ret["extra"]["providers"] == ["gmail", "icloud_mail"]


@pytest.mark.asyncio
async def test_search_with_icloud_account_id_pins_that_provider_only():
    provider = _Provider([_gmail_account()], [_icloud_account()])
    response = await provider.object_search(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_SEARCH, namespace=MAIL_NAMESPACE, query="x",
                            filters={"account_id": "acc-i"}),
    )
    assert response.ok
    assert provider._gmail.calls == []
    assert [row["provider"] for row in response.ret["items"]] == ["icloud_mail"]


@pytest.mark.asyncio
async def test_get_reads_an_icloud_message_by_its_ref():
    provider = _Provider([_gmail_account()], [_icloud_account()])
    response = await provider.object_get(
        _ctx(), NamedServiceRequest(operation=OBJECT_GET, namespace=MAIL_NAMESPACE, object_ref="mail:icloud_mail:acc-i:message:INBOX:7"),
    )
    assert response.ok
    obj = response.ret["object"]
    assert obj["provider"] == "icloud_mail" and obj["subject"] == "Hi" and obj["body_text"] == "hello body"
    assert provider._icloud.calls[0][1]["message_id"] == "INBOX:7"


@pytest.mark.asyncio
async def test_draft_on_icloud_ref_appends_via_icloud_and_never_sends():
    provider = _Provider([_gmail_account()], [_icloud_account()])
    response = await provider.object_action(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_ACTION, namespace=MAIL_NAMESPACE, action=ACTION_DRAFT,
                            object_ref="mail:icloud_mail:acc-i",
                            payload={"to": "x@y", "subject": "Juli", "body_markdown": "Hallo"}),
    )
    assert response.ok
    assert [name for name, _ in provider._icloud.calls] == ["create_draft"]
    assert provider._gmail.calls == []
    assert response.ret["extra"]["provider"] == "icloud_mail"
    assert response.ret["extra"]["result"]["mailbox"] == "Drafts"


@pytest.mark.asyncio
async def test_send_without_account_across_two_providers_asks_with_labeled_candidates():
    provider = _Provider([_gmail_account()], [_icloud_account()])
    response = await provider.object_action(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_ACTION, namespace=MAIL_NAMESPACE, action=ACTION_SEND,
                            payload={"to": "x@y", "subject": "s", "body_markdown": "b"}),
    )
    assert not response.ok
    assert response.error.code == "account_required"
    labels = [row["label"] for row in response.error.details["candidates"]]
    assert labels == ["Lena (Gmail)", "elena.viter@icloud.com (iCloud Mail)"]
    assert provider._gmail.calls == [] and provider._icloud.calls == []


@pytest.mark.asyncio
async def test_only_icloud_can_draft_so_no_question_is_asked():
    # Gmail row lacks gmail:compose; only iCloud may draft (email:send).
    provider = _Provider([_gmail_account("acc-g", "gmail:read", "gmail:send")], [_icloud_account()])
    response = await provider.object_action(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_ACTION, namespace=MAIL_NAMESPACE, action=ACTION_DRAFT,
                            payload={"subject": "s", "body_markdown": "b"}),
    )
    assert response.ok
    assert [name for name, _ in provider._icloud.calls] == ["create_draft"]


@pytest.mark.asyncio
async def test_forward_on_icloud_is_refused_honestly():
    provider = _Provider([], [_icloud_account()])
    response = await provider.object_action(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_ACTION, namespace=MAIL_NAMESPACE, action="forward",
                            object_ref="mail:icloud_mail:acc-i:message:INBOX:7", payload={"to": "x@y"}),
    )
    assert not response.ok
    assert response.error.code == "mail_provider_action_not_implemented"
