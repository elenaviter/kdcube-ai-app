# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Demand ordering on the door's gate-1 check (operator ruling 2026-07-25):
an account-backed operation with ZERO connected accounts on the backing
provider leads with the CONNECT demand (guided plan + agent-grant hand-off),
never with the agent grant; with an account present the gate-1 denial stands
(the Slack-story shape)."""

from types import SimpleNamespace

import pytest

import kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_credentials.consent_denial as consent_denial_mod
import kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_to_kdcube.store as store_mod
import kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.discovery as discovery_mod
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_credentials.consent_denial import (
    connect_first_denial,
    connect_first_denial_for_identity,
)

_MAIL_REQUIREMENT = {
    "provider_id": "google-mail",
    "connector_app_id": "google",
    "claims": ["gmail:read", "gmail:send"],
    "claims_by_operation": {
        "object.search": ["gmail:read"],
        "object.action.send": ["gmail:send"],
    },
}


def _wire(monkeypatch, *, requirements, accounts):
    class _FakeDiscovery:
        def __init__(self, *args, **kwargs):
            pass

        async def entries_for_namespace(self, namespace):
            return [SimpleNamespace(spec=SimpleNamespace(metadata={"connected_accounts": requirements}))]

    class _FakeStore:
        def __init__(self, *, user_id, **kwargs):
            assert user_id == "user-1"

        async def list_accounts(self, *, provider_id=""):
            return list(accounts)

    monkeypatch.setattr(discovery_mod, "RedisNamedServiceDiscovery", _FakeDiscovery)
    monkeypatch.setattr(discovery_mod, "_redis_client_from_settings", lambda: None)
    monkeypatch.setattr(store_mod, "DelegatedToKdcubeStore", _FakeStore)
    monkeypatch.setattr(
        consent_denial_mod,
        "delegated_credential_view",
        lambda request: SimpleNamespace(
            grantor_user_id="user-1",
            agent_client_id="kdcube-agent:workspace@2026-03-31-13-36:main",
            client_id="kdcube-agent:workspace@2026-03-31-13-36:main",
            resource="*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/named_services*",
        ),
    )


@pytest.mark.asyncio
async def test_zero_accounts_leads_with_connect(monkeypatch):
    _wire(monkeypatch, requirements=[_MAIL_REQUIREMENT], accounts=[])

    denial = await connect_first_denial(
        object(),
        namespace="mail",
        tool="search",
        operation="object.search",
        required=["mail:read", "named_services:use"],
        missing=["mail:read", "named_services:use"],
        tenant="t",
        project="p",
    )

    assert denial is not None
    assert denial["error"]["code"] == "needs_connected_account_consent"
    assert denial["reason"] == "connect_required"
    assert denial["retry_hint"] is True
    assert denial["provider_id"] == "google-mail"
    assert denial["namespace"] == "mail"
    assert denial["missing_grants"] == ["mail:read", "named_services:use"]
    consent = denial["consent"]
    assert consent["namespace"] == "mail"
    # The guided plan's hand-off needs the agent identity riding along, and
    # the DOOR claims the hand-off grant must cover (a fresh flow has no
    # recorded pending demand to fall back on).
    url = denial.get("connection_hub_url") or consent.get("url") or ""
    assert "agent_client_id" in url
    assert "agent_claims=" in url
    assert consent["agent_claims"] == ["mail:read", "named_services:use"]


@pytest.mark.asyncio
async def test_existing_account_keeps_gate1_order(monkeypatch):
    _wire(
        monkeypatch,
        requirements=[_MAIL_REQUIREMENT],
        accounts=[SimpleNamespace(account_id="acc-1", provider_id="google-mail", connected=True)],
    )

    denial = await connect_first_denial(
        object(),
        namespace="mail",
        tool="search",
        operation="object.search",
        required=["mail:read"],
        missing=["mail:read"],
        tenant="t",
        project="p",
    )

    assert denial is None


@pytest.mark.asyncio
async def test_disconnected_account_still_leads_with_connect(monkeypatch):
    """A stale/disconnected account record (a prior connect the user removed)
    cannot back the claim - connect must still lead, not the agent grant. Only
    a CONNECTED account keeps gate-1 order."""
    _wire(
        monkeypatch,
        requirements=[_MAIL_REQUIREMENT],
        accounts=[SimpleNamespace(account_id="acc-1", provider_id="google-mail", connected=False)],
    )

    denial = await connect_first_denial(
        object(),
        namespace="mail",
        tool="search",
        operation="object.search",
        required=["mail:read"],
        missing=["mail:read"],
        tenant="t",
        project="p",
    )

    assert denial is not None
    assert denial["reason"] == "connect_required"


@pytest.mark.asyncio
async def test_metadata_operation_does_not_lead_with_connect_mail(monkeypatch):
    """A metadata operation (provider.about, object.schema, capabilities) needs
    no account claim - it must NOT ask the user to connect an account. The
    differentiated mail realm has no mapping for it, so connect-first skips it
    (the missing door claim falls to a plain agent-grant)."""
    _wire(monkeypatch, requirements=[_MAIL_REQUIREMENT], accounts=[])

    denial = await connect_first_denial(
        object(),
        namespace="mail",
        tool="about",
        operation="provider.about",
        required=["named_services:use"],
        missing=["named_services:use"],
        tenant="t",
        project="p",
    )

    assert denial is None


@pytest.mark.asyncio
async def test_schema_operation_does_not_lead_with_connect_slack(monkeypatch):
    """object.schema on the flat slack realm needs only door admission; none of
    the missing claims are slack claims, so connect-first must NOT fire (never
    ask to connect the account, let alone for the whole vocabulary)."""
    _wire(monkeypatch, requirements=[_SLACK_REQUIREMENT], accounts=[])

    denial = await connect_first_denial(
        object(),
        namespace="slack",
        tool="schema",
        operation="object.schema",
        required=["named_services:use"],
        missing=["named_services:use"],
        tenant="t",
        project="p",
    )

    assert denial is None


@pytest.mark.asyncio
async def test_namespace_without_account_realm_is_untouched(monkeypatch):
    _wire(monkeypatch, requirements=[], accounts=[])

    denial = await connect_first_denial(
        object(),
        namespace="mem",
        tool="search",
        operation="object.search",
        required=["memories:read"],
        missing=["memories:read"],
        tenant="t",
        project="p",
    )

    assert denial is None


_SLACK_REQUIREMENT = {
    "provider_id": "slack",
    "connector_app_id": "slack",
    "claims": [
        "slack:channels", "slack:search", "slack:history",
        "slack:post", "slack:files:read", "slack:files:write",
    ],
}


@pytest.mark.asyncio
async def test_flat_claims_scope_to_the_missing_ask(monkeypatch):
    """A flat provider claim list is the whole vocabulary; the connect demand
    asks only for what THIS attempt is missing (operator regression ruling:
    never present the user with every claim in the world)."""
    _wire(monkeypatch, requirements=[_SLACK_REQUIREMENT], accounts=[])

    denial = await connect_first_denial(
        object(),
        namespace="slack",
        tool="search",
        operation="object.search",
        required=["named_services:use", "slack:search"],
        missing=["named_services:use", "slack:search"],
        tenant="t",
        project="p",
    )

    assert denial is not None
    consent_claims = denial["consent"].get("claims") or []
    assert consent_claims == ["slack:search"]


def _wire_no_discovery(monkeypatch, *, accounts):
    """Explicit-requirements callers never touch discovery: wire a discovery
    that fails loudly if constructed, plus the account store."""

    class _ExplodingDiscovery:
        def __init__(self, *args, **kwargs):
            raise AssertionError("discovery must not be used when requirements are explicit")

    class _FakeStore:
        def __init__(self, *, user_id, **kwargs):
            assert user_id == "user-1"

        async def list_accounts(self, *, provider_id=""):
            return list(accounts)

    monkeypatch.setattr(discovery_mod, "RedisNamedServiceDiscovery", _ExplodingDiscovery)
    monkeypatch.setattr(discovery_mod, "_redis_client_from_settings", lambda: None)
    monkeypatch.setattr(store_mod, "DelegatedToKdcubeStore", _FakeStore)


@pytest.mark.asyncio
async def test_explicit_requirements_skip_discovery_zero_accounts(monkeypatch):
    """A plain MCP tool passes its own declared requirement: discovery is
    skipped entirely and the connect-first denial still shapes the same."""
    _wire_no_discovery(monkeypatch, accounts=[])

    denial = await connect_first_denial_for_identity(
        grantor_user_id="user-1",
        agent_client_id="kdcube-agent:app:main",
        agent_resource="*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/productivity*",
        namespace="productivity_slack_search",
        tool="productivity_slack_search",
        operation="search",
        required=["slack:search"],
        missing=["slack:search"],
        tenant="t",
        project="p",
        requirements=[{"provider_id": "slack", "claims": ["slack:search"]}],
    )

    assert denial is not None
    assert denial["error"]["code"] == "needs_connected_account_consent"
    assert denial["reason"] == "connect_required"
    assert denial["retry_hint"] is True
    assert denial["provider_id"] == "slack"
    assert denial["namespace"] == "productivity_slack_search"
    assert denial["consent"].get("claims") == ["slack:search"]


@pytest.mark.asyncio
async def test_explicit_requirements_with_connected_account_return_none(monkeypatch):
    """With a usable account present the ordering does not apply - the caller
    falls back to the resolver's own account-level consent."""
    _wire_no_discovery(
        monkeypatch,
        accounts=[SimpleNamespace(account_id="acc-1", provider_id="slack", connected=True)],
    )

    denial = await connect_first_denial_for_identity(
        grantor_user_id="user-1",
        agent_client_id="kdcube-agent:app:main",
        agent_resource="*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/productivity*",
        namespace="productivity_slack_search",
        tool="productivity_slack_search",
        operation="search",
        required=["slack:search"],
        missing=["slack:search"],
        tenant="t",
        project="p",
        requirements=[{"provider_id": "slack", "claims": ["slack:search"]}],
    )

    assert denial is None


@pytest.mark.asyncio
async def test_slack_action_still_leads_with_connect(monkeypatch):
    """An account-touching action (post) whose claim is missing MUST still lead
    with connect - the metadata-op skip never suppresses a real account need."""
    _wire(monkeypatch, requirements=[_SLACK_REQUIREMENT], accounts=[])

    denial = await connect_first_denial(
        object(),
        namespace="slack",
        tool="action",
        operation="object.action",
        required=["named_services:use", "slack:post"],
        missing=["named_services:use", "slack:post"],
        tenant="t",
        project="p",
    )

    assert denial is not None
    assert denial["reason"] == "connect_required"
    assert denial["consent"].get("claims") == ["slack:post"]


@pytest.mark.asyncio
async def test_mail_send_action_still_leads_with_connect(monkeypatch):
    """The differentiated mail realm maps object.action.send -> gmail:send; a
    send with no account MUST lead with connect for exactly that claim."""
    _wire(monkeypatch, requirements=[_MAIL_REQUIREMENT], accounts=[])

    denial = await connect_first_denial(
        object(),
        namespace="mail",
        tool="action",
        operation="object.action.send",
        required=["mail:send", "named_services:use"],
        missing=["mail:send", "named_services:use"],
        tenant="t",
        project="p",
    )

    assert denial is not None
    assert denial["reason"] == "connect_required"
    assert denial["consent"].get("claims") == ["gmail:send"]
