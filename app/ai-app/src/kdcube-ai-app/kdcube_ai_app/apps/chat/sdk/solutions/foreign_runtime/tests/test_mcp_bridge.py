# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""The runtime-neutral delegated-MCP bridge (foreign_runtime/mcp_bridge.py).

Offline: `resolve_mcp_server_map` is stubbed via monkeypatch — no network, no
hub. What is proven: the agent's delegated-client id derivation, the drop_sink
pass-through, and the connect-required outcome shaping (consent_pending drops
produce the payload; operational drop reasons never do).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_mcp import (
    DROP_CONSENT_PENDING,
    DROP_MINT_ERROR,
    DROP_NO_USER,
    DROP_PROVIDER_ERROR,
    delegated_client_id_for_agent,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import mcp_bridge


# ── connect_required_outcome (pure shaping) ──────────────────────────────────

def test_consent_pending_drops_shape_the_connect_required_payload() -> None:
    drop_sink = {
        "gmail": DROP_CONSENT_PENDING,
        "calendar": DROP_CONSENT_PENDING,
    }
    outcome = mcp_bridge.connect_required_outcome(
        drop_sink, connection_hub_url="https://hub.example/connections"
    )
    assert outcome == {
        "status": "connect_required",
        "connections": ["gmail", "calendar"],
        "connection_hub_url": "https://hub.example/connections",
    }


def test_operational_drop_reasons_are_excluded() -> None:
    drop_sink = {
        "gmail": DROP_CONSENT_PENDING,
        "broken": DROP_PROVIDER_ERROR,
        "anon": DROP_NO_USER,
        "mintless": DROP_MINT_ERROR,
    }
    outcome = mcp_bridge.connect_required_outcome(drop_sink)
    assert outcome is not None
    assert outcome["connections"] == ["gmail"]
    assert outcome["connection_hub_url"] is None


def test_no_consent_pending_drops_yield_none() -> None:
    assert mcp_bridge.connect_required_outcome({}) is None
    assert mcp_bridge.connect_required_outcome(None) is None
    assert mcp_bridge.connect_required_outcome({"broken": DROP_PROVIDER_ERROR}) is None


# ── resolve_turn_mcp (stubbed resolver) ──────────────────────────────────────

def test_resolve_turn_mcp_derives_the_agent_client_id_and_passes_the_drop_sink(monkeypatch) -> None:
    captured: dict = {}

    async def _stub_resolver(connections, *, user_sub=None, client_id="", bearer_provider=None, drop_sink=None, **kw):
        captured["connections"] = connections
        captured["user_sub"] = user_sub
        captured["client_id"] = client_id
        captured["bearer_provider"] = bearer_provider
        captured["drop_sink"] = drop_sink
        if drop_sink is not None:
            drop_sink["gmail"] = DROP_CONSENT_PENDING
        return {"tasks": {"url": "https://mcp.example/tasks", "transport": "streamable_http"}}

    monkeypatch.setattr(mcp_bridge, "resolve_mcp_server_map", _stub_resolver)

    conns = [{"kind": "mcp", "server_id": "tasks", "url": "https://mcp.example/tasks"}]
    drop_sink: dict = {}
    entrypoint = SimpleNamespace()  # untouched: user_sub given, resolver stubbed
    server_map = asyncio.run(mcp_bridge.resolve_turn_mcp(
        entrypoint, conns,
        agent_id="lg-react", application="my-app@1-0",
        user_sub="user-1", drop_sink=drop_sink,
    ))

    # The server map is the resolver's, untouched.
    assert set(server_map) == {"tasks"}
    # client_id derivation: the agent IS a delegated-client entity keyed by
    # application + agent_id.
    expected = delegated_client_id_for_agent("my-app@1-0", "lg-react")
    assert captured["client_id"] == expected == "kdcube-agent:my-app@1-0:lg-react"
    # The SAME drop_sink dict is passed through, so the caller reads the drops.
    assert captured["drop_sink"] is drop_sink
    assert drop_sink == {"gmail": DROP_CONSENT_PENDING}
    # Explicit user_sub is honored verbatim; a bearer provider is always wired.
    assert captured["user_sub"] == "user-1"
    assert callable(captured["bearer_provider"])
    assert captured["connections"] == conns
    # ...and the drops shape the structured outcome.
    assert mcp_bridge.connect_required_outcome(drop_sink) == {
        "status": "connect_required",
        "connections": ["gmail"],
        "connection_hub_url": None,
    }


def test_resolve_turn_mcp_resolves_the_user_from_the_turn_context_when_unset(monkeypatch) -> None:
    captured: dict = {}

    async def _stub_resolver(connections, *, user_sub=None, **kw):
        captured["user_sub"] = user_sub
        return {}

    monkeypatch.setattr(mcp_bridge, "resolve_mcp_server_map", _stub_resolver)
    monkeypatch.setattr(mcp_bridge, "current_turn_user_sub", lambda ep: "ctx-user")

    asyncio.run(mcp_bridge.resolve_turn_mcp(
        SimpleNamespace(), [], agent_id="a", application="app",
    ))
    assert captured["user_sub"] == "ctx-user"


# ── current_turn_user_sub ────────────────────────────────────────────────────

def test_current_turn_user_sub_is_empty_when_nothing_is_bound() -> None:
    class _NoTurn:
        @property
        def comm(self):
            raise RuntimeError("no turn task bound")

    # Outside a turn: no accounting context user, comm raises -> "".
    assert mcp_bridge.current_turn_user_sub(_NoTurn()) == ""


def test_current_turn_user_sub_falls_back_to_the_comm_user() -> None:
    ep = SimpleNamespace(comm=SimpleNamespace(user_id="comm-user"))
    assert mcp_bridge.current_turn_user_sub(ep) == "comm-user"


# ── announce_connect_required (the house consent card) ───────────────────────

_CONN = {
    "name": "press",
    "kind": "mcp",
    "server_id": "press",
    "url": "https://host/mcp/press",
    "resource": "*/mcp/press*",
    "delegated": True,
    "scopes": ["press:read"],
}


def _announce_recorder(monkeypatch) -> list:
    """Capture what the seam hands the SDK's consent announcer."""
    captured: list = []

    async def _announce(consent):
        captured.append(consent)

    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.solutions.connections.mcp_consent.announce_agent_consent",
        _announce,
    )
    return captured


def test_consent_pending_drops_raise_one_demand_per_connection(monkeypatch) -> None:
    captured = _announce_recorder(monkeypatch)

    announced = asyncio.run(mcp_bridge.announce_connect_required(
        SimpleNamespace(), [dict(_CONN)], {"press": DROP_CONSENT_PENDING},
        agent_id="press", application="my-app@1-0", tenant="t1", project="p1",
    ))

    assert [item["server_id"] for item in announced] == ["press"]
    assert len(captured) == 1
    consent = captured[0]
    # The demand is the SAME MCPConsentRequired a KDCube-MCP 403 produces, so
    # the chat's one banner path serves both.
    payload = consent.chat_event_payload()
    assert payload["error"]["code"] == "needs_connected_account_consent"
    block = payload["consent"]
    assert block["agent_client_id"] == "kdcube-agent:my-app@1-0:press"
    assert block["claims"] == ["press:read"]
    assert block["resource"] == "*/mcp/press*"
    assert block["grant"]["operation"] == "delegated_agent_grant_create"


def test_operational_drops_raise_no_demand(monkeypatch) -> None:
    captured = _announce_recorder(monkeypatch)

    announced = asyncio.run(mcp_bridge.announce_connect_required(
        SimpleNamespace(), [dict(_CONN)], {"press": DROP_PROVIDER_ERROR},
        agent_id="press", application="my-app@1-0",
    ))

    assert announced == []
    assert captured == []
    assert asyncio.run(mcp_bridge.announce_connect_required(
        SimpleNamespace(), [dict(_CONN)], {}, agent_id="press", application="my-app@1-0",
    )) == []


def test_the_hub_link_is_the_platforms_and_absent_without_a_public_base(monkeypatch) -> None:
    """The seam never hand-writes a hub path: the link comes from the platform
    builder over the deployment's public base URL, so with none bound the demand
    simply carries no URL (the card still acts through its grant fields)."""
    captured = _announce_recorder(monkeypatch)
    calls: list = []

    def _hub_url(*, tenant, project, client_id, resource, claims, **kw):
        calls.append({"tenant": tenant, "project": project, "client_id": client_id,
                      "resource": resource, "claims": list(claims)})
        return ""

    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials"
        ".consent_denial.connection_hub_grant_url",
        _hub_url,
    )

    asyncio.run(mcp_bridge.announce_connect_required(
        SimpleNamespace(), [dict(_CONN)], {"press": DROP_CONSENT_PENDING},
        agent_id="press", application="my-app@1-0", tenant="t1", project="p1",
    ))

    assert calls == [{
        "tenant": "t1", "project": "p1",
        "client_id": "kdcube-agent:my-app@1-0:press",
        "resource": "*/mcp/press*", "claims": ["press:read"],
    }]
    assert captured[0].chat_event_payload()["consent"]["url"] == ""
