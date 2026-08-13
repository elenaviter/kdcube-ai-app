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
