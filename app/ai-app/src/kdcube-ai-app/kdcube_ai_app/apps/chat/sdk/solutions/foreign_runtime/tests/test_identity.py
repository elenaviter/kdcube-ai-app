# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""The shared multi-user + multi-agent isolation gate (foreign_runtime/identity.py).

A foreign agent was single-user. One deployment is tenant/project-bound, but one
process serves many users and one app can host several agents. ``identity.py``
keeps the keys partitioned by deployment, user, conversation, and active agent
id. Ported from the ported-langgraph-agents bundle's tests/test_identity_isolation.py
(stdlib-only, no DB / API).
"""
from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.identity import turn_identity


def test_same_raw_user_in_different_tenants_gets_different_keys() -> None:
    a = turn_identity(
        {"tenant": "t1", "project": "p", "user": "alice", "conversation_id": "c1"},
        agent_id="lg-solution",
    )
    b = turn_identity(
        {"tenant": "t2", "project": "p", "user": "alice", "conversation_id": "c1"},
        agent_id="lg-solution",
    )
    assert a.user_id != b.user_id
    assert a.user_id == "t1:p:lg-solution:alice"
    assert b.user_id == "t2:p:lg-solution:alice"


def test_same_raw_user_in_different_projects_gets_different_keys() -> None:
    a = turn_identity({"tenant": "t", "project": "p1", "user": "alice"}, agent_id="lg-solution")
    b = turn_identity({"tenant": "t", "project": "p2", "user": "alice"}, agent_id="lg-solution")
    assert a.user_id != b.user_id


def test_two_agents_get_different_keys_for_the_same_user_and_conversation() -> None:
    """The multi-agent invariant: the SAME (tenant, project, user, conversation)
    resolves to DIFFERENT per-user + per-conversation keys under two agents, so
    one agent's memory can never bleed into the other's."""
    state = {"tenant": "t", "project": "p", "user": "alice", "conversation_id": "c1"}
    sol = turn_identity(state, agent_id="lg-solution")
    pre = turn_identity(state, agent_id="lg-react")

    assert sol.user_id != pre.user_id
    assert sol.thread_id != pre.thread_id
    assert sol.user_id == "t:p:lg-solution:alice"
    assert pre.user_id == "t:p:lg-react:alice"
    assert sol.agent_id == "lg-solution"
    assert pre.agent_id == "lg-react"


def test_thread_id_is_scoped_by_user_and_agent() -> None:
    ident = turn_identity(
        {"tenant": "t", "project": "p", "user": "alice", "conversation_id": "conv-42"},
        agent_id="lg-react",
    )
    assert ident.thread_id == "t:p:lg-react:alice:conv-42"


def test_shared_conversation_id_across_users_never_collides() -> None:
    a = turn_identity(
        {"tenant": "t", "project": "p", "user": "alice", "conversation_id": "shared"},
        agent_id="lg-solution",
    )
    b = turn_identity(
        {"tenant": "t", "project": "p", "user": "bob", "conversation_id": "shared"},
        agent_id="lg-solution",
    )
    assert a.thread_id != b.thread_id


def test_session_id_and_fallback_thread_id() -> None:
    by_session = turn_identity(
        {"tenant": "t", "project": "p", "user": "alice", "session_id": "sess-7"},
        agent_id="lg-solution",
    )
    assert by_session.thread_id == "t:p:lg-solution:alice:sess-7"
    by_fallback = turn_identity(
        {"tenant": "t", "project": "p", "user": "alice"},
        agent_id="lg-solution",
        fallback_thread_id="thread-9",
    )
    assert by_fallback.thread_id == "t:p:lg-solution:alice:thread-9"


def test_anonymous_fallback_and_blank_agent() -> None:
    # No user, no fingerprint -> "anonymous".
    anon = turn_identity({"tenant": "t", "project": "p"}, agent_id="lg-solution")
    assert anon.user_id == "t:p:lg-solution:anonymous"
    # Fingerprint is used when no resolved user is present.
    fp = turn_identity({"tenant": "t", "project": "p", "fingerprint": "fp-123"}, agent_id="lg-solution")
    assert fp.user_id == "t:p:lg-solution:fp-123"
    # A blank agent id folds to "default" so keys stay deterministic.
    bare = turn_identity({}, agent_id="")
    assert bare.user_id == "t:p:default:anonymous"
    assert bare.thread_id == "t:p:default:anonymous:default"


def test_missing_tenant_project_use_safe_placeholders() -> None:
    ident = turn_identity({"user": "alice"}, agent_id="lg-react")
    assert ident.user_id == "t:p:lg-react:alice"
    assert ident.thread_id == "t:p:lg-react:alice:default"


def test_externally_scoped_user_gets_an_isolated_memory_key() -> None:
    # A secondary ingress (e.g. Telegram) resolves a sender to a scoped platform
    # user like `telegram_<id>` before the turn, so state["user"] is already
    # that scoped id. It folds identically.
    tg = turn_identity(
        {"tenant": "t", "project": "p", "user": "telegram_12345", "conversation_id": "telegram_chat_7"},
        agent_id="lg-solution",
    )
    assert tg.user_id == "t:p:lg-solution:telegram_12345"

    browser = turn_identity({"tenant": "t", "project": "p", "user": "12345"}, agent_id="lg-solution")
    other_tg = turn_identity({"tenant": "t", "project": "p", "user": "telegram_99999"}, agent_id="lg-solution")
    assert tg.user_id != browser.user_id
    assert tg.user_id != other_tg.user_id
