# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Run-to-completion turn recording (foreign_runtime/turn_record.py).

Offline, no platform: stub entrypoints. What is proven honestly here:
persist_turn_artifacts' record-user re-scoping (economics/authority projection
fallback, caller state never mutated) and its tolerance of empty state; the
timing emit's fail-open posture (a comm that raises never breaks the turn) and
its event shape; the title/is-new probes' fail-open skips.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.turn_record import (
    conversation_is_new,
    emit_turn_timing,
    finalize_conversation_title,
    persist_turn_artifacts,
)


class _RecordingEntrypoint:
    """The two base persistence methods, recording what state they were given
    (the real ones on BaseEntrypoint swallow their own failures)."""

    def __init__(self) -> None:
        self.events_states: list[dict] = []
        self.stream_states: list[dict] = []

    async def _save_events_artifact(self, *, state: dict) -> None:
        self.events_states.append(state)

    async def _persist_stream_artifacts_fallback(self, *, state: dict) -> None:
        self.stream_states.append(state)


# ── persist_turn_artifacts ───────────────────────────────────────────────────

def test_persist_rescopes_the_record_user_from_the_economics_projection() -> None:
    ep = _RecordingEntrypoint()
    state = {
        "user": "",
        "economics_user": "econ-user",
        "authority_user": "auth-user",
        "conversation_id": "c1",
    }
    original = dict(state)

    asyncio.run(persist_turn_artifacts(ep, state))

    # Both artifacts saved under the SAME re-scoped state; economics_user wins.
    assert len(ep.events_states) == 1 and len(ep.stream_states) == 1
    assert ep.events_states[0]["user"] == "econ-user"
    assert ep.stream_states[0] is ep.events_states[0]
    # The caller's state is never mutated.
    assert state == original


def test_persist_fallback_order_authority_then_actor_then_fingerprint() -> None:
    ep = _RecordingEntrypoint()
    asyncio.run(persist_turn_artifacts(ep, {"authority_user": "auth-user", "actor_user": "actor"}))
    assert ep.events_states[0]["user"] == "auth-user"

    ep2 = _RecordingEntrypoint()
    asyncio.run(persist_turn_artifacts(ep2, {"actor_user": "actor", "fingerprint": "fp-1"}))
    assert ep2.events_states[0]["user"] == "actor"

    ep3 = _RecordingEntrypoint()
    asyncio.run(persist_turn_artifacts(ep3, {"fingerprint": "fp-1"}))
    assert ep3.events_states[0]["user"] == "fp-1"


def test_persist_keeps_a_resolved_user_untouched() -> None:
    ep = _RecordingEntrypoint()
    state = {"user": "alice", "economics_user": "econ-user"}
    asyncio.run(persist_turn_artifacts(ep, state))
    # `user` already set -> the state object passes through unchanged.
    assert ep.events_states[0] is state
    assert ep.events_states[0]["user"] == "alice"


def test_persist_is_tolerant_of_an_empty_state() -> None:
    ep = _RecordingEntrypoint()
    state: dict = {}
    asyncio.run(persist_turn_artifacts(ep, state, result={"answer": "x"}))
    # Nothing to re-scope from -> the empty state passes through as-is; the
    # base methods (fail-open themselves) still both run.
    assert ep.events_states == [state]
    assert ep.stream_states == [state]


# ── emit_turn_timing ─────────────────────────────────────────────────────────

def test_emit_turn_timing_authors_the_react_shaped_summary_event() -> None:
    events: list[dict] = []

    class _Comm:
        async def event(self, **kwargs) -> None:
            events.append(kwargs)

    ep = SimpleNamespace(comm=_Comm())
    asyncio.run(emit_turn_timing(ep, started_ms=1000, total_ms=250))

    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "chat.turn.summary"
    assert evt["route"] == "chat.step"
    assert evt["step"] == "turn.summary"
    assert evt["status"] == "completed"
    assert evt["agent"] == "turn_controller"
    assert evt["data"]["elapsed_ms"] == 250
    assert evt["data"]["started_ms"] == 1000
    assert evt["data"]["ended_ms"] >= 1000


def test_emit_turn_timing_fails_open_when_the_comm_raises() -> None:
    class _NoComm:
        @property
        def comm(self):
            raise RuntimeError("no turn task bound")

    # Never raises; the turn stands.
    asyncio.run(emit_turn_timing(_NoComm(), started_ms=1, total_ms=2))


def test_emit_turn_timing_is_inert_with_no_comm() -> None:
    asyncio.run(emit_turn_timing(SimpleNamespace(comm=None), started_ms=1, total_ms=2))


# ── finalize_conversation_title / conversation_is_new ────────────────────────

def test_title_skips_without_a_model_service_and_leaves_state_untouched() -> None:
    ep = SimpleNamespace(models_service=None)
    state = {"conversation_id": "c1"}
    asyncio.run(finalize_conversation_title(
        ep, state, conversation_id="c1", question="hello", title_role="a.answer",
    ))
    assert "conversation_title" not in state


def test_conversation_is_new_fails_safe_without_a_ctx_client() -> None:
    class _EP:
        async def get_ctx_client(self):
            return None

    assert asyncio.run(conversation_is_new(_EP(), {}, conversation_id="c1")) is False


def test_conversation_is_new_fails_safe_when_the_probe_raises() -> None:
    class _EP:
        async def get_ctx_client(self):
            raise RuntimeError("pool down")

    assert asyncio.run(conversation_is_new(_EP(), {"user": "u"}, conversation_id="c1")) is False


def test_conversation_is_new_reads_under_the_economics_user() -> None:
    calls: list[dict] = []

    class _Client:
        async def recent(self, **kwargs):
            calls.append(kwargs)
            return {"items": []}

    class _EP:
        async def get_ctx_client(self):
            return _Client()

    state = {"economics_user": "econ-user", "user": "raw-user"}
    is_new = asyncio.run(conversation_is_new(_EP(), state, conversation_id="c1"))
    assert is_new is True
    assert calls[0]["user_id"] == "econ-user"
    assert calls[0]["conversation_id"] == "c1"
    assert calls[0]["kinds"] == ["artifact:turn.log"]


def test_the_generated_title_is_where_the_recorder_looks():
    """LIVE: every hosted-runtime conversation listed as "Untitled conversation"
    while its live header showed the generated name. The seam names the
    conversation BEFORE the agent runs — so the name survives a failed turn —
    and leaves it on `state`; the recorder read only `result`. The two ends must
    agree, and this pins which key carries it."""
    import inspect
    from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import turn_record

    source = inspect.getsource(turn_record.finalize_conversation_title)
    assert 'state["conversation_title"] = title' in source

    from kdcube_ai_app.apps.chat.sdk.solutions.chatbot import entrypoint as base
    recorder = inspect.getsource(base.BaseEntrypoint._record_turn_log_fallback)
    assert '(state or {}).get("conversation_title")' in recorder
