# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Reactive-event lane finalization (run-to-completion turn path).

The module logic is exercised with lightweight fakes for the lane source and a
real orchestrator over an in-memory state table, so the consumer transitions are
real. The door tests drive ``BaseEntrypoint.run`` to prove the finalize fires on
success and error but not on cancel.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

import kdcube_ai_app.apps.chat.sdk.solutions.chatbot.reactive_lane as rl
from kdcube_ai_app.apps.chat.sdk.events.event_bus.orchestrator import (
    ConversationEventBusOrchestrator,
)
from kdcube_ai_app.apps.chat.sdk.events.event_bus.state import EventLaneState


_OWN_TS = "2026-07-13T11:00:00Z"
_FOLLOWUP_TS = "2026-07-13T11:00:05Z"  # landed DURING the turn


# ── fakes ────────────────────────────────────────────────────────────────────

class _MemoryLaneStateTable:
    def __init__(self, state: EventLaneState, actions: list[str] | None = None) -> None:
        self.state = state
        self.actions = actions

    async def get(self) -> EventLaneState:
        return self.state

    async def put(self, state: EventLaneState) -> EventLaneState:
        self.state = state
        return state

    async def update(self, mutator, **kwargs):
        if self.actions is not None:
            self.actions.append(str(kwargs.get("operation") or "update"))
        self.state = mutator(self.state) or self.state
        return self.state

    @contextlib.asynccontextmanager
    async def lock(self, **_kwargs):
        yield "memory-lock"


class _Event(SimpleNamespace):
    def task_payload_model(self):
        return {"event_id": self.message_id}


def _event(*, ts, message_id, sequence, consumed_at=None, promoted_at=None, reactive=True):
    return _Event(
        message_id=message_id,
        sequence=sequence,
        created_at=ts,
        consumed_at=consumed_at,
        promoted_at=promoted_at,
        failed_at=None,
        payload={"event": {"timestamp": ts, "reactive": reactive}},
    )


class _FakeSource:
    def __init__(self, events):
        self._by_id = {e.message_id: e for e in events}
        self._list = list(events)
        self.tenant = "tenant-a"
        self.project = "project-a"
        self.user_id = "user-1"
        self.conversation_id = "conv-1"
        self.agent_id = "agent-x"
        self.consumed_calls = []
        self.consumed_event_calls = []

    async def get_event(self, message_id):
        return self._by_id.get(message_id)

    async def read_since(self, cursor, *, limit=None):
        del cursor, limit
        return list(self._list)

    async def mark_consumed_up_to(self, *, max_sequence, turn_id):
        self.consumed_calls.append((max_sequence, turn_id))
        updated = 0
        for event in self._list:
            if int(event.sequence or 0) <= int(max_sequence) and event.consumed_at is None:
                event.consumed_at = 1.0
                updated += 1
        return updated

    async def mark_consumed_event(self, *, message_id, turn_id):
        self.consumed_event_calls.append((message_id, turn_id))
        event = self._by_id.get(message_id)
        if event is None:
            return None
        if event.consumed_at is None:
            event.consumed_at = 1.0
        return event


def _install(
    monkeypatch,
    *,
    state: EventLaneState,
    source: _FakeSource,
    published: list,
    actions: list[str] | None = None,
):
    """Wire the module's builders to fakes; return the real orchestrator so a
    test can inspect the post-finalize consumer state."""
    orchestrator = ConversationEventBusOrchestrator(table=_MemoryLaneStateTable(state, actions))

    class _FakePublisher:
        def __init__(self, _enqueuer=None):
            pass

        async def publish_for_event(self, *, payload, event, **kwargs):
            del payload, kwargs
            published.append(event.message_id)
            if actions is not None:
                actions.append(f"publish:{event.message_id}")
            return SimpleNamespace(success=True, reason="queued")

    wakeup = SimpleNamespace(
        event_lane=SimpleNamespace(event_id="evt-own"),
        actor=SimpleNamespace(tenant_id="tenant-a", project_id="project-a"),
        routing=SimpleNamespace(turn_id="turn-1"),
    )
    monkeypatch.setattr(rl, "_lane_wakeup_from_comm_context", lambda comm_context: wakeup)
    monkeypatch.setattr(rl, "_source_for_wakeup", lambda redis, wk: source)
    monkeypatch.setattr(rl.ConversationEventBusOrchestrator, "for_source", staticmethod(lambda src: orchestrator))
    monkeypatch.setattr(rl, "EventLaneWakePublisher", _FakePublisher)
    return orchestrator


def _comm_context():
    return SimpleNamespace(
        bundle_call_context={"event_lane_wakeup": {"event_lane": {"event_id": "evt-own"}}},
        routing=SimpleNamespace(turn_id="turn-1"),
    )


# ── module logic ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_to_completion_release_frees_consumer_and_consumes_own_event(monkeypatch):
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    source = _FakeSource([own])
    published: list = []
    # Run-to-completion left the reservation dangling (consumer "scheduled").
    orchestrator = _install(
        monkeypatch,
        state=EventLaneState(consumer_status="scheduled", consumer_status_at="2026-07-13T11:00:00Z"),
        source=source,
        published=published,
    )

    result = await rl.finalize_reactive_event_lane(redis=object(), comm_context=_comm_context())

    assert result is True
    assert source.consumed_calls == [(1, "turn-1")]        # own event consumed (exactly-once)
    assert (await orchestrator.state()).consumer_status == "none"  # reservation released


@pytest.mark.asyncio
async def test_release_rewakes_mid_turn_followup_not_own_event(monkeypatch):
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    followup = _event(ts=_FOLLOWUP_TS, message_id="evt-followup", sequence=2)
    source = _FakeSource([own, followup])
    published: list = []
    _install(
        monkeypatch,
        state=EventLaneState(consumer_status="scheduled", consumer_status_at="2026-07-13T11:00:00Z"),
        source=source,
        published=published,
    )

    await rl.finalize_reactive_event_lane(redis=object(), comm_context=_comm_context())

    # The mid-turn followup is re-woken (promoted to the next turn); the turn's
    # own event is never re-woken (no double run).
    assert published == ["evt-followup"]


@pytest.mark.asyncio
async def test_release_consumes_folded_ingress_batch_without_rewake(monkeypatch):
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    attachment = _event(ts=_FOLLOWUP_TS, message_id="evt-attachment", sequence=2)
    source = _FakeSource([own, attachment])
    published: list = []
    _install(
        monkeypatch,
        state=EventLaneState(consumer_status="scheduled", consumer_status_at="2026-07-13T11:00:00Z"),
        source=source,
        published=published,
    )

    await rl.finalize_reactive_event_lane(
        redis=object(),
        comm_context=_comm_context(),
        consumed_event_ids=["evt-own", "evt-attachment"],
    )

    assert source.consumed_calls == [(1, "turn-1")]
    assert source.consumed_event_calls == [("evt-attachment", "turn-1")]
    assert own.consumed_at is not None
    assert attachment.consumed_at is not None
    assert published == []


@pytest.mark.asyncio
async def test_release_consumes_folded_ids_without_swallowing_interleaved_followup(monkeypatch):
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    followup = _event(ts=_FOLLOWUP_TS, message_id="evt-followup", sequence=2)
    attachment = _event(ts=_FOLLOWUP_TS, message_id="evt-attachment", sequence=3)
    source = _FakeSource([own, followup, attachment])
    published: list = []
    _install(
        monkeypatch,
        state=EventLaneState(consumer_status="scheduled", consumer_status_at="2026-07-13T11:00:00Z"),
        source=source,
        published=published,
    )

    await rl.finalize_reactive_event_lane(
        redis=object(),
        comm_context=_comm_context(),
        consumed_event_ids=["evt-own", "evt-attachment"],
    )

    assert source.consumed_calls == [(1, "turn-1")]
    assert source.consumed_event_calls == [("evt-attachment", "turn-1")]
    assert own.consumed_at is not None
    assert attachment.consumed_at is not None
    assert followup.consumed_at is None
    assert published == ["evt-followup"]


@pytest.mark.asyncio
async def test_release_precedes_liveness_rewake(monkeypatch):
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    followup = _event(ts=_FOLLOWUP_TS, message_id="evt-followup", sequence=2)
    source = _FakeSource([own, followup])
    published: list[str] = []
    actions: list[str] = []
    _install(
        monkeypatch,
        state=EventLaneState(consumer_status="scheduled", consumer_status_at=_OWN_TS),
        source=source,
        published=published,
        actions=actions,
    )

    await rl.finalize_reactive_event_lane(redis=object(), comm_context=_comm_context())

    assert actions == ["mark_consumer_none", "publish:evt-followup"]


@pytest.mark.asyncio
async def test_folded_followup_is_not_rewoken(monkeypatch):
    """The surfaced regression: a ReAct turn FOLDED a mid-turn followup (advancing
    the lane's ``last_processed_reactive_event_timestamp`` cursor past it) but never
    set ``consumed_at`` on it. If finalize's re-wake runs (the reservation was still
    held — a race/close-path where mark_consumer_none hadn't landed), it must NOT
    re-wake that folded followup: doing so re-runs it as a second turn (the
    duplicate user bubble + answer). The skip honors the SAME cursor ReAct uses."""
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    followup = _event(ts=_FOLLOWUP_TS, message_id="evt-followup", sequence=2)
    source = _FakeSource([own, followup])
    published: list = []
    _install(
        monkeypatch,
        state=EventLaneState(
            consumer_status="scheduled",              # reservation still held (re-wake path runs)
            last_processed_reactive_event_timestamp=_FOLLOWUP_TS,  # ReAct folded the followup
        ),
        source=source,
        published=published,
    )

    await rl.finalize_reactive_event_lane(redis=object(), comm_context=_comm_context())

    assert published == []  # the folded followup is NOT re-woken → no double turn


@pytest.mark.asyncio
async def test_noop_when_reservation_released_and_own_event_accounted_react_state(monkeypatch):
    """The post-ReAct lane state: consumer already ``none`` and the reactive
    cursor already past the own event. Finalize is inert — no re-release, no
    re-wake — with NO agent-type check."""
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    source = _FakeSource([own])
    published: list = []
    orchestrator = _install(
        monkeypatch,
        state=EventLaneState(
            consumer_status="none",
            last_processed_reactive_event_timestamp=_OWN_TS,  # cursor past own event (ReAct advanced it)
        ),
        source=source,
        published=published,
    )

    result = await rl.finalize_reactive_event_lane(redis=object(), comm_context=_comm_context())

    assert result is False
    assert source.consumed_calls == []
    assert published == []
    assert (await orchestrator.state()).consumer_status == "none"


@pytest.mark.asyncio
async def test_release_is_idempotent_on_second_call(monkeypatch):
    own = _event(ts=_OWN_TS, message_id="evt-own", sequence=1)
    source = _FakeSource([own])
    published: list = []
    _install(
        monkeypatch,
        state=EventLaneState(consumer_status="scheduled", consumer_status_at="2026-07-13T11:00:00Z"),
        source=source,
        published=published,
    )
    comm = _comm_context()

    first = await rl.finalize_reactive_event_lane(redis=object(), comm_context=comm)
    second = await rl.finalize_reactive_event_lane(redis=object(), comm_context=comm)

    assert first is True
    assert second is False  # consumer now none + own event consumed → inert


@pytest.mark.asyncio
async def test_noop_when_turn_was_not_a_lane_wakeup(monkeypatch):
    monkeypatch.setattr(rl, "_lane_wakeup_from_comm_context", lambda comm_context: None)
    result = await rl.finalize_reactive_event_lane(
        redis=object(),
        comm_context=SimpleNamespace(bundle_call_context={}),
    )
    assert result is False


# ── the door: BaseEntrypoint.run() finally ───────────────────────────────────

@pytest.mark.asyncio
async def test_base_run_finalizes_lane_on_success_and_error_but_not_cancel(monkeypatch):
    import asyncio

    import kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint as entrypoint_mod
    from kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint import BaseEntrypoint
    import kdcube_ai_app.infra.accounting as accounting_mod
    import kdcube_ai_app.apps.chat.sdk.runtime.turn_recording as turn_recording_mod

    calls: list[str] = []

    async def _recorder(**kwargs):
        calls.append("finalize")
        return True

    monkeypatch.setattr(entrypoint_mod, "finalize_reactive_event_lane", _recorder)
    monkeypatch.setattr(entrypoint_mod, "create_storage_backend", lambda *a, **k: object())
    monkeypatch.setattr(turn_recording_mod, "reset_turn_log_recorded", lambda: None)

    @contextlib.asynccontextmanager
    async def _noop_accounting(*a, **k):
        yield

    monkeypatch.setattr(accounting_mod, "with_accounting", _noop_accounting)
    monkeypatch.setattr(accounting_mod, "_get_storage", lambda: SimpleNamespace())
    monkeypatch.setattr(accounting_mod.AccountingSystem, "init_storage", staticmethod(lambda *a, **k: None))

    def _build(execute_core):
        ep = object.__new__(BaseEntrypoint)
        ep._app_state = {}
        ep._turn_id = "turn-1"
        ep.config = SimpleNamespace(ai_bundle_spec=SimpleNamespace(id="bundle@1"), tenant=None, project=None)
        ep.settings = SimpleNamespace(TENANT="tenant-a", PROJECT="project-a")
        ep.comm_context = SimpleNamespace(
            actor=SimpleNamespace(tenant_id="tenant-a", project_id="project-a"),
            user=SimpleNamespace(user_id="u", fingerprint="fp", timezone=None),
            request=SimpleNamespace(request_id="req-1"),
            event=SimpleNamespace(agent_id=None),
            routing=SimpleNamespace(turn_id="turn-1"),
            bundle_call_context={},
        )
        ep.logger = SimpleNamespace(log=lambda *a, **k: None)
        ep.redis = None

        async def _noop(*a, **k):
            return None

        ep.refresh_bundle_props = _noop
        ep.pre_run_hook = _noop
        ep.run_accounting = _noop
        ep.post_run_hook = _noop
        ep._record_turn_log_fallback = _noop
        ep.project_app_state = lambda result: result
        ep.execute_core = execute_core
        return ep

    # success
    async def _ok(*, state, thread_id, params):
        return {"final_answer": "hi"}

    calls.clear()
    await _build(_ok).run()
    assert calls == ["finalize"]

    # error
    async def _boom(*, state, thread_id, params):
        raise RuntimeError("boom")

    calls.clear()
    with pytest.raises(RuntimeError):
        await _build(_boom).run()
    assert calls == ["finalize"]

    # cancel → skipped (stays on the recovery path)
    async def _cancel(*, state, thread_id, params):
        raise asyncio.CancelledError()

    calls.clear()
    with pytest.raises(asyncio.CancelledError):
        await _build(_cancel).run()
    assert calls == []
