# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Watching the lane while a foreign-loop turn runs.

The turn folds the lane once, at its start, and is then deaf until it ends — so
a message typed mid-run, including the stop control (a steer with no text),
reaches nobody for as long as the turn lasts. This watcher is the listening
half: read-only, deduped, lane-ordered, and with no opinion about what to DO,
because interrupting means cancelling a task in one runtime and denying a tool
call in another.

Offline: the lane source is faked; no redis, no store, no network.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kdcube_ai_app.apps.chat.external_events import ConversationExternalEvent
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import live_watch as mod


def _accepted(text: str, *, event_id: str, etype: str = "event.user.followup") -> dict:
    return {
        "event_id": event_id,
        "type": etype,
        "reactive": True,
        "payload": {"mime": "text/plain", "event": {"text": text}},
    }


def _lane_event(*, message_id, sequence, accepted, batch_id="batch-1", consumed_at=None):
    return ConversationExternalEvent(
        message_id=message_id,
        batch_id=batch_id,
        kind="external_event",
        created_at=1000.0 + sequence,
        sequence=sequence,
        payload={"text": "", "event": dict(accepted), "is_continuation": False},
        consumed_at=consumed_at,
    )


class _FakeSource:
    """A lane that grows while the turn runs, like the real one."""

    def __init__(self, events):
        self._events = list(events)

    def append(self, event):
        self._events.append(event)

    async def get_event(self, message_id):
        for item in self._events:
            if item.message_id == message_id:
                return item
        return None

    async def read_since(self, cursor, *, limit=None):
        return list(self._events)


def _entrypoint():
    return SimpleNamespace(
        redis=object(),
        comm_context=SimpleNamespace(bundle_call_context={"event_lane_wakeup": {
            "meta": {"task_id": "task-1", "created_at": 1000.0},
            "routing": {"conversation_id": "conv-1", "session_id": "s-1", "bundle_id": "b-1"},
            "actor": {"tenant_id": "t", "project_id": "p"},
            "user": {"user_id": "u", "user_type": "registered"},
            "event_lane": {"tenant": "t", "project": "p", "conversation_id": "conv-1",
                           "user_id": "u", "agent_id": "press", "event_id": "m-own"},
        }}),
    )


def _run(coro):
    return asyncio.run(coro)


def test_a_message_typed_mid_run_is_seen_while_the_turn_is_still_going(monkeypatch):
    """The whole point: the turn does not have to end for this to be known."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)
    delivered: list = []

    async def scenario():
        async with mod.LiveLaneWatch(
            _entrypoint(), {}, on_arrival=delivered.extend, poll_seconds=0.05,
        ) as watch:
            assert watch.live
            await asyncio.sleep(0.12)
            source.append(_lane_event(
                message_id="m-f1", sequence=2, batch_id="batch-2",
                accepted=_accepted("actually Y", event_id="e1"),
            ))
            await asyncio.sleep(0.2)
            return watch.arrived(), watch.steer_seen

    arrived, steer = _run(scenario())

    assert [item["event_id"] for item in arrived] == ["e1"]
    assert [item["event_id"] for item in delivered] == ["e1"]
    assert arrived[0]["_kdcube_lane_batch_id"] == "batch-2"
    assert steer is False


def test_the_stop_control_is_recognised(monkeypatch):
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)

    async def scenario():
        async with mod.LiveLaneWatch(_entrypoint(), {}, poll_seconds=0.05) as watch:
            source.append(_lane_event(
                message_id="m-stop", sequence=2, batch_id="batch-2",
                accepted=_accepted("", event_id="e-stop", etype="event.user.steer"),
            ))
            await asyncio.sleep(0.2)
            return watch.steer_seen, watch.arrived()

    steer, arrived = _run(scenario())

    assert steer is True
    assert mod.event_is_bare_steer(arrived[0]) is True


def test_the_turns_own_batch_is_not_an_arrival(monkeypatch):
    """What the start-of-turn fold already delivered is this turn's input. A
    watcher that re-announced it would have every lane deliver its own prompt
    back as if the person had just typed it."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    sibling = _lane_event(message_id="m-att", sequence=2,
                          accepted=_accepted("", event_id="e-att", etype="event.user.attachment.file"))
    source = _FakeSource([own, sibling])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)
    state = {"_kdcube_folded_external_events_message_ids": ["m-own", "m-att"]}

    async def scenario():
        async with mod.LiveLaneWatch(_entrypoint(), state, poll_seconds=0.05) as watch:
            await asyncio.sleep(0.2)
            return watch.arrived()

    assert _run(scenario()) == []


def test_an_event_is_delivered_once(monkeypatch):
    """The lane is re-read every poll; a second read must not re-announce."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)
    delivered: list = []

    async def scenario():
        async with mod.LiveLaneWatch(
            _entrypoint(), {}, on_arrival=delivered.extend, poll_seconds=0.05,
        ) as watch:
            source.append(_lane_event(
                message_id="m-f1", sequence=2, batch_id="b2",
                accepted=_accepted("once", event_id="e1"),
            ))
            await asyncio.sleep(0.35)   # several polls over the same lane
            return watch.arrived()

    arrived = _run(scenario())

    assert len(arrived) == 1 and len(delivered) == 1


def test_what_another_turn_consumed_is_not_an_arrival(monkeypatch):
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)

    async def scenario():
        async with mod.LiveLaneWatch(_entrypoint(), {}, poll_seconds=0.05) as watch:
            source.append(_lane_event(
                message_id="m-old", sequence=2, accepted=_accepted("answered", event_id="e-old"),
                consumed_at=999.0,
            ))
            await asyncio.sleep(0.2)
            return watch.arrived()

    assert _run(scenario()) == []


def test_no_wakeup_means_not_live_rather_than_quiet(monkeypatch):
    """A direct invocation has no lane to anchor on. "Nothing arrived" and
    "nobody was listening" must not look the same to a caller deciding whether
    a stop could even have been honoured."""
    entrypoint = SimpleNamespace(redis=object(), comm_context=SimpleNamespace(bundle_call_context={}))

    async def scenario():
        async with mod.LiveLaneWatch(entrypoint, {}, poll_seconds=0.05) as watch:
            await asyncio.sleep(0.1)
            return watch.live, watch.arrived()

    live, arrived = _run(scenario())
    assert live is False and arrived == []


def test_a_failing_delivery_does_not_break_the_run(monkeypatch):
    """Events stay pending and the handoff folds them into the next turn —
    which is exactly the behaviour without a watcher at all."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)

    def boom(_events):
        raise RuntimeError("delivery is broken")

    async def scenario():
        async with mod.LiveLaneWatch(
            _entrypoint(), {}, on_arrival=boom, poll_seconds=0.05,
        ) as watch:
            source.append(_lane_event(
                message_id="m-f1", sequence=2, batch_id="b2",
                accepted=_accepted("still seen", event_id="e1"),
            ))
            await asyncio.sleep(0.2)
            return watch.arrived()

    arrived = _run(scenario())
    assert [item["event_id"] for item in arrived] == ["e1"]


# ── the interrupt half ───────────────────────────────────────────────────────


def test_a_steer_cancels_the_run(monkeypatch):
    """The behaviour the whole design exists for: a run that is visibly wrong
    ends when the person says so, instead of when its timeout says so."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)
    finished = {"ran_to_completion": False}

    async def long_run():
        await asyncio.sleep(5)                      # the wrong, long run
        finished["ran_to_completion"] = True
        return {"answer": "eventually"}

    async def scenario():
        async def stop_soon():
            await asyncio.sleep(0.12)
            source.append(_lane_event(
                message_id="m-stop", sequence=2, batch_id="b2",
                accepted=_accepted("", event_id="e-stop", etype="event.user.steer"),
            ))

        stopper = asyncio.create_task(stop_soon())
        outcome = await mod.run_until_stopped(
            _entrypoint(), {}, long_run, poll_seconds=0.05,
        )
        await stopper
        return outcome

    outcome = _run(scenario())

    assert outcome.stopped is True
    assert outcome.result is None
    assert finished["ran_to_completion"] is False   # it really did not finish
    # And the steer is among what was seen, so a lane can say what stopped it.
    assert any(mod.event_is_bare_steer(body) for body in outcome.arrived)


def test_a_run_nobody_stops_returns_its_answer(monkeypatch):
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: _FakeSource([own]))

    async def quick_run():
        await asyncio.sleep(0.05)
        return {"answer": "done"}

    outcome = _run(mod.run_until_stopped(_entrypoint(), {}, quick_run, poll_seconds=0.05))

    assert outcome.stopped is False
    assert outcome.result == {"answer": "done"}


def test_a_followup_does_not_stop_the_run(monkeypatch):
    """Only a steer stops. A follow-up is "and this too", and cancelling for it
    would throw away the work the person is adding to."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)

    async def run():
        source.append(_lane_event(
            message_id="m-f1", sequence=2, batch_id="b2",
            accepted=_accepted("and this too", event_id="e1"),
        ))
        await asyncio.sleep(0.2)
        return {"answer": "finished anyway"}

    outcome = _run(mod.run_until_stopped(_entrypoint(), {}, run, poll_seconds=0.05))

    assert outcome.stopped is False
    assert outcome.result == {"answer": "finished anyway"}
    assert [item["event_id"] for item in outcome.arrived] == ["e1"]


def test_the_stopped_events_are_left_pending_for_the_next_turn(monkeypatch):
    """Fold-then-interrupt, satisfied by construction: the watcher never
    consumes, so everything it saw is still pending when the run is cancelled
    and the handoff folds it into the next turn. Nothing is written that a
    crash could lose, because nothing is written."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)
    stop_event = _lane_event(
        message_id="m-stop", sequence=2, batch_id="b2",
        accepted=_accepted("use the other channel", event_id="e-stop",
                           etype="event.user.steer"),
    )

    async def run():
        source.append(stop_event)
        await asyncio.sleep(5)
        return {"answer": "never"}

    outcome = _run(mod.run_until_stopped(_entrypoint(), {}, run, poll_seconds=0.05))

    assert outcome.stopped is True
    assert stop_event.consumed_at is None            # untouched on the lane
    assert getattr(stop_event, "promoted_at", None) is None


# ── the reservation has to last as long as the turn ──────────────────────────


class _FakeOrchestrator:
    marks: list = []

    def __init__(self, source):
        self._source = source

    @staticmethod
    def for_source(source):
        return _FakeOrchestrator(source)

    async def mark_consumer_active(self, *, turn_id=""):
        _FakeOrchestrator.marks.append(turn_id)
        return None


def _install_orchestrator(monkeypatch):
    import kdcube_ai_app.apps.chat.sdk.events.event_bus.orchestrator as orch

    _FakeOrchestrator.marks = []
    monkeypatch.setattr(
        orch.ConversationEventBusOrchestrator, "for_source",
        staticmethod(_FakeOrchestrator.for_source),
    )


def test_the_watch_keeps_the_lane_reservation_alive(monkeypatch):
    """The reason this matters, in one sentence: a run-to-completion turn never
    touched the lane, so its consumer stayed `scheduled` — fresh for 30 seconds
    — while the turn itself can run for fifteen minutes. Past that half-minute a
    message typed by the person was ADMITTED as a new turn instead of waiting
    for the handoff, which is what "one turn for the pending lane" and the stop
    boundary both assume.
    """
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: _FakeSource([own]))
    _install_orchestrator(monkeypatch)

    async def scenario():
        async with mod.LiveLaneWatch(
            _entrypoint(), {"turn_id": "turn-7"}, poll_seconds=0.05,
        ):
            await asyncio.sleep(0.3)

    _run(scenario())

    assert len(_FakeOrchestrator.marks) >= 3      # heartbeat, repeatedly
    assert set(_FakeOrchestrator.marks) == {"turn-7"}


def test_the_heartbeat_can_be_turned_off(monkeypatch):
    """A caller that owns the lane state itself (ReAct does) must not have a
    second writer marking it."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: _FakeSource([own]))
    _install_orchestrator(monkeypatch)

    async def scenario():
        async with mod.LiveLaneWatch(
            _entrypoint(), {"turn_id": "turn-7"}, poll_seconds=0.05, heartbeat=False,
        ):
            await asyncio.sleep(0.2)

    _run(scenario())

    assert _FakeOrchestrator.marks == []


def test_a_failing_heartbeat_does_not_break_the_watch(monkeypatch):
    """A lane that cannot be marked goes stale, which is the self-healing
    behaviour that existed before this — not a reason to stop watching."""
    own = _lane_event(message_id="m-own", sequence=1, accepted=_accepted("do X", event_id="e0"))
    source = _FakeSource([own])
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: source)

    import kdcube_ai_app.apps.chat.sdk.events.event_bus.orchestrator as orch

    def _boom(_source):
        raise RuntimeError("lane state unavailable")

    monkeypatch.setattr(orch.ConversationEventBusOrchestrator, "for_source", staticmethod(_boom))

    async def scenario():
        async with mod.LiveLaneWatch(_entrypoint(), {"turn_id": "t"}, poll_seconds=0.05) as watch:
            source.append(_lane_event(
                message_id="m-f1", sequence=2, batch_id="b2",
                accepted=_accepted("still seen", event_id="e1"),
            ))
            await asyncio.sleep(0.2)
            return watch.arrived()

    assert [item["event_id"] for item in _run(scenario())] == ["e1"]
