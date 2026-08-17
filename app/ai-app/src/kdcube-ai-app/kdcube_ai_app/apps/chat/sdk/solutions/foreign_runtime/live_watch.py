# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── live_watch.py ── watching the lane WHILE a foreign-loop turn runs ──
#
# A run-to-completion turn folds the lane once, at its start, and then goes
# deaf: a message typed while it works — including the stop control, which is a
# steer with no text — reaches nobody until the turn ends. On the press lane a
# turn is bounded by `timeout_seconds` (900 in the deployed descriptor), so a
# run that is visibly wrong cannot be stopped for up to fifteen minutes.
#
# ReAct does not have this problem: it owns its loop and checks between
# iterations. LangGraph and Claude Code do NOT own theirs — the graph and the
# CLI's agentic loop each manage their own iteration — so the answer is not to
# seize the loop but to WATCH, and let the lane tell each runtime, in its own
# terms, that something arrived.
#
# This module is the watching half, and it is lane-neutral on purpose:
#
#   * it reads the same conversation lane the start-of-turn fold reads
#   * it only sees events for THIS turn's conversation that landed after the
#     turn's own event, and never one a previous turn consumed
#   * it dedups by lane message id, so a re-read cannot deliver twice
#   * it classifies steer vs followup the way ReAct's live-event lane does
#
# What it deliberately does NOT do is decide anything. It has no opinion about
# stopping, folding or interrupting: a lane provides those, because "interrupt"
# means cancelling a task in one runtime and denying a tool call in another.
# The watcher's whole contract is "here is what arrived, in order, once".

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from kdcube_ai_app.apps.chat.sdk.protocol import external_event_text
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.external_events import (
    LANE_BATCH_ID_KEY,
    LANE_SEQUENCE_KEY,
    LANE_TS_KEY,
    _accepted_body,
    _lane_source,
    _lane_wakeup,
)

LOGGER = logging.getLogger("kdcube.foreign_runtime.live_watch")

#: The lane occurrence an arrival came from, carried on the arrival itself.
LANE_MESSAGE_ID_KEY = "_kdcube_lane_message_id"

#: How often the lane is re-read while a turn runs. A turn lives for minutes and
#: a person waiting on a stop notices a second; polling is chosen over a
#: subscription because the lane is already read this way everywhere else and a
#: poll cannot leave a subscription dangling when a turn dies mid-flight.
POLL_SECONDS = 1.0

STEER_EVENT_TYPE = "event.user.steer"


def event_is_steer(body: Dict[str, Any]) -> bool:
    return str((body or {}).get("type") or "").strip() == STEER_EVENT_TYPE


def event_is_bare_steer(body: Dict[str, Any]) -> bool:
    """The stop control: a steer with nothing else said."""
    return event_is_steer(body) and not external_event_text(body)


class LiveLaneWatch:
    """What arrived on the lane since this turn started.

    Poll-driven and read-only — it never consumes, promotes or reserves
    anything, so a turn that dies mid-flight leaves the lane exactly as the
    start-of-turn fold left it and the normal handoff still runs.

    Use as an async context manager around the work being watched::

        async with LiveLaneWatch(entrypoint, state, on_arrival=deliver) as watch:
            await run_the_agent()
            ...
        if watch.steer_seen:
            ...
    """

    def __init__(
        self,
        entrypoint: Any,
        state: Dict[str, Any],
        *,
        on_arrival: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        poll_seconds: float = POLL_SECONDS,
        heartbeat: bool = True,
    ) -> None:
        self._entrypoint = entrypoint
        self._state = state if isinstance(state, dict) else {}
        self._heartbeat_enabled = bool(heartbeat)
        self._turn_id = str((state or {}).get("turn_id") or "") if isinstance(state, dict) else ""
        self._on_arrival = on_arrival
        self._poll_seconds = max(0.1, float(poll_seconds or POLL_SECONDS))
        self._task: Optional[asyncio.Task] = None
        self._seen: set[str] = set()
        self._arrived: List[Dict[str, Any]] = []
        self._steer_seen = False
        self._live = False
        self._source: Any = None

    # -- what the caller reads ------------------------------------------------

    @property
    def live(self) -> bool:
        """Whether the lane could actually be watched. A direct invocation has
        no wakeup to anchor on, and a lane that cannot be read is reported as
        not live rather than as quiet — "nothing arrived" and "nobody was
        listening" must not look the same."""
        return self._live

    @property
    def steer_seen(self) -> bool:
        return self._steer_seen

    def arrived(self) -> List[Dict[str, Any]]:
        """Everything seen so far, in lane order. Stamped exactly like the
        start-of-turn fold, so a caller handles one shape."""
        return list(self._arrived)
    async def consume_delivered(self, message_ids: Sequence[str]) -> int:
        """Close the lane events a runtime actually delivered to its model.

        The watcher reads and does not write — except here, and only for what a
        lane REPORTS as delivered. An event the model read mid-run was answered
        by this turn, so leaving it pending would wake another turn for a
        message that has already been dealt with. Everything the lane cannot
        vouch for stays pending, which is the safe direction: at worst it is
        folded into the next turn and read twice by a person, rather than
        silently dropped.
        """
        ids = [str(item or "").strip() for item in (message_ids or []) if str(item or "").strip()]
        if not ids or self._source is None:
            return 0
        closed = 0
        for message_id in ids:
            try:
                event = await self._source.mark_consumed_event(
                    message_id=message_id, turn_id=self._turn_id,
                )
                if event is not None:
                    closed += 1
            except Exception:
                LOGGER.debug("[live-watch] could not close %s", message_id, exc_info=True)
        LOGGER.info(
            "[live-watch] closed %d/%d delivered event(s) as answered by this turn",
            closed, len(ids),
        )
        return closed

    # -- lifecycle ------------------------------------------------------------

    async def __aenter__(self) -> "LiveLaneWatch":
        try:
            source, own = await self._anchor()
        except Exception:
            LOGGER.debug("[live-watch] could not open the lane; not watching", exc_info=True)
            return self
        if source is None or own is None:
            return self
        self._live = True
        self._source = source
        self._task = asyncio.create_task(self._poll(source, own))
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - best effort
            pass

    # -- internals ------------------------------------------------------------

    async def _anchor(self):
        redis = getattr(self._entrypoint, "redis", None)
        comm_context = getattr(self._entrypoint, "comm_context", None)
        if redis is None or comm_context is None:
            return None, None
        wakeup = _lane_wakeup(comm_context)
        if wakeup is None:
            return None, None
        event_id = str(getattr(wakeup.event_lane, "event_id", "") or "").strip()
        if not event_id:
            return None, None
        source = _lane_source(redis, wakeup)
        own = await source.get_event(event_id)
        if own is None:
            return None, None
        # Everything the fold already delivered is this turn's own input, not an
        # arrival: seeding them keeps a poll from re-announcing the batch the
        # turn started with.
        from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.external_events import (
            folded_external_events_message_ids,
        )

        for message_id in folded_external_events_message_ids(self._state):
            self._seen.add(message_id)
        self._seen.add(str(getattr(own, "message_id", "") or ""))
        return source, own

    async def _heartbeat(self, source: Any) -> None:
        """Tell the lane this turn is still consuming it.

        A run-to-completion turn never touched the lane, so its consumer stayed
        `scheduled` — and `schedule_consumer_from_wake` treats `scheduled` as
        fresh only within 30 seconds. For a turn bounded at fifteen minutes that
        means the reservation is stale for most of its life, and a message typed
        after the first half-minute is ADMITTED as a new turn instead of waiting
        for the handoff. Everything downstream — one turn for the pending lane,
        the stop as a boundary — assumes the events wait, so the wait has to be
        real.

        ReAct keeps this alive from `browser.py`; a foreign lane had no
        equivalent until this poll, which is already ticking once a second.
        Best-effort: a lane that cannot be marked is left to go stale, which is
        the self-healing behaviour that existed before.
        """
        try:
            from kdcube_ai_app.apps.chat.sdk.events.event_bus.orchestrator import (
                ConversationEventBusOrchestrator,
            )

            orchestrator = ConversationEventBusOrchestrator.for_source(source)
            await orchestrator.mark_consumer_active(turn_id=self._turn_id)
        except Exception:
            LOGGER.debug("[live-watch] consumer heartbeat failed", exc_info=True)

    async def _poll(self, source: Any, own: Any) -> None:
        own_sequence = int(getattr(own, "sequence", 0) or 0)
        while True:
            if self._heartbeat_enabled:
                await self._heartbeat(source)
            try:
                await self._read_once(source, own_sequence)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.debug("[live-watch] lane read failed; will retry", exc_info=True)
            await asyncio.sleep(self._poll_seconds)

    async def _read_once(self, source: Any, own_sequence: int) -> None:
        events = await source.read_since(0, limit=100)
        fresh: List[Dict[str, Any]] = []
        for item in sorted(
            events or [], key=lambda e: int(getattr(e, "sequence", 0) or 0)
        ):
            message_id = str(getattr(item, "message_id", "") or "")
            if not message_id or message_id in self._seen:
                continue
            if int(getattr(item, "sequence", 0) or 0) <= own_sequence:
                continue
            if getattr(item, "consumed_at", None) is not None:
                continue
            if getattr(item, "promoted_at", None) is not None:
                continue
            if getattr(item, "failed_at", None) is not None:
                continue
            body = _accepted_body(item)
            if not body:
                continue
            body[LANE_BATCH_ID_KEY] = str(getattr(item, "batch_id", "") or "")
            body[LANE_SEQUENCE_KEY] = int(getattr(item, "sequence", 0) or 0)
            # WHICH lane event this is. A lane that delivers an arrival to its
            # runtime has to be able to say afterwards which ones actually
            # landed, and only the message id survives that round trip.
            body[LANE_MESSAGE_ID_KEY] = message_id
            from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.external_events import (
                _lane_timestamp,
            )

            stamp = _lane_timestamp(item, body)
            if stamp:
                body[LANE_TS_KEY] = stamp
            self._seen.add(message_id)
            fresh.append(body)
        if not fresh:
            return
        self._arrived.extend(fresh)
        if any(event_is_steer(body) for body in fresh):
            self._steer_seen = True
        LOGGER.info(
            "[live-watch] %d event(s) arrived mid-turn (steer=%s)",
            len(fresh), self._steer_seen,
        )
        if self._on_arrival is None:
            return
        try:
            result = self._on_arrival(fresh)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # A lane that cannot be delivered to is not a reason to break the
            # run being watched: the events stay pending and the handoff folds
            # them into the next turn, which is the behaviour without a watcher
            # at all.
            LOGGER.warning("[live-watch] delivery failed; events stay pending", exc_info=True)


class StoppedRun:
    """The outcome of a watched run: what it returned, and whether a person
    stopped it before it got there."""

    __slots__ = ("result", "stopped", "watch")

    def __init__(self, *, result: Any, stopped: bool, watch: "LiveLaneWatch") -> None:
        self.result = result
        self.stopped = stopped
        self.watch = watch

    @property
    def arrived(self) -> List[Dict[str, Any]]:
        return self.watch.arrived()


#: What a stopped turn answers with. Plain language, in band: the person sees
#: why the stream ended rather than an answer that trails off, and the model's
#: next turn reads the same sentence in its history instead of inferring that it
#: mysteriously fell silent.
STOPPED_ANSWER = (
    "Stopped at your request. Whatever I had finished before that is above; "
    "nothing further was done."
)


async def run_until_stopped(
    entrypoint: Any,
    state: Dict[str, Any],
    runner: Callable[[], Any],
    *,
    on_arrival: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    poll_seconds: float = POLL_SECONDS,
) -> StoppedRun:
    """Run ``runner()`` while watching the lane, and cancel it when a steer lands.

    The interrupt half of fold-then-interrupt, for a runtime whose loop can be
    cancelled at an await point — LangGraph, whose checkpointer holds the last
    completed node, so a cancelled stream loses nothing that was finished.

    The FOLD half needs no work here and that is the point: the watcher is
    read-only, so every event it saw — the follow-ups and the steer itself — is
    still pending on the lane when the run is cancelled. The handoff then folds
    them into the next turn, which is where the person's next answer comes from.
    Nothing is written that a crash could lose, because nothing is written.

    A cancellation that arrives without a steer is somebody else's (the platform
    cancelling the turn) and is re-raised untouched.
    """
    loop_task: Optional[asyncio.Task] = None

    def _arrival(events: List[Dict[str, Any]]) -> None:
        if loop_task is not None and any(event_is_steer(body) for body in events):
            LOGGER.info("[live-watch] steer arrived; cancelling the run")
            loop_task.cancel()
        if on_arrival is not None:
            on_arrival(events)

    async with LiveLaneWatch(
        entrypoint, state, on_arrival=_arrival, poll_seconds=poll_seconds,
    ) as watch:
        loop_task = asyncio.create_task(_as_coroutine(runner))
        try:
            result = await loop_task
        except asyncio.CancelledError:
            if not watch.steer_seen:
                raise
            return StoppedRun(result=None, stopped=True, watch=watch)
        return StoppedRun(result=result, stopped=False, watch=watch)


async def _as_coroutine(runner: Callable[[], Any]) -> Any:
    outcome = runner()
    if asyncio.iscoroutine(outcome):
        return await outcome
    return outcome
