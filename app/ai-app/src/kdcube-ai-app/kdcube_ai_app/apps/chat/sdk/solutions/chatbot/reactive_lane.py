# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Reactive-event lane finalization for the run-to-completion turn path.

Every reactive turn is dispatched to the processor as a conversation
external-event lane *wakeup*, which reserves the lane consumer
(``consumer_status="scheduled"``) before the turn runs. A turn whose
``execute_core`` drives a ``BaseWorkflow`` (the ReAct path) opens the lane
handler and releases that reservation itself (``react/browser.py``:
``close_external_event_handler`` + ``post_save_external_event_handoff``). A
run-to-completion ``execute_core`` is bespoke and never touches the lane, so the
reservation is left dangling — and within the scheduled-consumer TTL the NEXT
turn's wakeup is dropped as ``scheduled_consumer_fresh`` and silently never runs.

This module finalizes the reactive-event lane from the shared reactive-event door
(``BaseEntrypoint.run`` / ``BaseEntrypointWithEconomics.run``, around
``execute_core``). It is a STATE-CONDITIONAL, IDEMPOTENT invariant, NOT an
agent-type branch:

  * If the lane reservation is already released (consumer ``none``) and the turn's
    own event is already accounted for (consumed, or covered by the reactive
    cursor) → no-op. This is exactly the post-ReAct lane state, so a ReAct turn is
    inert here with no ``if react`` check.
  * Otherwise (a run-to-completion turn left it reserved) → account for the wake
    and every exact event id in its read-only pending-lane snapshot, release the
    turn-owned consumer reservation, and wake at most one eligible intent after
    the last steer boundary.

It reuses the existing lane primitives only: the source's ``mark_consumed_up_to``
(the same exactly-once mark ``BaseWorkflow`` uses), the orchestrator's
``mark_consumer_none``, and the lane wake re-publish (the ``post_save`` mechanism).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from kdcube_ai_app.apps.chat.sdk.events.event_bus.orchestrator import (
    ConversationEventBusOrchestrator,
)
from kdcube_ai_app.apps.chat.sdk.events.event_bus.state import (
    event_is_reactive,
    event_timestamp,
    timestamp_lte,
)
from kdcube_ai_app.apps.chat.sdk.events.event_bus.wakeup import (
    EventLaneWakePublisher,
    RedisEventLaneWakeEnqueuer,
)

logger = logging.getLogger(__name__)


def _lane_wakeup_from_comm_context(comm_context: Any):
    """The ``ExternalEventLaneWakeup`` this turn was dispatched from, or ``None``
    when the turn did not enter through a lane wakeup (nothing to finalize)."""
    bundle_ctx = getattr(comm_context, "bundle_call_context", None) or {}
    wakeup_raw = bundle_ctx.get("event_lane_wakeup")
    if not isinstance(wakeup_raw, dict):
        return None
    from kdcube_ai_app.apps.chat.sdk.protocol import ExternalEventLaneWakeup

    try:
        return ExternalEventLaneWakeup.model_validate(wakeup_raw)
    except Exception:
        return None


def _source_for_wakeup(redis: Any, wakeup: Any):
    from kdcube_ai_app.apps.chat.external_events import (
        build_conversation_external_event_source,
    )
    from kdcube_ai_app.apps.chat.sdk.event_identity import (
        DEFAULT_REACT_AGENT_ID,
        normalize_agent_id,
    )

    lane = wakeup.event_lane
    return build_conversation_external_event_source(
        redis=redis,
        tenant=lane.tenant or wakeup.actor.tenant_id,
        project=lane.project or wakeup.actor.project_id,
        conversation_id=lane.conversation_id or wakeup.routing.conversation_id or wakeup.routing.session_id,
        user_id=lane.user_id or wakeup.user.user_id or wakeup.user.fingerprint or "",
        agent_id=normalize_agent_id(lane.agent_id, default=DEFAULT_REACT_AGENT_ID),
    )


def _is_steer(event: Any) -> bool:
    """A steer, with or without text — the stop control either way."""
    payload = getattr(event, "payload", None)
    body = payload.get("event") if isinstance(payload, dict) else None
    if not isinstance(body, dict):
        return False
    return str(body.get("type") or "").strip() == "event.user.steer"


def _is_bare_steer(event: Any) -> bool:
    """A steer carrying no text — the "stop" control with nothing else said.

    It asks a RUNNING turn to wrap up. Reaching this function means the turn
    already ended, so there is nothing left to ask, and waking a turn for it
    means the model is handed an empty message: pressing stop bought a turn
    instead of saving one. A steer WITH text remains pending, but the steer
    itself still does not wake a turn.
    """
    from kdcube_ai_app.apps.chat.sdk.protocol import external_event_text

    payload = getattr(event, "payload", None)
    body = payload.get("event") if isinstance(payload, dict) else None
    if not isinstance(body, dict):
        return False
    if str(body.get("type") or "").strip() != "event.user.steer":
        return False
    return not external_event_text(body)


async def _rewake_pending_reactive_events(
    *,
    source: Any,
    state: Any,
    wake_publisher: Optional[EventLaneWakePublisher],
    own_ts: str,
    turn_id: str,
) -> None:
    """Hand the lane's pending work to ONE next turn.

    Every unconsumed reactive event that landed after the turn's own event (the
    follow-ups queued mid-turn) belongs to the next turn — and to the SAME next
    turn, because the foreign-runtime fold now takes the whole pending lane.
    Waking one per event, which is what this did until 2026-08-17, produced a
    turn per message: the agent answered the first without knowing the second
    existed, and a correction was read only after the work it corrected had
    been paid for. One wake, and the fold absorbs the rest.

    Exactly-once holds as before: consumed/promoted/failed events, the turn's
    own event, and anything at or earlier than it are skipped, and a duplicate
    wake is dropped by the lane guards.

    A followup a ReAct turn FOLDED into itself is left alone: ReAct does not set
    ``consumed_at`` on a folded event (it tracks folding by advancing the lane's
    ``last_processed_reactive_event_timestamp`` cursor), so this skip MUST also
    honor that cursor — the SAME gate ReAct's own post-save handoff uses
    (``browser.py::post_save_external_event_handoff``). Without it, this re-wake
    is narrower than ReAct's and re-runs a folded followup as a second turn
    (a duplicate user message + answer)."""
    if wake_publisher is None:
        return
    try:
        pending = await source.read_since(0, limit=100)
    except Exception:
        logger.debug("reactive lane finalize: read_since failed", exc_info=True)
        return
    candidates = []
    for event in pending or []:
        if getattr(event, "consumed_at", None) is not None:
            continue
        if getattr(event, "promoted_at", None) is not None:
            continue
        if getattr(event, "failed_at", None) is not None:
            continue
        if not event_is_reactive(event):
            continue
        if timestamp_lte(event_timestamp(event), own_ts):
            continue
        # Already folded/processed by the turn (ReAct's cursor is past it) — the
        # turn already handled this event, so re-waking it would double-run it.
        if state is not None and state.event_was_processed(event):
            continue
        candidates.append(event)

    # THE STEER IS A BOUNDARY. Everything at or before it was said while the run
    # it stopped was going, and the stop supersedes it: those events are not
    # thrown away — they stay pending and fold into whatever turn comes next —
    # but they do not START one. Only what arrives AFTER the stop is new intent,
    # and only new intent is worth spending a turn on. Without this, pressing
    # stop still bought a turn: the follow-ups typed just before it woke one
    # immediately, which is the spending the button exists to prevent.
    steer_sequences = [
        int(getattr(event, "sequence", 0) or 0)
        for event in candidates
        if _is_steer(event)
    ]
    boundary = max(steer_sequences) if steer_sequences else None

    live: list = []
    for event in candidates:
        if _is_bare_steer(event):
            # A bare stop is spent the moment the turn it aimed at is over:
            # nothing is left to ask, and waking for it hands the agent an empty
            # message. Terminalized rather than left to linger.
            try:
                await source.mark_consumed_event(
                    message_id=str(getattr(event, "message_id", "") or ""),
                    turn_id=turn_id,
                )
                logger.info(
                    "[reactive-lane] bare steer terminalized at handoff (the turn it "
                    "asked to stop had already ended) conversation=%s turn_id=%s event_id=%s",
                    getattr(source, "conversation_id", None),
                    turn_id,
                    getattr(event, "message_id", ""),
                )
            except Exception:
                logger.debug(
                    "reactive lane finalize: bare-steer terminalize failed", exc_info=True
                )
            continue
        if boundary is not None and int(getattr(event, "sequence", 0) or 0) <= boundary:
            # Said before the stop, so the stop supersedes it as a REASON to
            # run. Left pending on purpose: the next turn folds it in, in order,
            # whenever a person starts one.
            continue
        live.append(event)
    if not live:
        if boundary is not None:
            logger.info(
                "[reactive-lane] stop was the last word on the lane; %d earlier "
                "event(s) stay pending for the next turn conversation=%s turn_id=%s",
                len(candidates), getattr(source, "conversation_id", None), turn_id,
            )
        return

    live.sort(key=lambda item: int(getattr(item, "sequence", 0) or 0))
    event = live[0]
    try:
        payload = event.task_payload_model()
    except Exception:
        return
    result = await wake_publisher.publish_for_event(
        payload=payload,
        event=event,
        tenant=getattr(source, "tenant", None),
        project=getattr(source, "project", None),
        user_id=getattr(source, "user_id", None),
        conversation_id=getattr(source, "conversation_id", None),
        agent_id=getattr(source, "agent_id", None),
        reason="run_to_completion_handoff",
    )
    if getattr(result, "success", False):
        logger.info(
            "[reactive-lane] run-to-completion handoff woke ONE turn for %d pending "
            "event(s) conversation=%s turn_id=%s event_id=%s event_ts=%s",
            len(live),
            getattr(source, "conversation_id", None),
            turn_id,
            getattr(event, "message_id", ""),
            event_timestamp(event),
        )
    else:
        logger.warning(
            "[reactive-lane] run-to-completion handoff re-wake not queued "
            "conversation=%s turn_id=%s event_id=%s pending=%d reason=%s",
            getattr(source, "conversation_id", None),
            turn_id,
            getattr(event, "message_id", ""),
            len(live),
            getattr(result, "reason", ""),
        )


async def finalize_reactive_event_lane(
    *,
    redis: Any,
    comm_context: Any,
    turn_id: str = "",
    consumed_event_ids: Optional[list[str]] = None,
) -> bool:
    """Release the reactive-event lane for a completed run-to-completion turn.

    Returns ``True`` when the lane was finalized (a run-to-completion turn left it
    reserved), ``False`` on a no-op — not a lane-wakeup turn, or the reservation
    was already released and the own event already accounted for (the post-ReAct
    state). Never raises; best-effort by contract.
    """
    log = logger
    if redis is None or comm_context is None:
        return False
    wakeup = _lane_wakeup_from_comm_context(comm_context)
    if wakeup is None:
        return False
    event_id = str(getattr(wakeup.event_lane, "event_id", "") or "").strip()
    if not event_id:
        return False

    turn_id = str(turn_id or getattr(getattr(comm_context, "routing", None), "turn_id", "") or "")

    try:
        source = _source_for_wakeup(redis, wakeup)
        event = await source.get_event(event_id)
        if event is None:
            return False
        orchestrator = ConversationEventBusOrchestrator.for_source(source)
        state = await orchestrator.state()

        # STATE-CONDITIONAL no-op (never an agent-type check): the reservation is
        # already released AND the turn's own event is already accounted for. This
        # is precisely the state a ReAct turn's BaseWorkflow leaves behind, so a
        # ReAct turn is inert here.
        already_released = str(getattr(state, "consumer_status", "") or "") == "none"
        own_accounted = (
            getattr(event, "consumed_at", None) is not None
            or state.event_was_processed(event)
        )
        if already_released and own_accounted:
            return False

        # Run-to-completion left the lane reserved. Mark the own event consumed
        # (exactly-once: a re-delivered wakeup for it is dropped as
        # event_already_consumed) using the same cursor primitive BaseWorkflow
        # uses for the wakeup occurrence itself.
        own_seq = int(getattr(event, "sequence", 0) or 0)
        if own_seq > 0:
            try:
                await source.mark_consumed_up_to(max_sequence=own_seq, turn_id=turn_id)
            except Exception:
                log.debug("reactive lane finalize: mark_consumed_up_to failed", exc_info=True)

        # The foreign-runtime start fold may span several ingress batches: the
        # wake occurrence, attachment/context siblings, and messages queued
        # while the previous turn ran. Finalize exactly what this turn saw.
        # A range consume would also swallow an event that arrived after the
        # snapshot while execute_core was already running.
        folded_ids = []
        seen_folded_ids = set()
        for item in consumed_event_ids or []:
            event_key = str(item or "").strip()
            if not event_key or event_key == event_id or event_key in seen_folded_ids:
                continue
            seen_folded_ids.add(event_key)
            folded_ids.append(event_key)
        exact_consumed = 0
        for folded_event_id in folded_ids:
            try:
                consumed_event = await source.mark_consumed_event(
                    message_id=folded_event_id,
                    turn_id=turn_id,
                )
                if consumed_event is not None:
                    exact_consumed += 1
            except Exception:
                log.debug("reactive lane finalize: mark_consumed_event failed", exc_info=True)
        if folded_ids:
            log.info(
                "[reactive-lane] run-to-completion consumed folded lane snapshot "
                "conversation=%s turn_id=%s own_sequence=%s folded_events=%s exact_consumed=%s",
                getattr(source, "conversation_id", None),
                turn_id,
                own_seq,
                folded_ids,
                exact_consumed,
            )

        # Release the consumer reservation so the next turn's wakeup is not dropped
        # as scheduled_consumer_fresh. Every event already has its atomic ingress
        # wake; the re-wake below is a liveness duplicate and must observe the
        # released reservation, not race ahead of it.
        await orchestrator.mark_consumer_none(turn_id=turn_id)

        wake_publisher = EventLaneWakePublisher(
            RedisEventLaneWakeEnqueuer(
                redis=redis,
                tenant=str(getattr(wakeup.actor, "tenant_id", "") or getattr(source, "tenant", "") or ""),
                project=str(getattr(wakeup.actor, "project_id", "") or getattr(source, "project", "") or ""),
            )
        )
        await _rewake_pending_reactive_events(
            source=source,
            state=state,
            wake_publisher=wake_publisher,
            own_ts=event_timestamp(event),
            turn_id=turn_id,
        )
        return True
    except Exception:
        log.debug("reactive lane finalize failed (best-effort)", exc_info=True)
        return False
