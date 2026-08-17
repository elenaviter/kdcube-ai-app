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
  * Otherwise (a run-to-completion turn left it reserved) → release the consumer,
    mark the turn's own event consumed for exactly-once, and re-wake any reactive
    event that landed during the turn (a queued followup) so it promotes to the
    next turn.

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


async def _rewake_pending_reactive_events(
    *,
    source: Any,
    state: Any,
    wake_publisher: Optional[EventLaneWakePublisher],
    own_ts: str,
    turn_id: str,
) -> None:
    """Re-publish a lane wakeup for every unconsumed reactive event that landed
    AFTER the turn's own event (a followup queued mid-turn). Exactly-once holds:
    consumed/promoted/failed events, the turn's own event, and anything at/earlier
    than it are skipped, and a duplicate wake is dropped by the lane guards.

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
        try:
            payload = event.task_payload_model()
        except Exception:
            continue
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
                "[reactive-lane] run-to-completion handoff re-woke reactive event "
                "conversation=%s turn_id=%s event_id=%s event_ts=%s",
                getattr(source, "conversation_id", None),
                turn_id,
                getattr(event, "message_id", ""),
                event_timestamp(event),
            )
        else:
            logger.warning(
                "[reactive-lane] run-to-completion handoff re-wake not queued "
                "conversation=%s turn_id=%s event_id=%s reason=%s",
                getattr(source, "conversation_id", None),
                turn_id,
                getattr(event, "message_id", ""),
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

        # A single visible UI send arrives as an ingress batch: prompt plus
        # sibling attachments/context events. The foreign-runtime batch fold reads
        # those siblings into this turn, so the finalizer must terminalize those
        # exact lane occurrences before liveness re-wake scans the lane. This is
        # deliberately exact-id based, not a sequence-range consume: an
        # accepted occurrence from a different batch can interleave between
        # same-batch siblings and must remain pending for the next turn.
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
                "[reactive-lane] run-to-completion consumed folded event batch "
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
