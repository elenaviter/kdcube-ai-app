# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── external_events.py ── deliver the WHOLE ingress batch to this turn ──
#
# A browser message arrives at ingress as one BATCH of external events —
# context events, the user prompt, and one `event.user.attachment.file` event
# per hosted file — sharing a `batch_id`. The platform dispatches the turn as
# a conversation event-lane *wakeup* that names ONE of those events (the
# prompt), and the wakeup's rehydrated `request.external_events` carries only
# that event. The ReAct workflow never notices: it opens the lane and folds
# every pending event itself. A run-to-completion foreign-runtime turn reads
# only `state["external_events"]` — so without this seam the turn never sees
# the attachment events (the model answers "whats here" blind to the attached
# image).
#
# This module is the foreign-runtime equivalent of ReAct's lane fold: read the
# lane and hand back EVERY event still pending at the start of this turn, in
# lane order, as accepted-event dicts. READ-ONLY on the lane: no consumption
# marks, no reservation changes — lane bookkeeping stays with the shared
# finalize.
#
# It folded only the wakeup event's own BATCH until 2026-08-17, and the rest of
# the pending lane was left to promote one turn each. Two follow-ups typed
# while a turn was running therefore produced two more turns, answered in
# isolation: the agent replied to the first without knowing the second existed,
# and a second message that CORRECTED the first was read only after the work it
# corrected had been paid for. ReAct never had this problem — it folds the
# pending lane into its next iteration — so the foreign-runtime seam now folds
# the same way it does.

from __future__ import annotations

import logging
from typing import Any, Dict, List

LOGGER = logging.getLogger("kdcube.foreign_runtime.external_events")

#: How much of the lane one fold will read. A cap that truncates silently would
#: drop a person's message and read as if they never sent it, so a truncated
#: scan is logged as one.
FOLD_SCAN_LIMIT = 100
#
# Fail-open everywhere: any trouble (no wakeup context, no redis, lane read
# fails) leaves the dispatched events untouched.

#: Stamped onto every folded event: the submission it belongs to, and its place
#: in lane order. A turn folds several submissions now, and a frame that cannot
#: tell them apart attributes one message's files to another.
LANE_BATCH_ID_KEY = "_kdcube_lane_batch_id"
LANE_SEQUENCE_KEY = "_kdcube_lane_sequence"
#: WHEN the person sent it. A folded turn records several messages at once, and
#: stamping them all with the moment the turn ended puts them in the timeline
#: where the answer is rather than where they were typed. The lane knows the
#: real time; nothing downstream could reconstruct it.
LANE_TS_KEY = "_kdcube_lane_ts"

FOLDED_EXTERNAL_EVENTS_BATCH_ID_STATE_KEY = "_kdcube_folded_external_events_batch_id"
FOLDED_EXTERNAL_EVENTS_MESSAGE_IDS_STATE_KEY = "_kdcube_folded_external_events_message_ids"


def folded_external_events_message_ids(state: Any) -> List[str]:
    if not isinstance(state, dict):
        return []
    raw = state.get(FOLDED_EXTERNAL_EVENTS_MESSAGE_IDS_STATE_KEY)
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _lane_wakeup(comm_context: Any) -> Any:
    """The `ExternalEventLaneWakeup` this turn was dispatched from, or None
    (a direct/test invocation has no lane to fold)."""
    bundle_ctx = getattr(comm_context, "bundle_call_context", None) or {}
    wakeup_raw = bundle_ctx.get("event_lane_wakeup")
    if not isinstance(wakeup_raw, dict):
        return None
    try:
        from kdcube_ai_app.apps.chat.sdk.protocol import ExternalEventLaneWakeup

        return ExternalEventLaneWakeup.model_validate(wakeup_raw)
    except Exception:
        return None


def _lane_source(redis: Any, wakeup: Any) -> Any:
    """The conversation event-lane source the wakeup was published on (the
    same lane ingress wrote — lanes are partitioned by agent_id)."""
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


def _lane_timestamp(lane_event: Any, body: Dict[str, Any]) -> str:
    """When this event landed, as the lane recorded it.

    Prefers the accepted event's own timestamp, falls back to the lane
    occurrence's `created_at` (epoch seconds are rendered as UTC ISO, which is
    what every block in the turn log carries). Empty when neither is readable —
    the caller then keeps the behaviour it had before, stamping the record
    moment."""
    for candidate in (body.get("timestamp"), getattr(lane_event, "created_at", None)):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, (int, float)) and candidate > 0:
            import datetime as _dt

            return (
                _dt.datetime.fromtimestamp(float(candidate), _dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
    return ""


def _accepted_body(lane_event: Any) -> Dict[str, Any]:
    """The ingress-accepted event dict a lane occurrence carries (the payload
    envelope's `event`, with `hosted_uri` etc. merged in) — the exact item
    shape `state["external_events"]` holds."""
    payload = getattr(lane_event, "payload", None)
    accepted = payload.get("event") if isinstance(payload, dict) else None
    return dict(accepted) if isinstance(accepted, dict) and accepted else {}


async def fold_turn_external_events(entrypoint: Any, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Everything pending on the lane for this turn, as accepted-event dicts in
    lane order: the wakeup event, its batch siblings (attachments, context
    refs), and any message that queued while the previous turn was running.

    Returns the state's own dispatched events untouched when there is nothing
    more to fold, or on any trouble at all."""
    events = list(state.get("external_events") or [])
    try:
        redis = getattr(entrypoint, "redis", None)
        comm_context = getattr(entrypoint, "comm_context", None)
        if redis is None or comm_context is None:
            return events
        wakeup = _lane_wakeup(comm_context)
        if wakeup is None:
            return events
        event_id = str(getattr(wakeup.event_lane, "event_id", "") or "").strip()
        if not event_id:
            return events
        source = _lane_source(redis, wakeup)
        own = await source.get_event(event_id)
        if own is None:
            return events
        batch_id = str(getattr(own, "batch_id", "") or "").strip()
        lane_events = await source.read_since(0, limit=FOLD_SCAN_LIMIT)
        if lane_events and len(lane_events) >= FOLD_SCAN_LIMIT:
            LOGGER.warning(
                "[foreign-runtime] turn fold read the scan cap (%d events); anything "
                "beyond it stays pending for the next turn rather than folding here",
                FOLD_SCAN_LIMIT,
            )
        own_id = str(getattr(own, "message_id", "") or "")
        # PENDING, not "same batch". The turn's own event plus everything the
        # lane still owes an answer for — a queued follow-up belongs in THIS
        # turn, not in a turn of its own that cannot see the ones after it.
        # Consumed / promoted / failed occurrences are somebody else's history.
        pending = []
        for item in lane_events or []:
            if str(getattr(item, "message_id", "") or "") == own_id:
                pending.append(item)
                continue
            if getattr(item, "consumed_at", None) is not None:
                continue
            if getattr(item, "promoted_at", None) is not None:
                continue
            if getattr(item, "failed_at", None) is not None:
                continue
            pending.append(item)
        if len(pending) <= 1:
            return events
        pending.sort(key=lambda item: int(getattr(item, "sequence", 0) or 0))
        bodies = []
        for item in pending:
            body = _accepted_body(item)
            if not body:
                continue
            # WHICH SUBMISSION THIS CAME FROM. A batch is one thing a person
            # sent — their text plus the files and refs that rode with it — and
            # a turn can now fold several. Without this stamp a frame can only
            # list the cargo flat, so an attachment sent with the third message
            # reads as if it arrived with the first. Namespaced and additive:
            # the accepted-event shape every existing consumer reads is
            # untouched.
            body[LANE_BATCH_ID_KEY] = str(getattr(item, "batch_id", "") or "")
            body[LANE_SEQUENCE_KEY] = int(getattr(item, "sequence", 0) or 0)
            stamp = _lane_timestamp(item, body)
            if stamp:
                body[LANE_TS_KEY] = stamp
            bodies.append(body)
        if len(bodies) <= 1:
            return events
        try:
            message_ids = [
                str(getattr(item, "message_id", "") or "").strip()
                for item in pending
                if str(getattr(item, "message_id", "") or "").strip()
            ]
            if isinstance(state, dict) and message_ids:
                # The finalizer consumes exactly these ids, so what folded is
                # what gets terminalized — and anything that lands after this
                # read stays pending for the next handoff.
                state[FOLDED_EXTERNAL_EVENTS_BATCH_ID_STATE_KEY] = batch_id
                state[FOLDED_EXTERNAL_EVENTS_MESSAGE_IDS_STATE_KEY] = message_ids
        except Exception:
            pass
        batches = {
            str(getattr(item, "batch_id", "") or "").strip()
            for item in pending
            if str(getattr(item, "batch_id", "") or "").strip()
        }
        LOGGER.info(
            "[foreign-runtime] turn fold: %d pending event(s) across %d batch(es) "
            "(dispatch carried %d) wakeup_batch_id=%s",
            len(bodies), len(batches), len(events), batch_id,
        )
        return bodies
    except Exception:
        LOGGER.warning(
            "[foreign-runtime] turn batch fold failed; using the dispatched events",
            exc_info=True,
        )
        return events
