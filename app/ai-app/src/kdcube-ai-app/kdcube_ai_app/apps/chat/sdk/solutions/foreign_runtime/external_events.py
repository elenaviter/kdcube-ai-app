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
# This module is the foreign-runtime equivalent of ReAct's lane fold, scoped
# to exactly this turn's batch: read the lane, take the wakeup event's batch
# siblings (skipping any a previous turn already consumed — a re-woken queued
# followup promotes alone), and hand back the accepted-event dicts in lane
# order. READ-ONLY on the lane: no consumption marks, no reservation changes —
# lane bookkeeping stays with the shared finalize exactly as it behaves today.
#
# Fail-open everywhere: any trouble (no wakeup context, no redis, lane read
# fails) leaves the dispatched events untouched.

from __future__ import annotations

import logging
from typing import Any, Dict, List

LOGGER = logging.getLogger("kdcube.foreign_runtime.external_events")


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


def _accepted_body(lane_event: Any) -> Dict[str, Any]:
    """The ingress-accepted event dict a lane occurrence carries (the payload
    envelope's `event`, with `hosted_uri` etc. merged in) — the exact item
    shape `state["external_events"]` holds."""
    payload = getattr(lane_event, "payload", None)
    accepted = payload.get("event") if isinstance(payload, dict) else None
    return dict(accepted) if isinstance(accepted, dict) and accepted else {}


async def fold_turn_external_events(entrypoint: Any, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """This turn's full ingress batch as accepted-event dicts, lane-ordered.

    Returns the enriched event list when the wakeup event has batch siblings
    to fold (the attachment events beside the prompt); otherwise the state's
    own dispatched events, untouched."""
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
        if not batch_id:
            return events
        lane_events = await source.read_since(0)
        own_id = str(getattr(own, "message_id", "") or "")
        siblings = [
            item
            for item in lane_events or []
            if str(getattr(item, "batch_id", "") or "") == batch_id
            and (
                str(getattr(item, "message_id", "") or "") == own_id
                or getattr(item, "consumed_at", None) is None
            )
        ]
        if len(siblings) <= 1:
            return events
        siblings.sort(key=lambda item: int(getattr(item, "sequence", 0) or 0))
        bodies = [body for body in (_accepted_body(item) for item in siblings) if body]
        if len(bodies) <= 1:
            return events
        LOGGER.info(
            "[foreign-runtime] turn batch fold: %d event(s) in this turn's batch (dispatch carried %d)",
            len(bodies), len(events),
        )
        return bodies
    except Exception:
        LOGGER.warning(
            "[foreign-runtime] turn batch fold failed; using the dispatched events",
            exc_info=True,
        )
        return events
