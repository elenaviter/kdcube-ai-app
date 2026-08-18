# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── turn_batch.py ── deliver the WHOLE pending lane to this turn ──
#
# Lifted into the SDK's shared foreign-runtime seam
# (`kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.external_events`);
# this module delegates to it and re-exports the same names.
#
# Why it exists at all: a browser message arrives at ingress as one BATCH of
# external events (context events, the user prompt, one attachment event per
# hosted file) sharing a `batch_id`, while the lane-wakeup dispatch hands a
# run-to-completion turn only its single wakeup event (the prompt). The fold
# reads the lane once and delivers that occurrence, its batch siblings, and all
# other still-pending events in sequence order — READ-ONLY on the lane,
# fail-open everywhere (any trouble leaves the dispatched events untouched).
#
# `_lane_source` (with `_lane_wakeup` / `_accepted_body`) stays a MODULE-LEVEL
# name here on purpose: it is this bundle's offline-test injection point (the
# tests rebind it to a fake lane source). The fold below threads a rebound
# `_lane_source` through the seam for the duration of the call; when nothing is
# rebound (production), it is a pure delegation.

from __future__ import annotations

from typing import Any, Dict, List

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import (
    external_events as _seam,
)

_lane_wakeup = _seam._lane_wakeup
_lane_source = _seam._lane_source
_accepted_body = _seam._accepted_body


async def fold_turn_external_events(entrypoint: Any, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """This turn's whole pending lane as accepted-event dicts, lane-ordered.

    Includes the wake occurrence, its attachment/context siblings, and messages
    queued while the previous turn ran. The shared foreign-runtime seam keeps
    the read non-consuming and records exact ids for finalization."""
    override = _lane_source  # module global: the offline tests' injection point
    if override is _seam._lane_source:
        return await _seam.fold_turn_external_events(entrypoint, state)
    original = _seam._lane_source
    _seam._lane_source = override
    try:
        return await _seam.fold_turn_external_events(entrypoint, state)
    finally:
        _seam._lane_source = original
