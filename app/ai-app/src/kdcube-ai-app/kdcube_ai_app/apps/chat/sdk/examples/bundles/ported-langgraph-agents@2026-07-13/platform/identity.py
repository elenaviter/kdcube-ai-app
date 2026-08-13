# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── identity.py ── the shared multi-user + multi-agent isolation gate ──
#
# Lifted into the SDK's shared foreign-runtime seam
# (`kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.identity`); this module
# re-exports it unchanged so the bundle keeps one import home for the gate.
#
# In this app the gate maps the PLATFORM identity onto each agent's per-user +
# per-conversation keys AND folds the ACTIVE agent_id into them, separating the
# two hosted agents (`lg-solution`, `lg-react`) even though they share a store.
# (Storage rows are also tagged with the scope columns tenant/project/bundle_id/
# agent_id — see pg_target.py — so the store filters `WHERE agent_id = …`; the
# fold is the belt-and-suspenders key-level guarantee.) The Telegram webhook
# (the 2nd ingress) needs no special case: the SDK resolves a Telegram sender to
# the platform ``user`` ``telegram_<id>`` before the turn, so it folds identically.

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.identity import (
    TurnIdentity,
    normalize_agent_id,
    turn_identity,
)

__all__ = ["TurnIdentity", "normalize_agent_id", "turn_identity"]
