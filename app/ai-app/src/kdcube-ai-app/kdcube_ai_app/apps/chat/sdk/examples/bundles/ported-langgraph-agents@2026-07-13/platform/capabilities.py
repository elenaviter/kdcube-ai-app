# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── platform/capabilities.py ── the per-turn, per-agent model-pick seam ──
#
# Lifted into the SDK's shared foreign-runtime seam
# (`kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.capabilities`); this
# module re-exports it unchanged so the bundle keeps one import home for the seam.
#
# In this app the generic `simple_model_pick` provider is declared PER AGENT in
# config (`surfaces.as_consumer.agents.lg-solution` / `.lg-react`), so the
# Capabilities widget is active for each agent with zero adapter code. The
# resolved pick is bound by `entrypoint.py` onto `bundle_call_context.role_models`
# around that agent's graph run — the KDCube model router overlays it on that
# agent's answer role (`lg-solution.answer` / `lg-react.answer`) for that turn
# only. The tool opt-outs feed `platform/tool_pick.py` (admin ceiling ∩
# user-enabled). Everything fails open: any absence or error yields no override.

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.capabilities import (
    resolve_turn_disabled_tools,
    resolve_turn_role_models,
)

__all__ = ["resolve_turn_disabled_tools", "resolve_turn_role_models"]
