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
# only. Everything fails open: any absence or error yields no override.
#
# The picker saves ONE deny map with a category per pickable kind, and lg-react
# narrows by all three, so the turn reads the WHOLE map once
# (`resolve_turn_selection_disabled`) and slices it (`disabled_category`) —
# one store round trip, not three:
#
#   tools           -> platform/tool_pick.py (plain + code-exec tool groups)
#   mcp             -> platform/tools_mcp.py (servers, keyed by SERVER ID)
#   named_services  -> platform/named_services.py (namespaces + operations)

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.capabilities import (
    DISABLED_MCP,
    DISABLED_NAMED_SERVICES,
    DISABLED_TOOLS,
    disabled_category,
    resolve_turn_disabled_tools,
    resolve_turn_role_models,
    resolve_turn_selection_disabled,
)

__all__ = [
    "DISABLED_MCP",
    "DISABLED_NAMED_SERVICES",
    "DISABLED_TOOLS",
    "disabled_category",
    "resolve_turn_disabled_tools",
    "resolve_turn_role_models",
    "resolve_turn_selection_disabled",
]
