# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── dispatch.py ── the per-agent registry: ONE app, MANY agents ──
#
# Each hosted foreign-runtime agent is described by an AgentSpec: how to BUILD
# its runtime object (graph/session, with its own deps/checkpointer/store), how
# to STREAM it (its own adapter — different agent shapes need different stream
# adapters), how to shape its INPUTS, its accounted model role, and its
# `agent_id` (the row-scope discriminator that keeps co-hosted agents' rows
# apart inside the SHARED tenant/project schema). The build/stream/input
# callables take the entrypoint explicitly so the spec stays a plain value
# object and the entrypoint methods stay thin.
#
# The callables are runtime-typed (`Callable[..., Any]`): the SPEC is neutral,
# the runtime-specific pieces (a LangGraph graph builder, a Claude Code session
# runner, ...) are injected by the app that registers the spec. This module
# imports no agent framework.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.identity import normalize_agent_id


@dataclass(frozen=True)
class AgentSpec:
    """One hosted agent, as the dispatcher sees it.

    - ``agent_id``     — the id the agent is dispatched + configured under (and
      the row-scope discriminator in shared storage).
    - ``role``         — the accounted model role the agent's answer bills under
      (e.g. ``"<agent>.answer"``).
    - ``build_graph``  — ``async (entrypoint, *, disabled=None) -> runtime``:
      build the agent's runtime object FOR THIS TURN (never cached — scaled
      serving: turns hop workers, so no process-local runtime is continuity).
      ``disabled`` is this conversation's saved capability deny map (the whole
      block ``resolve_turn_selection_disabled`` returns, sliced per category by
      the builder); an agent with nothing pickable ignores it.
    - ``stream``       — ``async (runtime, inputs, run_config) -> str``: run one
      turn through the agent's OWN stream adapter; returns the answer text.
    - ``build_inputs`` — ``(question, ident, attachments) -> (inputs, run_config)``.
      ``question`` arrives ALREADY FRAMED (turn frame); ``ident`` is the
      ``TurnIdentity``; ``attachments`` is the turn's materialized multimodal
      blocks (image/document), empty for text-only.
    """

    agent_id: str
    role: str
    build_graph: Callable[..., Any]
    stream: Callable[..., Any]
    build_inputs: Callable[..., Any]


def resolve_agent_spec(
    agents: Mapping[str, AgentSpec],
    requested: Any,
    default_agent_id: str,
) -> AgentSpec:
    """Resolve the ACTIVE agent from a turn's requested agent id.

    Unknown/blank ``requested`` falls back to ``default_agent_id`` — the exact
    dispatch rule ``execute_core`` applies: normalize the id (blank/None folds to
    the default), then fall back again if the normalized id names no registered
    spec. A turn is never refused over an unknown agent id."""
    agent_id = normalize_agent_id(requested, default=default_agent_id)
    if agent_id not in agents:
        agent_id = default_agent_id
    return agents[agent_id]
