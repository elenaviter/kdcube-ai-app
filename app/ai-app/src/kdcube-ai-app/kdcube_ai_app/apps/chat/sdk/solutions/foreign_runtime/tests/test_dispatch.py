# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""The per-agent dispatch registry (foreign_runtime/dispatch.py).

Unknown/blank agent ids fall back to the default spec — a turn is never
refused over an unknown agent id — and distinct registered specs resolve to
themselves. Mirrors the dispatch rule the ported bundle's execute_core applies.
"""
from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.dispatch import (
    AgentSpec,
    resolve_agent_spec,
)


def _spec(agent_id: str) -> AgentSpec:
    return AgentSpec(
        agent_id=agent_id,
        role=f"{agent_id}.answer",
        build_graph=lambda ep, **kw: None,
        stream=lambda graph, inputs, cfg: "",
        build_inputs=lambda q, ident, att: ({}, {}),
    )


AGENTS = {"lg-solution": _spec("lg-solution"), "lg-react": _spec("lg-react")}


def test_distinct_specs_resolve_to_themselves() -> None:
    sol = resolve_agent_spec(AGENTS, "lg-solution", "lg-solution")
    pre = resolve_agent_spec(AGENTS, "lg-react", "lg-solution")
    assert sol is AGENTS["lg-solution"]
    assert pre is AGENTS["lg-react"]
    assert sol.role != pre.role


def test_unknown_agent_id_falls_back_to_default() -> None:
    spec = resolve_agent_spec(AGENTS, "no-such-agent", "lg-solution")
    assert spec is AGENTS["lg-solution"]


def test_blank_and_none_agent_id_fall_back_to_default() -> None:
    assert resolve_agent_spec(AGENTS, "", "lg-solution") is AGENTS["lg-solution"]
    assert resolve_agent_spec(AGENTS, None, "lg-solution") is AGENTS["lg-solution"]
    assert resolve_agent_spec(AGENTS, "   ", "lg-react") is AGENTS["lg-react"]


def test_spec_is_frozen() -> None:
    spec = AGENTS["lg-solution"]
    try:
        spec.role = "other"  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised
