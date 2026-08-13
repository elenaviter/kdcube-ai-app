# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
"""The shared foreign-runtime turn seam.

The platform-side building blocks for hosting a FOREIGN agent runtime
(LangGraph, Claude Code, ...) behind a KDCube run-to-completion turn:
per-turn identity isolation, the ingress-batch event fold, the per-agent
dispatch registry, the generic stream-adapter pieces, run-to-completion turn
recording (timing / artifacts / first-turn title), delegated MCP resolution,
and the per-turn capability (model pick / tool opt-out) seam.

Runtime adapters plug in: every runtime-typed piece (graph builders, stream
adapters, tool binders) is injected by the hosting app — no langchain/langgraph
is imported anywhere in this package. Extracted from the ported-langgraph-agents
example bundle's platform layer; that bundle refactors onto this seam in a
later phase.
"""

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.capabilities import (
    resolve_turn_disabled_tools,
    resolve_turn_role_models,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.dispatch import (
    AgentSpec,
    resolve_agent_spec,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.external_events import (
    fold_turn_external_events,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.identity import (
    TurnIdentity,
    normalize_agent_id,
    turn_identity,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.mcp_bridge import (
    agent_grant_bearer_provider,
    connect_required_outcome,
    current_turn_user_sub,
    load_mcp_server_instructions_safe,
    resolve_turn_mcp,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.stream_contract import (
    content_text,
    tool_call_views,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.turn_record import (
    conversation_is_new,
    emit_turn_timing,
    finalize_conversation_title,
    persist_turn_artifacts,
)

__all__ = [
    "AgentSpec",
    "TurnIdentity",
    "agent_grant_bearer_provider",
    "connect_required_outcome",
    "content_text",
    "conversation_is_new",
    "current_turn_user_sub",
    "emit_turn_timing",
    "finalize_conversation_title",
    "fold_turn_external_events",
    "load_mcp_server_instructions_safe",
    "normalize_agent_id",
    "persist_turn_artifacts",
    "resolve_agent_spec",
    "resolve_turn_disabled_tools",
    "resolve_turn_mcp",
    "resolve_turn_role_models",
    "tool_call_views",
    "turn_identity",
]
