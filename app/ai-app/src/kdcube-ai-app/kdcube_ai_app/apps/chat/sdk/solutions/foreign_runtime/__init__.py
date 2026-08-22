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
    declared_python_tool_enabled,
    disabled_category,
    resolve_turn_disabled_mcp,
    resolve_turn_disabled_namespaces,
    resolve_turn_disabled_tools,
    resolve_turn_model_pick,
    resolve_turn_role_models,
    resolve_turn_selection_disabled,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.dispatch import (
    AgentSpec,
    resolve_agent_spec,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.external_events import (
    fold_turn_external_events,
    folded_external_events_message_ids,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.identity import (
    TurnIdentity,
    normalize_agent_id,
    turn_identity,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.mcp_bridge import (
    agent_grant_bearer_provider,
    claude_code_tool_rules,
    connect_required_outcome,
    connection_allowed_tools,
    connection_server_id,
    current_turn_user_sub,
    load_mcp_server_instructions_safe,
    claude_code_mcp_servers,
    narrow_mcp_connections,
    resolve_turn_mcp,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.named_services import (
    named_service_door_servers,
    named_service_door_tools,
    named_service_roster_block,
    named_service_roster_lines,
    named_service_inventory,
    narrow_named_service_inventory,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.stream_contract import (
    content_text,
    tool_call_views,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_tools import (
    WORKSPACE_CHECKOUT_TOOL,
    WORKSPACE_MCP_SERVER_ID,
    WORKSPACE_PUBLISH_TOOL,
    WORKSPACE_PULL_TOOL,
    WORKSPACE_TURN_SUMMARY_TOOL,
    WorkspacePublishError,
    build_workspace_hosting_service,
    checkout_into_workspace,
    publish_workspace_files,
    pull_into_workspace,
    pull_report_text,
    workspace_artifact_root,
    workspace_mcp_server,
    workspace_pull_dir,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.publication import (
    WorkspacePublicationApprover,
    WorkspacePublicationDecision,
    WorkspacePublicationFile,
    WorkspacePublicationPolicy,
    WorkspacePublicationRequest,
    validate_workspace_publication,
    workspace_publication_mime,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_broker import (
    WorkspaceBroker,
    WorkspaceBrokerError,
    broker_source_resolver,
    request_workspace_broker,
    start_workspace_broker,
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
    "WORKSPACE_CHECKOUT_TOOL",
    "WORKSPACE_MCP_SERVER_ID",
    "WORKSPACE_PUBLISH_TOOL",
    "WORKSPACE_PULL_TOOL",
    "WORKSPACE_TURN_SUMMARY_TOOL",
    "WorkspaceBroker",
    "WorkspaceBrokerError",
    "WorkspacePublishError",
    "WorkspacePublicationApprover",
    "WorkspacePublicationDecision",
    "WorkspacePublicationFile",
    "WorkspacePublicationPolicy",
    "WorkspacePublicationRequest",
    "agent_grant_bearer_provider",
    "claude_code_mcp_servers",
    "claude_code_tool_rules",
    "connect_required_outcome",
    "connection_allowed_tools",
    "connection_server_id",
    "content_text",
    "conversation_is_new",
    "checkout_into_workspace",
    "broker_source_resolver",
    "build_workspace_hosting_service",
    "current_turn_user_sub",
    "declared_python_tool_enabled",
    "disabled_category",
    "emit_turn_timing",
    "finalize_conversation_title",
    "fold_turn_external_events",
    "folded_external_events_message_ids",
    "load_mcp_server_instructions_safe",
    "named_service_door_servers",
    "named_service_door_tools",
    "named_service_roster_block",
    "named_service_roster_lines",
    "named_service_inventory",
    "narrow_mcp_connections",
    "narrow_named_service_inventory",
    "normalize_agent_id",
    "persist_turn_artifacts",
    "publish_workspace_files",
    "pull_into_workspace",
    "pull_report_text",
    "resolve_agent_spec",
    "resolve_turn_disabled_mcp",
    "resolve_turn_disabled_namespaces",
    "resolve_turn_disabled_tools",
    "resolve_turn_mcp",
    "resolve_turn_model_pick",
    "resolve_turn_role_models",
    "resolve_turn_selection_disabled",
    "request_workspace_broker",
    "start_workspace_broker",
    "tool_call_views",
    "turn_identity",
    "validate_workspace_publication",
    "workspace_artifact_root",
    "workspace_mcp_server",
    "workspace_pull_dir",
    "workspace_publication_mime",
]
