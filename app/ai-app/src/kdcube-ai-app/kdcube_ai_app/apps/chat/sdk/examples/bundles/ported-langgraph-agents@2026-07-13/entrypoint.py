# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── entrypoint.py ── composition root ── ONE app, MANY agents ──
#
# This app hosts TWO ported LangGraph agents behind a SINGLE `execute_core`,
# dispatched by `agent_id`:
#
#   - `lg-solution` — the preserved research graph under
#     `solution/lg_solution/` (KB retrieval + per-user pgvector memory + a nested
#     subagent). Linear shape with a DEDICATED answer node, so its stream adapter
#     (platform/stream_solution.py) streams that node's tokens.
#   - `lg-react` — the `langchain.agents.create_agent` solution under
#     `solution/lg_prebuilt/` (plain + MCP tools). Its `model` node LOOPS
#     (once per tool-decision cycle) with no dedicated answer node, so its stream
#     adapter (platform/stream_prebuilt.py) streams ONLY the final model turn.
#
# Both solution packages preserve their framework/domain boundaries from the
# standalone "before" (poc/lg-solution, poc/lg-prebuilt-agent). Deliberate async,
# configuration, model-injection, package-import, and prompt-composition seams are
# explicit. The platform integration is:
#
#   1. a dispatcher      — execute_core resolves agent_id and runs the right graph
#                          through its own stream adapter (the teaching point:
#                          different agent shapes -> different stream adapters,
#                          selected by agent_id)
#   2. scaled serving    — this app builds a fresh graph bound to each turn's
#                          model/tool choices. Durable state lives in shared Postgres
#                          (checkpointer, keyed by thread_id); only connections are
#                          long-lived on self.
#   3. isolation         — platform identity + agent_id -> per-agent per-user keys
#                          (platform/identity.py), plus one tenant/project schema
#                          with app-prefixed tables and agent_id row scope
#   4. streaming         — astream_events -> comm_ctx (the two stream adapters)
#   5. a shared ingress  — a Telegram Bot API webhook (platform/telegram.py) that
#                          drives the SAME execute_core for the DEFAULT agent
#
# Persistence is mapped by DATA KIND (see docs/storage/README.md): each agent's OWN
# working store (memory/KB + the LangGraph checkpointer) is routed onto KDCube's
# SHARED Postgres (self.pg_pool) in the ONE tenant/project schema when hosted,
# with app-prefixed tables and agent_id row scope; it uses its own
# DATABASE_URL standalone, in-memory offline. KDCube owns the conversation record.

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from kdcube_ai_app.apps.chat.sdk.protocol import (
    ExternalEventPayload,
    external_events_text,
    external_events_texts,
)
from kdcube_ai_app.apps.chat.sdk.util import _now_ms
from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import bind_current_bundle_call_context_patch
from kdcube_ai_app.apps.chat.sdk.tools.backends.summary.conversation_title import (
    generate_conversation_title,
    emit_conversation_title_event,
)
from kdcube_ai_app.infra.service_hub.inventory import Config
from kdcube_ai_app.infra.plugin.bundle_loader import api, bundle_entrypoint, bundle_id, ui_widget
from kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint_with_economic import (
    BaseEntrypointWithEconomics,
)
# Reused SDK adapter (unchanged): route a create_react agent's model calls through
# KDCube's accounted, still-streaming model service.
from kdcube_ai_app.apps.chat.sdk.frameworks.langchain import KDCubeChatModel

# The preserved solution packages, imported package-relative as subpackages.
#   lg-solution (the research graph)
from .solution.lg_solution.deps import build_deps as build_solution_deps
from .solution.lg_solution.graph import build_graph as build_solution_graph
from .solution.lg_solution.config import get_config as get_solution_config
from .solution.lg_solution._pg import StorageScope
from .solution.lg_solution.memory import ensure_memory_tables
from .solution.lg_solution.knowledge import ensure_kb_tables
#   lg-react (the create_react agent)
from .solution.lg_prebuilt.agent import build_agent as build_prebuilt_agent, AGENT_NODE as PREBUILT_AGENT_NODE
from .solution.lg_prebuilt.tools import plain_tool_registry
from .solution.lg_prebuilt.config import get_config as get_prebuilt_config
from .solution.lg_prebuilt.llm import StubChatModel

# The platform seams.
from .platform.pg_target import resolve_solution_pg, resolve_solution_memory, schema_for_scope
from .platform.turn_batch import fold_turn_external_events
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.live_watch import (
    STOPPED_ANSWER,
    run_until_stopped,
)
from .platform.identity import turn_identity
from .platform.stop_repair import repair_unanswered_tool_calls
from .platform.stream_solution import stream_graph_turn
from .platform.stream_prebuilt import stream_react_turn
from .platform.tools_mcp import load_mcp_tools_for_connections, consent_request_tools
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_mcp import (
    delegated_client_id_for_agent,
)
# The shared foreign-runtime seam (SDK): per-agent dispatch, delegated-MCP
# identity/bearer, and run-to-completion turn recording.
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.dispatch import (
    AgentSpec,
    resolve_agent_spec,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.mcp_bridge import (
    agent_grant_bearer_provider,
    current_turn_user_sub,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.turn_record import (
    conversation_is_new,
    emit_turn_timing,
    finalize_conversation_title,
    persist_turn_artifacts,
    record_foreign_runtime_turn_log,
)
from .platform.tool_pick import agent_tool_connections, run_python_bound, select_bound_tools
from .platform.turn_workspace import (
    build_checkout_tool,
    build_pull_files_tool,
    build_read_file_tool,
    frame_turn_input,
    prepare_turn_workspace,
)
from .platform.capabilities import (
    DISABLED_MCP,
    DISABLED_NAMED_SERVICES,
    DISABLED_TOOLS,
    disabled_category,
    resolve_turn_role_models,
    resolve_turn_selection_disabled,
)
from .platform.named_services import build_named_services_block, narrow_bound_door_tools
from .platform.code_exec import build_code_exec_context, code_exec_scope
from .platform import telegram as telegram_ingress

LOGGER = logging.getLogger("kdcube.ported_langgraph_agents")

BUNDLE_ID = "ported-langgraph-agents@2026-07-13"

# The agent the app dispatches to when a turn declares no agent_id (both ingresses,
# including the Telegram webhook, land here by default).
DEFAULT_AGENT_ID = "lg-solution"

# ── lg-solution constants ────────────────────────────────────────────────────
# Graph nodes surfaced to the chat timeline as progress steps; the dedicated
# answer node whose model tokens are the user-visible answer. `compact` runs the
# accounted conversation-summary call — surfaced as a step, NEVER as the answer.
SOLUTION_STEP_NODES = {"compact", "retrieve", "plan", "delegate", "answer"}
SOLUTION_ANSWER_NODE = "answer"
# The model role lg-solution's chat + embedding calls bill under, and the economics
# flow label for its retrieval/memory embeddings.
SOLUTION_ANSWER_ROLE = "lg-solution.answer"
SOLUTION_RETRIEVAL_FLOW = "lg-solution.retrieval"
# A DISTINCT accounted role for lg-solution's conversation-compaction summary call,
# so it bills apart from the answer and (running in the non-answer `compact` node)
# is never streamed to the user as the answer.
SOLUTION_SUMMARY_ROLE = "lg-solution.summary"

# ── lg-react constants ────────────────────────────────────────────────────
# The agent id lg-react is dispatched + configured under. Its per-agent config
# (capabilities, tools) lives at `surfaces.as_consumer.agents.lg-react`.
PREBUILT_AGENT_ID = "lg-react"
# The model role the create_react agent's chat calls bill under.
PREBUILT_ANSWER_ROLE = "lg-react.answer"
# A DISTINCT accounted role for lg-react's compaction (LangMem SummarizationNode)
# summary call, so it bills apart from the answer and (running in the
# `summarization` pre-model node, not the `agent` node) is never streamed.
PREBUILT_SUMMARY_ROLE = "lg-react.summary"


# ── provider-surface visibility helpers ──────────────────────────────────────
# A widget/api is registered in the bundle MANIFEST by its decorator (not by the
# ui.widgets build config); these helpers give the decorators an open default
# with a config override path, so `config.visibility.widget.<alias>` /
# `.api...` can tighten it later without editing code. Empty selectors = open.

def _api_visibility(
    alias: str,
    *,
    route: str = "operations",
    method: str = "POST",
    user_types: Tuple[str, ...] = (),
    roles: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    base = f"surfaces.as_provider.api.{route}.{alias}.{method.upper()}.visibility"
    return {
        "user_types": user_types,
        "user_types_config": f"{base}.user_types",
        "roles": roles,
        "roles_config": f"{base}.roles",
    }


def _widget_visibility(
    alias: str,
    *,
    user_types: Tuple[str, ...] = (),
    roles: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    base = f"surfaces.as_provider.widget.{alias}.visibility"
    return {
        "user_types": user_types,
        "user_types_config": f"{base}.user_types",
        "roles": roles,
        "roles_config": f"{base}.roles",
    }


# ── the per-agent registry ───────────────────────────────────────────────────
# Each agent is described by an `AgentSpec` (the shared foreign-runtime seam's
# value object): how to BUILD its graph (with its own deps/checkpointer/store),
# how to STREAM it (its own adapter), how to shape its INPUTS, its model role,
# and its `agent_id` (the row-scope discriminator that keeps the two agents'
# rows apart inside the SHARED tenant/project schema). The module-level
# build/stream/input functions take the entrypoint explicitly so the spec stays
# a plain value object and the entrypoint methods stay thin.


def _solution_scope(ep: "LGPortedAgentsBundle", agent_id: str) -> StorageScope:
    """The row-level storage scope for one agent, resolved from the process runtime
    identity: tenant/project from platform settings (mirrors `UserMemoryStore` /
    `conv_index`), bundle_id from the ai_bundle_spec, agent_id the dispatched agent.
    Carried as DATA and written into every memory/KB row's scope columns."""
    return StorageScope(
        tenant=str(getattr(ep.settings, "TENANT", "") or ""),
        project=str(getattr(ep.settings, "PROJECT", "") or ""),
        bundle_id=ep._named_services_bundle_id(),
        agent_id=agent_id,
    )


async def _build_solution_graph(
    ep: "LGPortedAgentsBundle",
    *,
    disabled: Optional[Mapping[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build lg-solution's research graph. (`disabled` is accepted for a
    uniform build signature but ignored — the linear research graph has no pickable
    tool loop.) Route its pgvector memory + KB onto
    KDCube's SHARED asyncpg pool (in the ONE per-tenant/project schema, rows scoped
    by columns) and its checkpointer onto the same shared Postgres via a psycopg
    DSN, hand the graph an accounted + economics-guarded model/embedding edge, open
    its checkpointer, seed the KB."""
    del disabled, state
    own = get_solution_config()
    scope = _solution_scope(ep, "lg-solution")
    schema = schema_for_scope(scope.tenant, scope.project)
    database_url = await ep._hosted_database_url(own.database_url, schema)
    # Hosted config comes from the DESCRIPTOR: the answer model's output budget
    # overlays the solution config's standalone default (see _agent_max_tokens).
    config = replace(
        own,
        database_url=database_url,
        max_tokens=ep._agent_max_tokens("lg-solution", own.max_tokens),
    )
    # memory + KB drive KDCube's asyncpg pool DIRECTLY (no psycopg), in the shared
    # tenant/project schema; the agent's rows stay apart via the agent_id column.
    mem = resolve_solution_memory(getattr(ep, "pg_pool", None), schema)
    deps = build_solution_deps(
        config=config,
        models_service=getattr(ep, "models_service", None),
        model_role=SOLUTION_ANSWER_ROLE,
        embedding_service=ep._solution_embedding_service,
        pg_pool=mem.pool,
        schema=mem.schema,
        scope=scope,
        # The compact node's summary call bills on a DISTINCT accounted role and,
        # running outside the answer node, never streams as the answer.
        summary_model_role=SOLUTION_SUMMARY_ROLE,
    )
    checkpointer = await ep._open_checkpointer("lg-solution", deps.config.database_url)
    try:
        await deps.knowledge.seed()  # best-effort; no-op if the DB is unreachable
    except Exception:
        pass
    return build_solution_graph(deps, checkpointer=checkpointer)


def _current_turn_user_sub(ep: "LGPortedAgentsBundle") -> str:
    """This turn's user subject, resolved from the BOUND turn context at graph-build
    time — the accounting context (bound around execute_core), else the comm. Used to
    mint the per-user delegated MCP bearer without threading identity through the
    build signatures. Empty when no user is bound (a delegated MCP connection then
    resolves to nothing — no unauthenticated call). Delegates to the shared
    foreign-runtime seam."""
    return current_turn_user_sub(ep)


async def _build_prebuilt_graph(
    ep: "LGPortedAgentsBundle",
    *,
    disabled: Optional[Mapping[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build lg-react's create_agent graph FOR THIS TURN (never cached).

    Tools are DECLARED as a connection list on the agent
    (`surfaces.as_consumer.agents.lg-react.tools`, the standard KDCube shape) — the
    ADMIN CEILING. This binds EXACTLY the declared capabilities the user has NOT
    opted out of this turn, so the picker takes effect: what the admin does not
    declare is never bound; what is declared is on by default and the user may opt
    out per conversation. `disabled` is the saved capabilities-widget deny map,
    with one category per pickable kind, and lg-react narrows by all three:

      * `tools`          — the plain + code-exec tool groups (by group alias);
      * `mcp`            — MCP servers by SERVER ID (a server turned off is never
                           dialled) and individual tools under a surviving server;
      * `named_services` — the namespaces this agent consumes and the operations
                           it may run in each (the inventory the prompt states).

    Checkpointer reused across per-turn builds. `build_agent` remains
    solution-owned and injectable."""
    disabled_tools = disabled_category(disabled, DISABLED_TOOLS)
    disabled_mcp = disabled_category(disabled, DISABLED_MCP)
    disabled_namespaces = disabled_category(disabled, DISABLED_NAMED_SERVICES)
    own = get_prebuilt_config()
    schema = schema_for_scope(
        str(getattr(ep.settings, "TENANT", "") or ""),
        str(getattr(ep.settings, "PROJECT", "") or ""),
    )
    database_url = await ep._hosted_database_url(own.database_url, schema)
    config = replace(own, database_url=database_url)
    model = ep._build_prebuilt_model(config)
    summary_model = ep._build_prebuilt_summary_model(config)

    connections = agent_tool_connections(ep, PREBUILT_AGENT_ID)

    def _run_python_factory() -> Any:
        # Lazy + package-relative: only import the code-exec tool when the exec
        # connection is declared and enabled for this turn.
        from .platform.code_exec_tool import build_run_python_tool
        return build_run_python_tool()

    def _pull_files_factory() -> Any:
        # Workspace companion: materialize conversation files into the sandbox
        # working directory by their conv:fi: links.
        return build_pull_files_tool()

    def _read_file_factory() -> Any:
        # Workspace companion: view one conversation file in visible context
        # (text bounded; images/PDF as visual payloads — react.read semantics).
        return build_read_file_tool()

    def _checkout_factory() -> Any:
        # Workspace companion: create/reset editable current-turn state from a
        # durable file or owner-provided object ref.
        return build_checkout_tool()

    def _web_tool_factories() -> Dict[str, Callable[[], Any]]:
        # web_search / web_fetch run PAID backends (search provider + an LLM
        # that filters/segments the retrieved results) — the LLM side binds to
        # this app's ACCOUNTED model service, the search side meters through
        # the ambient turn accounting. Without a model service (standalone
        # offline) the tools stay unbound even if declared.
        svc = getattr(ep, "models_service", None)
        if svc is None:
            return {}

        def _search() -> Any:
            from .platform.web_tools import build_web_search_tool
            return build_web_search_tool(svc)

        def _fetch() -> Any:
            from .platform.web_tools import build_web_fetch_tool
            return build_web_fetch_tool(svc)

        return {"web_search": _search, "web_fetch": _fetch}

    extra_factories = _web_tool_factories()
    if isinstance(state, dict):
        from .platform.turn_context import build_record_turn_summary_tool

        extra_factories["record_turn_summary"] = lambda: build_record_turn_summary_tool(state)

    # Plain + code-exec tools narrowed to (admin-declared − user-disabled) this turn.
    tools = select_bound_tools(
        connections,
        disabled_tools,
        plain_registry=plain_tool_registry(),
        run_python_factory=_run_python_factory,
        pull_files_factory=_pull_files_factory,
        read_file_factory=_read_file_factory,
        checkout_factory=_checkout_factory,
        extra_factories=extra_factories,
    )
    # MCP tools declared as `kind: mcp` connections (optional; degrades to none).
    # A `delegated: true` connection is served under the AGENT's delegated-client
    # identity (this bundle's spec id + agent_id) — the agent is a "Delegated By
    # KDCube" entity, so consent is per-agent. The bearer is the token the user's
    # per-agent grant already bound, read through the Connection Hub named service;
    # when the user has not consented, no token is returned and we get a consent
    # demand per connection instead of the tools.
    application = ep._named_services_bundle_id()
    agent_client_id = delegated_client_id_for_agent(application, PREBUILT_AGENT_ID)
    mcp_server_guides: Dict[str, str] = {}
    mcp_tools, mcp_consents = await load_mcp_tools_for_connections(
        connections, user_sub=_current_turn_user_sub(ep), disabled_map=disabled_mcp,
        application=application, agent_id=PREBUILT_AGENT_ID,
        bearer_provider=_agent_grant_bearer_provider(ep, agent_client_id),
        instructions_sink=mcp_server_guides,
    )
    tools += mcp_tools
    # The named-service door publishes one tool per operation plus a generic
    # `named_services_call`, and the declared inventory is the administrator's
    # ceiling of operations. What the conversation's pick removed is removed here
    # too — otherwise the agent reads a narrowed inventory while still holding the
    # tools it names. The namespace half stays with the door's own gate (one set
    # of tools serves every namespace), which is why the inventory is also stated in
    # words.
    tools = narrow_bound_door_tools(tools, connections, disabled_namespaces)
    # Consent is demand-driven per tool: a build cannot know which capabilities
    # THIS turn will use, so pending connections raise nothing here. Each binds
    # a consent-gated stub instead — when the model decides the user's request
    # needs that capability, calling the stub raises exactly that connection's
    # consent demand in chat (the same attempt-time semantics connected-account
    # tools have) and returns the agent-explainable consent result.
    tools += consent_request_tools(mcp_consents, announce=_announce_mcp_consent)

    # The named-services instruction block: the agent-neutral SDK teaching block
    # (HOW to work a namespace, named with the tools this turn actually bound)
    # plus the foreign-runtime seam's namespace INVENTORY — the one block every
    # wrapped runtime renders, narrowed by this conversation's pick. Empty when
    # this agent has no namespaces this turn. Never raises (see the module).
    tool_names = {str(getattr(t, "name", "") or "") for t in tools}
    ns_block, connected_ns = await build_named_services_block(
        bundle_props=getattr(ep, "bundle_props", None) or {},
        agent_id=PREBUILT_AGENT_ID,
        connections=connections,
        disabled_namespaces=disabled_namespaces,
        tool_names=tool_names,
        redis=getattr(ep, "redis", None),
        tenant=str(getattr(ep.settings, "TENANT", "") or ""),
        project=str(getattr(ep.settings, "PROJECT", "") or ""),
        pull_tool="pull_files" if "pull_files" in tool_names else "run_python",
        read_tool="read_file" if "read_file" in tool_names else "",
    )

    # Config-declared agent-admin customization — the SAME convention the ReAct
    # harness honors: the descriptor prop is the administrator's voice, appended
    # inside the SDK's admin-customization HARD-OVERRIDE envelope. Key:
    # `surfaces.as_consumer.agents.lg-react.additional_instructions`.
    additional_instructions = ""
    try:
        additional_instructions = str(ep.bundle_prop(
            f"surfaces.as_consumer.agents.{PREBUILT_AGENT_ID}.additional_instructions", ""
        ) or "").strip()
    except Exception:
        additional_instructions = ""

    checkpointer = await ep._open_checkpointer("lg-react", config.database_url)
    return build_prebuilt_agent(
        config, model=model, tools=tools, checkpointer=checkpointer, summary_model=summary_model,
        system_prompt=_prebuilt_system_prompt(
            tools, mcp_consents, named_services_block=ns_block,
            additional_instructions=additional_instructions,
            connected_namespaces=connected_ns,
            mcp_server_guides=mcp_server_guides,
        ),
    )


def _agent_grant_bearer_provider(ep: "LGPortedAgentsBundle", client_id: str):
    """A bearer provider that reads THIS agent's consented per-agent grant token
    from the Connection Hub named service (`agent_grant.get_token`) for the turn's
    user. Returns None on any absence/failure (consent pending / caller unbound /
    hub unreachable) so the delegated connection is dropped and surfaces as a
    consent demand — never a blind call, never a failed build. Delegates to the
    shared foreign-runtime seam."""
    return agent_grant_bearer_provider(ep, client_id)


async def _announce_mcp_consent(c: Any) -> None:
    """Raise ONE pending MCP consent as the standard chat consent banner — the
    consent-gated stub calls this at the tool ATTEMPT, so only the capability
    the turn actually needs asks the user. The SDK announcer records the demand
    once per conversation and re-emits the event on later attempts while the
    block is still real (best effort; never raises)."""
    from kdcube_ai_app.apps.chat.sdk.integrations.prokura.mcp_consent import (
        announce_agent_consent,
    )

    await announce_agent_consent(c)


def _prebuilt_system_prompt(
    tools: List[Any],
    mcp_consents: Optional[List[Any]] = None,
    named_services_block: str = "",
    additional_instructions: str = "",
    connected_namespaces: Optional[List[str]] = None,
    mcp_server_guides: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """The lg-react system prompt for this turn's tool binding.

    Per-tool mechanics live in each tool's own description (the provider-native
    declarations); this prompt carries the CAPABILITY level — the facts that
    reshape strategy and cannot ride in a signature. Block order:

    1. the agent's own prose (solution-owned)
    2. conduct + trust guards (always: confidentiality, untrusted content,
       no background promises, elaboration, gender, tech-evolution)
    3. distributed-turn-workspace guide (when `run_python` is bound)
    4. capability story: exec-as-hands (with `run_python`) OR the prose-only
       output medium block (without it — chat is the only deliverable)
    5. [Consent-gated tools] note (while delegated MCP consents are pending)
    6. named-services block + sandbox caveat (when namespaces are connected;
       the short hand note when only ns tools are bound)
    7. conversation-recovery block (when a `conv` namespace is connected —
       history beyond the visible window is searchable, not lost)
    8. shared turn-context guide (when `record_turn_summary` is bound)
    9. admin customization envelope (descriptor prop) — ALWAYS LAST, the same
       SDK envelope the ReAct harness uses."""
    from kdcube_ai_app.apps.chat.sdk.skills.instructions.workspace_agent_instructions import (
        conversation_recovery_guide,
        exec_capability_guide,
        prose_only_output_guide,
        turn_summary_contribution_guide,
        workspace_agent_conduct_guards,
    )
    from .solution.lg_prebuilt.agent import SYSTEM_PROMPT

    names = {str(getattr(tool, "name", "") or "") for tool in tools or []}
    has_exec = "run_python" in names
    pull_tool = "pull_files" if "pull_files" in names else "run_python"

    parts: List[str] = [SYSTEM_PROMPT.strip(), workspace_agent_conduct_guards()]
    if has_exec:
        from kdcube_ai_app.apps.chat.sdk.skills.instructions.shared_instructions import (
            distributed_turn_workspace_guide,
        )

        parts.append(distributed_turn_workspace_guide(
            exec_tool="run_python",
            pull_tool=pull_tool,
            read_tool="read_file" if "read_file" in names else "",
            checkout_tool="checkout" if "checkout" in names else "",
        ).strip())
        parts.append(exec_capability_guide(exec_tool="run_python", pull_tool=pull_tool))
    else:
        parts.append(prose_only_output_guide())

    consent_note = _mcp_consent_prompt_note(mcp_consents).strip()
    if consent_note:
        parts.append(consent_note)

    if str(named_services_block or "").strip():
        parts.append(named_services_block.strip() + _named_services_sandbox_caveat(tools))
    else:
        legacy_note = _named_services_usage_note(tools).strip()
        if legacy_note:
            parts.append(legacy_note)

    # Each MCP server's own operating guide (its initialize-result
    # `instructions`) — the text MCP-native clients surface to their model and
    # the LangChain tool loader drops. The server is the authority on its own
    # workflow (e.g. the named-services door's list-first / schema-first
    # ritual with its exact tool names).
    for server_id, guide in sorted((mcp_server_guides or {}).items()):
        text = str(guide or "").strip()
        if text:
            parts.append(f"[SERVICE GUIDE — {server_id}]\n{text}")

    conv_ns = next(
        (str(ns) for ns in (connected_namespaces or []) if str(ns).strip().lower() == "conv"),
        None,
    )
    if conv_ns:
        parts.append(conversation_recovery_guide(
            namespace=conv_ns,
            pull_tool=pull_tool if has_exec else "",
        ))

    if "record_turn_summary" in names:
        parts.append(turn_summary_contribution_guide())

    prompt = "\n\n".join(part for part in parts if part)
    if str(additional_instructions or "").strip():
        from kdcube_ai_app.apps.chat.sdk.solutions.react.decision_prompt import (
            append_agent_admin_customization,
        )

        prompt = append_agent_admin_customization(
            prompt, additional_instructions=additional_instructions
        )
    return prompt or None


def _named_services_sandbox_caveat(tools: List[Any]) -> str:
    """The bundle-specific file/network truth appended under the SDK block:
    the exec sandbox has no network egress, so URL flows can never run there —
    the ref-based action path is the only working one."""
    names = {str(getattr(t, "name", "") or "") for t in tools or []}
    if "run_python" not in names:
        return ""
    return (
        "\nSandbox note: `run_python` has no network access — fetching or "
        "POSTing URLs from it always fails; URL-based upload flows belong to "
        "external clients, and your path is the ref-based action."
    )


def _named_services_usage_note(tools: List[Any]) -> str:
    """Usage guidance appended when the named-services bridge tools are bound.

    The bridge is generic: each namespace declares its own objects, actions,
    and files contract, and the ONLY reliable way to call it is schema-first.
    Files move by reference — the service hosts bytes server-side; the exec
    sandbox has no network egress, so URL flows can never run there."""
    if not any(str(getattr(t, "name", "") or "").startswith("named_services") for t in tools or []):
        return ""
    return (
        "\n\n[Named-service tools] These are generic doors into per-namespace "
        "services (mail, slack, ...). Before the FIRST call into a namespace, "
        "call its schema/about operation and follow what it declares — objects, "
        "actions, argument shapes, and the files contract. Files move BY "
        "REFERENCE: pass the file's ref (a conv:fi:... or staged:... reference) "
        "to the service's own upload/file action and the service hosts the bytes "
        "server-side. run_python has no network access — fetching or POSTing "
        "URLs from it always fails; URL-based upload flows belong to external "
        "clients, and your path is the ref-based action. Files coming BACK to "
        "the user ride the same rule: when a result says a file was delivered "
        "as a file card, the user already sees it in this chat — mention it in "
        "words. Never construct, guess, or re-type download URLs into your "
        "message; a link you type by hand arrives broken."
    )


def _mcp_consent_prompt_note(mcp_consents: Optional[List[Any]]) -> str:
    if not mcp_consents:
        return ""
    lines = []
    for c in mcp_consents:
        claims = ", ".join(getattr(c, "claims", []) or []) or "required access"
        label = str((getattr(c, "consent", {}) or {}).get("tool_name") or "a tool")
        lines.append(f"- {label}: consent-gated; needs the user's approval for {claims}")
    return (
        "\n\n[Consent-gated tools] These capabilities exist but need the user's "
        "approval for YOU (this agent) before they work:\n"
        + "\n".join(lines)
        + "\nWhen the user's request needs one of them, CALL that tool: the call "
        "raises a consent request the user can approve right in this chat, and "
        "returns the consent status. Tell the user you have asked for their "
        "approval and what it covers. After they grant it, the real capability "
        "is available from the next message on."
    )


def _solution_inputs(
    question: str, ident: Any, attachments: list
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # `question` arrives ALREADY FRAMED (`frame_turn_input`: turn start + user
    # message + arriving-file links) — the frame is part of the turn, so it
    # rides the checkpointed `messages` history too. `attachments` stays a
    # pass-through slot for callers that add multimodal blocks explicitly.
    inputs = {
        "question": question,
        "user_id": ident.user_id,
        "messages": [("user", question)],
        "attachments": list(attachments or []),
    }
    run_config = {"configurable": {"thread_id": ident.thread_id}}
    return inputs, run_config


def _prebuilt_inputs(
    question: str, ident: Any, attachments: list
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # `question` arrives ALREADY FRAMED (`frame_turn_input`); create_react's
    # `messages` list IS the model input and the checkpointed history, so the
    # frame lands in both. `attachments` stays a pass-through for explicit
    # multimodal blocks; plain turns stay a plain ("user", text) tuple.
    if attachments:
        from langchain_core.messages import HumanMessage

        user_msg: Any = HumanMessage(
            content=[{"type": "text", "text": question}, *attachments]
        )
    else:
        user_msg = ("user", question)
    inputs = {"messages": [user_msg]}
    run_config = {"configurable": {"thread_id": ident.thread_id}}
    return inputs, run_config


async def _stream_solution(graph: Any, inputs: Dict[str, Any], run_config: Dict[str, Any]) -> str:
    return await stream_graph_turn(
        graph, inputs, run_config,
        answer_node=SOLUTION_ANSWER_NODE, step_nodes=SOLUTION_STEP_NODES,
    )


async def _stream_prebuilt(graph: Any, inputs: Dict[str, Any], run_config: Dict[str, Any]) -> str:
    return await stream_react_turn(graph, inputs, run_config, agent_node=PREBUILT_AGENT_NODE)


AGENTS: Dict[str, AgentSpec] = {
    "lg-solution": AgentSpec(
        agent_id="lg-solution",
        role=SOLUTION_ANSWER_ROLE,
        build_graph=_build_solution_graph,
        stream=_stream_solution,
        build_inputs=_solution_inputs,
    ),
    "lg-react": AgentSpec(
        agent_id="lg-react",
        role=PREBUILT_ANSWER_ROLE,
        build_graph=_build_prebuilt_graph,
        stream=_stream_prebuilt,
        build_inputs=_prebuilt_inputs,
    ),
}


# Bind the reusable Telegram SDK subsystem to this bundle's storage once, at module
# load. Safe when Telegram is unconfigured: with no bot token / webhook secret the
# webhook simply rejects every call. The webhook drives the DEFAULT agent.
telegram_ingress.configure(bundle_id=BUNDLE_ID)


@bundle_entrypoint(name="ported-langgraph-agents", version="1.0.0", priority=10)
@bundle_id(id=BUNDLE_ID)
class LGPortedAgentsBundle(BaseEntrypointWithEconomics):
    """Hosts TWO ported LangGraph agents behind one `execute_core`, dispatched by
    `agent_id` — the "one app, many agents" demonstration.

    Both agents are RUN-TO-COMPLETION: each turn runs to completion and does not
    consume in-turn followups/steers (a mid-turn followup is promoted to the next
    turn). Each is streamed to the reusable chat component, isolated per (agent,
    user, conversation), and recorded by the platform-owned conversation log.

    Economic ENFORCEMENT (T2b) closes the backend loop on top of T2a accounting:
      - Turn-level — deriving `BaseEntrypointWithEconomics` wraps every turn in the
        budget preflight + rate limiter, so overspend is blocked before
        `execute_core` runs (covers chat generation for either agent).
      - Per-call — lg-solution's retrieval/memory embeddings route through the
        economics-guarded search facade, budget-checked per embed. With no economics
        runtime the facade degrades to the raw accounted service, so both agents
        still run offline."""

    def __init__(
        self,
        config: Config,
        pg_pool: Any = None,
        redis: Any = None,
        comm_context: ExternalEventPayload = None,
    ):
        super().__init__(
            config=config,
            pg_pool=pg_pool,
            redis=redis,
            comm_context=comm_context,
        )
        # SCALED-SERVING INVARIANT: this app builds a fresh graph bound to each
        # turn's model/tool selection. KDCube is distributed — turn 1 can land on
        # worker 1 and turn 2 on worker 2 — so no process-local graph is continuity.
        # The bound graph instance exists for one turn only. The only
        # long-lived things on `self` are CONNECTIONS (the
        # checkpointer, like `pg_pool`): infra, re-established lazily per worker, not
        # rebuildable state. Every mutable byte is in shared Postgres keyed per
        # (agent, user, conversation). So ANY worker can serve ANY turn.
        self._checkpointer_cms: Dict[str, Any] = {}      # held so async PG savers aren't GC'd
        self._checkpointers: Dict[str, Any] = {}         # opened saver, reused across per-turn builds
        self._checkpointer_locks: Dict[str, asyncio.Lock] = {}  # open-once guard per agent

    # ── one-time provisioning (bundle load) ───────────────────────────────────

    async def on_bundle_load(self, **kwargs: Any) -> None:
        """Provision the memory + KB storage at load, once per process per
        tenant/project — the PRIMARY provisioning path (the stores' lazy `_prepare`
        stays as an idempotent `_ready`-guarded fallback for offline / a missed load).

        Bundle-level, NOT per-agent: both tables carry `agent_id` as a COLUMN, so a
        single ensure of the ONE `kdcube_{tenant}_{project}` schema + the two
        bundle-prefixed tables (`ported_langgraph_agents_memories`,
        `ported_langgraph_agents_kb`) covers both agents. All DDL is
        `IF NOT EXISTS` (safe across racing workers/replicas) and never
        `CREATE EXTENSION` — the platform PostgresSetup job owns the `vector`
        extension. Offline (no pool) skips cleanly; failures are non-fatal (the lazy
        fallback still self-heals on first turn)."""
        await super().on_bundle_load(**kwargs)
        pool = self.pg_pool or kwargs.get("pg_pool")
        if pool is None:
            LOGGER.info(
                "[ported-langgraph] on_bundle_load: no pg_pool available; skipping schema provisioning"
            )
            return
        ident = self.runtime_identity()
        scope = StorageScope(
            tenant=ident.get("tenant") or "default",
            project=ident.get("project") or "default",
            bundle_id=self._named_services_bundle_id(),
            agent_id="",  # bundle-level: the tables are shared, rows scoped by columns
        )
        schema = schema_for_scope(scope.tenant, scope.project)
        embed_dim = get_solution_config().embed_dim
        try:
            await ensure_memory_tables(pool, schema, embed_dim)
            await ensure_kb_tables(pool, schema, embed_dim)
        except Exception:
            LOGGER.exception(
                "[ported-langgraph] on_bundle_load: failed to ensure memory/KB schema "
                "tenant=%s project=%s",
                scope.tenant,
                scope.project,
            )
            return

    # ── storage edge (shared by both agents; one tenant/project schema) ───────

    async def _hosted_database_url(self, own_database_url: str, schema: str) -> str:
        """Resolve the psycopg DSN the checkpointer (+ optional `langgraph_store`)
        connect through. Hosted (self.pg_pool present) -> KDCube's shared Postgres
        with the per-tenant/project schema on the search_path (created idempotently
        first, via the asyncpg pool — no psycopg). Standalone -> the agent's own
        DATABASE_URL. Fail-open: a shared-DB failure degrades to the own/offline
        path (the checkpointer then falls back to MemorySaver)."""
        target = resolve_solution_pg(getattr(self, "pg_pool", None), own_database_url, schema)
        if not target.hosted:
            return own_database_url
        try:
            await self._ensure_schema_via_pool(target.schema)
        except Exception:
            return own_database_url
        return target.database_url

    async def _ensure_schema_via_pool(self, schema: str) -> None:
        """``CREATE SCHEMA IF NOT EXISTS`` on KDCube's SHARED asyncpg pool.

        Run before the checkpointer opens (its `setup()` creates tables in the
        search_path'd schema, which must already exist). Uses the pool — the same
        asyncpg pool memory/KB use — so schema creation needs no psycopg."""
        async with self.pg_pool.acquire() as con:
            await con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

    async def _open_checkpointer(self, agent_id: str, database_url: str):
        """Return the LangGraph Postgres checkpointer for one agent, opening it ONCE
        and reusing it across every per-turn graph rebuild.

        The graph is rebuilt each turn (scaled serving), but the checkpointer is a
        CONNECTION — infra, like `pg_pool` — so it is opened lazily per agent and
        held (the cm is retained so the async saver isn't GC'd), not reopened per
        turn (which would leak a psycopg connection every turn). Concurrency note:
        `from_conn_string` opens a single connection shared by concurrent turns of
        this agent — same as the prior cached-graph design; a pooled
        `AsyncConnectionPool` is a clean follow-up if concurrency grows.

        Falls back to an in-memory saver when no DB is reachable (offline
        degradation). The fallback is LOUD: an in-memory saver keeps NO cross-turn
        history — every conversation restarts empty on the next process, and a
        reloaded conversation has no memory. That is a silent, confusing failure if
        it happens by accident (the usual cause: the declared deps
        `langgraph-checkpoint-postgres` + `psycopg[binary]` v3 are not installed in
        the runtime venv, or the DSN is unreachable). We log WHY at WARNING so it is
        visible, not mysterious. The in-memory fallback is NOT cached (each rebuild
        retries Postgres, so a transient outage self-heals on the next turn)."""
        existing = self._checkpointers.get(agent_id)
        if existing is not None:
            return existing

        lock = self._checkpointer_locks.get(agent_id)
        if lock is None:
            # Safe to create lazily under asyncio: no await between get and set.
            lock = asyncio.Lock()
            self._checkpointer_locks[agent_id] = lock
        async with lock:
            existing = self._checkpointers.get(agent_id)
            if existing is not None:
                return existing
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                cm = AsyncPostgresSaver.from_conn_string(database_url)
                checkpointer = await cm.__aenter__()
                await checkpointer.setup()
                self._checkpointer_cms[agent_id] = cm
                self._checkpointers[agent_id] = checkpointer
                LOGGER.info(
                    "[ported-langgraph] %s checkpointer: durable Postgres (cross-turn history persists)",
                    agent_id,
                )
                return checkpointer
            except Exception:
                # NOT cached: the next per-turn rebuild retries Postgres, so a
                # transient outage self-heals on a later turn.
                from langgraph.checkpoint.memory import MemorySaver

                LOGGER.warning(
                    "[ported-langgraph] %s checkpointer FELL BACK to in-memory — cross-turn "
                    "history will NOT persist (lost on restart, absent for reloaded "
                    "conversations). Install `langgraph-checkpoint-postgres` + `psycopg[binary]` "
                    "(v3) in the runtime venv and ensure the DSN is reachable. Cause:",
                    agent_id,
                    exc_info=True,
                )
                return MemorySaver()

    # ── lg-solution model/embedding edge ──────────────────────────────────────

    def _solution_embedding_service(self):
        """Resolve the economics-guarded embedding facade for the CURRENT turn.
        Per-call (not cached): `search_model_service` reads the live comm_context,
        so each embed is guarded for the turn's own identity. Degrades to the raw
        accounted service when economics is off, or None with no model service."""
        return self.search_model_service(flow=SOLUTION_RETRIEVAL_FLOW)

    # ── lg-react model edge ────────────────────────────────────────────────

    def _build_prebuilt_model(self, config: Any):
        """lg-react's create_react `model`: HOSTED (a platform model service is
        present) -> an accounted `KDCubeChatModel` bound to the answer role. The
        platform service provides the model/key, so the standalone config's own
        `offline` flag (which only tracks the STANDALONE OpenAI key) is irrelevant
        here — mirror lg-solution's `llm.chat_model()`, which gates on
        `models_service` alone. Only with NO model service (truly standalone) ->
        the solution's deterministic stub, so the agent degrades like the CLI.

        The output budget (NOT the adapter's small default) bounds the answer
        model: this agent carries whole payloads in tool arguments (`run_python`
        code), and a ceiling below one complete tool call cuts the response
        mid-call — the truncated args fail validation and the model retries
        into the same wall until the recursion limit. Hosted, the budget is a
        DESCRIPTOR property (`_agent_max_tokens`); the standalone config's value
        is only the standalone fallback."""
        if getattr(self, "models_service", None) is not None:
            return KDCubeChatModel(
                models_service=self.models_service, role=PREBUILT_ANSWER_ROLE,
                temperature=0.2,
                max_tokens=self._agent_max_tokens(
                    PREBUILT_AGENT_ID, int(getattr(config, "max_tokens", 0) or 16384)
                ),
            )
        return StubChatModel()

    def _agent_max_tokens(self, agent_id: str, fallback: int) -> int:
        """One agent's answer-model OUTPUT budget, from the app descriptor:
        `surfaces.as_consumer.agents.<agent>.model.max_tokens`. KDCube apps are
        configured through descriptor properties (`bundle_prop`) — process env
        vars are not a configuration surface here; the standalone config's env
        knob is the STANDALONE idiom and serves only as the hosted fallback."""
        try:
            raw = self.bundle_prop(
                f"surfaces.as_consumer.agents.{agent_id}.model.max_tokens", None
            )
            value = int(raw or 0)
        except Exception:
            value = 0
        return value if value > 0 else int(fallback)

    def _build_prebuilt_summary_model(self, config: Any):
        """lg-react's compaction summary model: HOSTED -> an accounted
        `KDCubeChatModel` on the DISTINCT `lg-react.summary` role (so LangMem's
        SummarizationNode folds old turns on a separately-billed, non-streamed
        role). None with no model service (offline) -> the pre_model_hook falls back
        to trim, so a turn still runs without summarization."""
        del config
        if getattr(self, "models_service", None) is not None:
            return KDCubeChatModel(
                models_service=self.models_service, role=PREBUILT_SUMMARY_ROLE, temperature=0.2
            )
        return None

    # ── graph build (per turn) ────────────────────────────────────────────────

    async def _build_graph(
        self,
        agent_id: str,
        *,
        disabled: Optional[Mapping[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the agent's graph FOR THIS TURN — never cached in-process.

        KDCube is distributed (turns hop workers/machines), so a process-cached
        graph would neither survive the hop nor reflect this conversation's saved state
        (model pick, tool selection). Each agent builds with its own deps + the
        SHARED checkpointer connection (opened once per agent, reused — see
        `_open_checkpointer`), storing into the shared tenant/project schema scoped
        by its agent_id column. Rebuilding is cheap: the checkpointer is reused, and
        the rest is pure in-memory graph compilation. `disabled` is this conversation's
        saved capability opt-outs (tools, MCP servers, namespaces) — the per-turn build
        is exactly what lets them narrow what the agent gets (lg-react); an agent with
        nothing pickable ignores it."""
        return await AGENTS[agent_id].build_graph(self, disabled=disabled, state=state)

    # ── the turn (the dispatcher) ─────────────────────────────────────────────

    async def execute_core(
        self,
        *,
        state: Dict[str, Any],
        thread_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Resolve the ACTIVE agent from the turn state; unknown/blank -> default
        # (the shared foreign-runtime dispatch rule).
        spec = resolve_agent_spec(AGENTS, state.get("agent_id"), DEFAULT_AGENT_ID)
        agent_id = spec.agent_id

        # This (user, conversation)'s opt-outs from the capabilities widget —
        # the WHOLE deny map in ONE store read (tools, MCP servers, namespaces),
        # resolved BEFORE the build so it narrows what the agent gets (admin
        # ceiling ∩ user-enabled). Fails open to {} (everything the admin allows
        # stays in force).
        disabled = await resolve_turn_selection_disabled(self, state, agent_id)
        disabled_tools = disabled_category(disabled, DISABLED_TOOLS)

        # Built fresh every turn — no in-process graph cache (scaled serving).
        graph = await self._build_graph(agent_id, disabled=disabled, state=state)

        # (pending-lane fold) dispatch names one wake occurrence. Read the whole
        # still-pending lane once so this turn also sees attachment/context
        # siblings and messages queued while the previous turn ran. The shared
        # seam is read-only and records exact ids for door finalization.
        # Fail-open: the dispatched events stand.
        state["external_events"] = await fold_turn_external_events(self, state)

        # (state mapping) pull the user's question out of the platform external
        # events; each agent shapes it into its own inputs. Nothing else is
        # read for the model automatically: arriving files reach it as
        # METADATA + LINKS in the turn frame below, and the model chooses the
        # door — read_file to view, pull_files + run_python to process.
        question = external_events_text(state.get("external_events") or [])
        # A turn folds every message that queued while the previous one ran, so
        # there can be SEVERAL — and a later one often revises an earlier one.
        # `question` stays the first (it names the conversation); what the model
        # reads is all of them, in the order they were sent.
        user_texts = external_events_texts(state.get("external_events") or [])
        if len(user_texts) > 1:
            question_for_model = (
                "\n".join(f"{n}. {text}" for n, text in enumerate(user_texts, start=1))
                + "\n\nThey were all sent before you started, and a later one may "
                "revise an earlier one. Read them together and answer once."
            )
        else:
            question_for_model = question
        attachments: list = []

        # (code execution) build the per-turn code-exec scope for the ACTIVE agent —
        # BEFORE the inputs, because the exec workspace also receives this turn's
        # attachments. When `tools.code_exec.enabled` is on, this stands up an
        # isolated-runtime + hosting edge so a model-called `run_python` runs code
        # and hosts the files it produces into conversation storage (exactly like a
        # user attachment). Disabled / offline yields an inert context and the turn
        # runs unchanged.
        try:
            code_exec_ctx = build_code_exec_context(self, state, agent_id)
        except Exception:
            LOGGER.warning("[ported-langgraph] code-exec context build failed", exc_info=True)
            code_exec_ctx = None

        # (turn workspace) account for this turn's arriving files — metadata as
        # received + a durable conv:fi: link each — and frame the model's turn
        # text like React frames its timeline: [Turn start <id>] (the boundary
        # + the empty-fresh-workspace rule), [User message], [Files arriving
        # this turn]. A file is never silently dropped: with no workspace
        # tools the frame says exactly why contents are out of reach.
        try:
            workspace = await prepare_turn_workspace(
                code_exec_ctx,
                state.get("external_events") or [],
                exec_tool_bound=run_python_bound(agent_tool_connections(self, agent_id), disabled_tools),
            )
        except Exception:
            LOGGER.warning("[ported-langgraph] turn workspace preparation failed", exc_info=True)
            from .platform.turn_workspace import TurnWorkspace
            workspace = TurnWorkspace(live=False)
        framed_question = frame_turn_input(question_for_model, workspace)

        # Whole-turn summary at the boundary: one line tells the story of what this
        # turn was asked to do (cheap, non-fatal).
        LOGGER.info(
            "[ported-langgraph] TURN start agent=%s conversation=%s question_len=%d attachments=%d "
            "workspace_live=%s workspace_files=%d",
            agent_id, thread_id, len(question or ""), len(attachments or []),
            workspace.live, len(workspace.files),
        )

        # (isolation) map platform identity + agent_id onto the agent's per-user +
        # per-conversation keys — the gate that keeps the two agents' state apart.
        ident = turn_identity(state, agent_id=agent_id, fallback_thread_id=thread_id)
        inputs, run_config = spec.build_inputs(framed_question, ident, attachments)

        # First turn only: name the conversation from the USER'S QUESTION and emit it
        # to the client header BEFORE the agent runs, so the title appears even if the
        # agent's turn later errors — the title never depends on a successful answer.
        # Stashed on `state` for the turn recorder to persist. One small accounted
        # title call, first turn only. (The ORIGINAL question — the staged-files
        # note is model plumbing, not the user's ask.)
        await self._finalize_conversation_title(
            state=state, conversation_id=thread_id, question=question,
            title_role=spec.role,
        )

        # (streaming) run the agent's own astream_events loop through ITS stream
        # adapter, redirected at the chat component via comm_ctx. Returns the
        # answer in the shape the Telegram renderer reads. The graph runs INSIDE the
        # code-exec scope so `run_python`'s `host_files` resolves during the run.
        # (repair) Whatever interrupted the LAST run of this thread — a stop, a
        # timeout kill, a worker that died — may have left a tool call the model
        # made and no result for it, which the provider refuses to accept on the
        # NEXT request (`tool_use` without `tool_result`). Left alone it wedges
        # the whole conversation, not one turn, so every turn heals the thread it
        # is about to send before it sends it.
        await repair_unanswered_tool_calls(graph, run_config)

        async def _run_turn() -> Dict[str, Any]:
            async with code_exec_scope(code_exec_ctx):
                answer = await spec.stream(graph, inputs, run_config)
            return {"answer": answer, "final_answer": answer}

        # Capabilities model pick for the ACTIVE agent (per user, per conversation).
        # Bind the resolved answer-role override onto the bundle call context so the
        # model router overlays it on this agent's `get_client("<agent>.answer")`
        # for this turn only. Empty overlay => the router's configured default
        # routes. Fail-open by construction. Both ingresses honor the pick.
        role_models = await resolve_turn_role_models(self, state, agent_id)

        # Time the turn so its elapsed_ms lands in the recorded-events artifact the
        # reload reader replays (mirrors BaseWorkflow.report_timings). Captured here,
        # emitted after the run.
        _turn_started_ms = _now_ms()
        _turn_t0 = time.perf_counter()

        # Wrap the run so a Telegram-originated turn is delivered from the processor
        # side; a browser turn passes through unchanged.
        # (stop) The lane is watched while the graph streams, and a steer cancels
        # the run. This agent does not consume live events into its own loop —
        # that loop is LangGraph's and folding into it mid-run would contradict
        # the graph's own iteration — so the watcher is read-only and everything
        # it saw is still PENDING when the run is cancelled. Finalization spends
        # a bare steer as the stop boundary and wakes only for later intent;
        # retained text is folded when a later turn starts. The checkpointer
        # holds the last completed node, so a stop loses nothing that finished.
        async def _run_turn_stoppable() -> Dict[str, Any]:
            outcome = await run_until_stopped(self, state, _run_turn)
            if not outcome.stopped:
                return outcome.result
            LOGGER.info(
                "[ported-langgraph] TURN stopped by the user agent=%s conversation=%s "
                "pending_events=%d",
                agent_id, thread_id, len(outcome.arrived),
            )
            # A cancellation can land between the model asking for a tool and
            # the result arriving, and the checkpoint keeps the half it has.
            # Answered here so the thread is sendable at rest, truthfully: the
            # next turn reads that the tool did not run rather than inventing
            # what it would have returned.
            await repair_unanswered_tool_calls(graph, run_config)
            # Said in band, in plain language: the person sees why the stream
            # ended, and the next turn reads the same sentence in its history
            # instead of inferring that the agent fell silent.
            return {"answer": STOPPED_ANSWER, "final_answer": STOPPED_ANSWER}

        with bind_current_bundle_call_context_patch({"role_models": role_models}):
            result = await telegram_ingress.run_turn_with_delivery(
                self, runner=_run_turn_stoppable,
            )

        # The platform's canonical final answer (what the framework-neutral turn
        # recorder persists for reload). The conversation title was already emitted
        # before the agent ran (see above), so it is unaffected by a turn error.
        state["final_answer"] = (result or {}).get("final_answer") or (result or {}).get("answer") or ""

        # Surface the files this turn hosted (run_python outputs) so the framework-
        # neutral turn recorder persists their refs — reload re-surfaces the files and
        # a later turn can pull them by ref. Empty when code exec was off/unused.
        if code_exec_ctx is not None and getattr(code_exec_ctx, "hosted_files", None):
            state["hosted_files"] = list(code_exec_ctx.hosted_files)

        # Turn timing: emit the SAME `chat.turn.summary` event React authors, so the
        # turn's elapsed time is recorded (and, via the events artifact, restored on
        # reload) alongside the economics door's `accounting.usage` $ badge. The cost
        # itself is emitted by the door's post-run accounting; both ride the recorded
        # comm and are persisted together by `_save_events_artifact` (post_run_hook).
        _turn_total_ms = int((time.perf_counter() - _turn_t0) * 1000)
        await self._emit_turn_timing(started_ms=_turn_started_ms, total_ms=_turn_total_ms)

        LOGGER.info(
            "[ported-langgraph] TURN done agent=%s conversation=%s answer_len=%d elapsed_ms=%d",
            agent_id, thread_id, len(state.get("final_answer") or ""), _turn_total_ms,
        )
        return state

    async def _emit_turn_timing(self, *, started_ms: int, total_ms: int) -> None:
        """Emit the turn-timing summary event (mirrors ``BaseWorkflow.report_timings``).

        A run-to-completion turn has no React timeline to author ``chat.turn.summary``,
        so the turn's elapsed time never reaches the recorded-events artifact the
        conversation reload replays — a reloaded turn would lose its duration. Emitting
        the SAME event here, on the recorded comm (the one the economics door's
        ``accounting.usage`` cost badge also rides), threads ``elapsed_ms`` into that
        artifact so reload restores the time exactly like React. Field name/shape match
        BaseWorkflow so the same reload reader surfaces it. Best-effort: a timing-emit
        failure never affects the turn. Delegates to the shared foreign-runtime
        seam."""
        await emit_turn_timing(self, started_ms=started_ms, total_ms=total_ms)

    # ── per-turn economics persistence (reload) ───────────────────────────────

    async def post_run_hook(
        self,
        *,
        state: Dict[str, Any],
        result: Dict[str, Any],
        econ_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist this turn's recorded chat events as the ``conv.artifacts.events``
        artifact the conversation reload replays.

        The economics door emits the turn's cost live (``accounting.usage`` — the $
        badge) and this app emits the turn's elapsed time (``chat.turn.summary``);
        both are recorded on the turn's comm (recording started by the base
        ``pre_run_hook``). But a run-to-completion app writes no React timeline, so
        without this the recorded events are never saved — the turn streamed cost +
        time live yet a reloaded turn showed neither. Persisting the SAME events
        artifact the React/workspace path persists (its ``post_run_hook`` does exactly
        this) makes reload restore both, via the shared SDK mechanism (no hand-rolled
        economics format). Best-effort by construction: a persistence failure never
        affects the turn.

        The saved-artifact user is threaded from the economics/authority projection
        when the raw ``user`` state key is empty (this app carries the user on the
        authority projection — mirrors ``_conversation_is_new``'s user resolution), so
        the artifact is scoped to the same (user, conversation) the reload reads.
        The persistence orchestration is the shared foreign-runtime seam's."""
        await super().post_run_hook(state=state, result=result, econ_ctx=econ_ctx or {})
        await record_foreign_runtime_turn_log(self, state, result, econ_ctx=econ_ctx or {})
        await persist_turn_artifacts(self, state, result)

    # ── first-turn conversation title ─────────────────────────────────────────

    async def _finalize_conversation_title(
        self, *, state: Dict[str, Any], conversation_id: str, question: str,
        answer: Optional[str] = None, title_role: Optional[str] = None,
    ) -> None:
        """Propose a short conversation title on the FIRST turn, stream it to the
        client, and stash it on ``state`` for the turn recorder to persist.

        Generated from the user's QUESTION (an optional ``answer`` adds signal when
        available) so it can run BEFORE the agent — the title then appears even if
        the agent's turn later errors, and never depends on a successful answer. The
        first-turn signal is framework-neutral: a new conversation has no prior
        recorded turn log. Uses the reusable SDK utility directly (no ctx_browser /
        no thinking stream — this is a run-to-completion bundle). Fail-open by
        construction: any failure leaves the turn untouched. The chain itself is
        the shared foreign-runtime seam's (`finalize_conversation_title`)."""
        # The seam resolves the title utilities (`generate_conversation_title`,
        # `emit_conversation_title_event`) from their home module at CALL time;
        # this module's top-level imports of those names are the offline tests'
        # injection point, so thread any rebound one through for this call.
        # In production nothing is rebound and this is a pure delegation.
        from kdcube_ai_app.apps.chat.sdk.tools.backends.summary import (
            conversation_title as _title_impl,
        )

        overrides = {
            name: fn
            for name, fn in (
                ("generate_conversation_title", generate_conversation_title),
                ("emit_conversation_title_event", emit_conversation_title_event),
            )
            if getattr(_title_impl, name) is not fn
        }
        originals = {name: getattr(_title_impl, name) for name in overrides}
        for name, fn in overrides.items():
            setattr(_title_impl, name, fn)
        try:
            await finalize_conversation_title(
                self, state,
                conversation_id=conversation_id, question=question,
                answer=answer, title_role=title_role,
            )
        finally:
            for name, fn in originals.items():
                setattr(_title_impl, name, fn)

    async def _conversation_is_new(self, *, state: Dict[str, Any], conversation_id: str) -> bool:
        """A conversation is new when it has no prior recorded turn (the current
        turn's log is written after ``execute_core``). Read the platform conversation
        record — the same store the conversation list reads — so the signal matches
        what the user sees. Fail-safe to "not new" (skip the title) on any error.
        Delegates to the shared foreign-runtime seam (which reads the SAME user
        the economics door records the turn log under — `economics_user` first,
        raw actor keys as fallbacks)."""
        return await conversation_is_new(self, state, conversation_id=conversation_id)

    # ── scene chat widgets (one per agent) ────────────────────────────────────
    # The scene mounts two chat tiles as iframes at `widgets/chat_lg_solution`
    # and `widgets/chat_lg_react`. A widget is only reachable once it is declared
    # in the bundle MANIFEST — that declaration is the `@ui_widget` decorator
    # here, NOT the `ui.widgets` build config (which only builds the assets). Each
    # method returns a tiny static fallback; the real UI is served from the built
    # `sdk://solutions/chat/ui/widget` assets (per that alias's `ui.widgets`
    # entry), each build agent-bound to its own agent via VITE_CHAT_AGENT_ID.

    @api(
        alias="chat_lg_solution_widget",
        route="operations",
        **_api_visibility("chat_lg_solution_widget"),
    )
    @ui_widget(
        icon={
            "tailwind": "heroicons-outline:chat-bubble-left-right",
            "lucide": "MessagesSquare",
        },
        alias="chat_lg_solution",
        **_widget_visibility("chat_lg_solution"),
    )
    def chat_lg_solution_widget(self, **kwargs):
        del kwargs
        return [
            "<div style=\"font-family:system-ui,sans-serif;padding:16px\">"
            "lg-solution chat is served from sdk://solutions/chat/ui/widget after build."
            "</div>"
        ]

    @api(
        alias="chat_lg_react_widget",
        route="operations",
        **_api_visibility("chat_lg_react_widget"),
    )
    @ui_widget(
        icon={
            "tailwind": "heroicons-outline:chat-bubble-oval-left-ellipsis",
            "lucide": "MessageCircleMore",
        },
        alias="chat_lg_react",
        **_widget_visibility("chat_lg_react"),
    )
    def chat_lg_react_widget(self, **kwargs):
        del kwargs
        return [
            "<div style=\"font-family:system-ui,sans-serif;padding:16px\">"
            "lg-react chat is served from sdk://solutions/chat/ui/widget after build."
            "</div>"
        ]

    @api(
        method="POST",
        alias="telegram_webhook",
        route="public",
    )
    async def telegram_webhook(self, request: Any = None, **update) -> Dict[str, Any]:
        """The shared ingress: the Telegram Bot API webhook.

        `route="public"` exposes the route at the proc layer without platform auth
        — the trust boundary is the Telegram webhook secret, which the SDK verifies
        before any work. The SDK resolves the Telegram user into a `telegram_<id>`
        platform identity and drives the SAME turn (`execute_core`) for the DEFAULT
        agent, then renders the answer back over the Bot API. Thin: routes only."""
        return await telegram_ingress.handle_webhook(self, request=request, **update)

    @api(
        method="POST",
        alias="scene_surface_config",
        route="operations",
        **_api_visibility("scene_surface_config"),
    )
    async def scene_surface_config(self, **kwargs: Any) -> Dict[str, Any]:
        """The scene's descriptor-declared composition
        (`surfaces.as_consumer.ui.scene`): components merged over the scene's
        code defaults by alias — this is where the Connection Hub component is
        configured (bundle_id/widget_alias/target_surfaces), not hardcoded."""
        del kwargs
        scene_config = self.bundle_prop("surfaces.as_consumer.ui.scene", {}) or {}
        if not isinstance(scene_config, Mapping):
            scene_config = {}
        components = scene_config.get("components")
        if not isinstance(components, Mapping):
            components = {}
        return {"ok": True, "components": dict(components)}

    @api(
        method="POST",
        alias="scene_object_action",
        route="operations",
        **_api_visibility("scene_object_action"),
    )
    async def scene_object_action(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Serve the chat component's generic object-action operation for THIS bundle's
        hosted conversation FILES. The chat file card calls `scene_object_action` for
        any object ref; this bundle has no canvas/scene board — only conversation files
        a turn produced (e.g. code-exec output). For a `conv:fi:` file ref, delegate to
        the platform file resolver: it reads the bytes from conversation storage and
        returns a `download_url` the client fetches. Without this operation the file is
        hosted and shown but the Download action has no endpoint to serve it.

        Identity: the file is stored under the turn owner (svc-first — see
        `resolve_request_identity`), so the download resolves under that same owner;
        tenant/project come from the bundle runtime."""
        from kdcube_ai_app.apps.chat.sdk.runtime.harness.events.resolver import resolve_event_ref_action
        from kdcube_ai_app.apps.chat.sdk.event_identity import resolve_request_identity

        payload: Dict[str, Any] = (
            {str(k): v for k, v in data.items()} if isinstance(data, dict)
            else {
                str(k): v for k, v in kwargs.items()
                if k not in {"request", "alias", "route", "endpoint_alias"} and v is not None
            }
        )
        ident = self.runtime_identity()
        owner = ""
        try:
            owner = str(resolve_request_identity(self.comm).get("owner") or "").strip()
        except Exception:
            owner = ""
        user_id = str(
            payload.get("user_id") or owner or getattr(self.comm, "user_id", "") or ""
        ).strip()
        result = await resolve_event_ref_action(
            payload,
            tenant=ident.get("tenant") or "default",
            project=ident.get("project") or "default",
            user_id=user_id,
            storage_path=str(getattr(self.settings, "STORAGE_PATH", "")),
            require_embedded_conversation=True,
        )
        try:
            LOGGER.info(
                "[ported-langgraph] scene_object_action action=%s ref=%s user=%s ok=%s error=%s",
                payload.get("action"),
                payload.get("ref") or payload.get("object_ref") or payload.get("event_ref"),
                user_id, (result or {}).get("ok"), (result or {}).get("error"),
            )
        except Exception:
            pass
        return result
