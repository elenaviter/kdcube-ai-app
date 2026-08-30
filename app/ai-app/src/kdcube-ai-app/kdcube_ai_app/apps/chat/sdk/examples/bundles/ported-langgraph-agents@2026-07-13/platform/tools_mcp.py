# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── tools_mcp.py ── the "tools, both ways" seam (thin over the SDK) ──
#
# The preserved agent binds PLAIN LangChain tools (solution/tools.py) — "bring your
# own tools", external to the host and, running no accounted model calls, unmetered.
# This module adds the SECOND way: bind a KDCube-served MCP endpoint's tools as
# LangChain tools.
#
# The mechanism is now SHARED SDK, reused by any hosted LangGraph/LangChain agent:
#   - `solutions/connections/delegated_mcp.resolve_mcp_server_map` — framework-neutral:
#     turn the agent's `kind: mcp` connections into an MCP server map, minting a
#     per-user DELEGATED bearer for any connection marked `delegated: true` (the same
#     `@mcp`-surface auth platform bundles use) and injecting it; static connections
#     keep their declared headers.
#   - `frameworks/langchain/mcp.load_mcp_tools_from_server_map` — bind that map as
#     LangChain tools through KDCube's MCP SDK v2 adapter.
#
#   - `solutions/foreign_runtime/mcp_bridge.narrow_mcp_connections` — the per-turn
#     narrowing every wrapped runtime shares: the capabilities picker's
#     `disabled.mcp` deny map, keyed by SERVER ID, applied BEFORE resolution.
#
# This bundle file is the thin adapter: pass the agent's connection list + this
# turn's user, get LangChain tools.
#
# ACCOUNTING (the honest rule — "marked = counted"): binding a tool via MCP does not
# by itself make it accounted; a tool whose KDCube-side implementation runs a marked
# model call IS metered, a plain lookup is not.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_mcp import (
    resolve_mcp_server_map,
    delegated_client_id_for_agent,
    is_delegated_connection,
    connection_resource,
    DROP_CONSENT_PENDING,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.mcp_consent import (
    MCPConsentRequired,
    mcp_consent_from_denial,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.mcp_bridge import (
    connection_server_id,
    load_mcp_server_instructions_safe,
    narrow_mcp_connections,
)
from kdcube_ai_app.apps.chat.sdk.frameworks.langchain.mcp import (
    load_mcp_tools_from_server_map,
    load_error_looks_like_denial,
    mcp_adapters_available,  # re-exported for callers/tests
)

logger = logging.getLogger(__name__)

__all__ = [
    "mcp_connections",
    "narrow_bound_mcp_tools",
    "load_mcp_tools_for_connections",
    "consent_request_tools",
    "mcp_adapters_available",
]


def consent_request_tools(
    consents: List[MCPConsentRequired],
    *,
    announce: Any,
) -> List[Any]:
    """One consent-gated STUB tool per pending delegated connection.

    Consent is demand-driven per tool: a turn's build cannot know which
    capabilities the turn will use, so a pending connection must NOT raise a
    turn-start demand. Instead it binds a stub carrying the connection's name
    and claims; when the MODEL decides the user's request needs that
    capability, calling the stub raises exactly that connection's consent
    demand in chat (via ``announce``) and returns the agent-explainable consent
    result — the same attempt-time semantics connected-account tools have.
    Once the user grants, the next build binds the real tools and the stub
    disappears. Returns ``[]`` when LangChain is unavailable."""
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except Exception:  # pragma: no cover - langchain-less environments
        return []

    class _ConsentRequestArgs(BaseModel):
        reason: str = Field(
            default="",
            description="One line on what the user asked for that needs this capability.",
        )

    tools: List[Any] = []
    for c in consents:
        alias = str((getattr(c, "consent", {}) or {}).get("tool_name") or "").strip() or "restricted_capability"
        claims = ", ".join(getattr(c, "claims", []) or []) or "the required access"

        async def _request(reason: str = "", _consent: MCPConsentRequired = c) -> Dict[str, Any]:
            del reason
            try:
                await announce(_consent)
            except Exception:
                logger.info("consent stub: announce failed (non-fatal)", exc_info=True)
            return _consent.to_tool_result()

        tools.append(StructuredTool.from_function(
            coroutine=_request,
            name=alias,
            description=(
                f"{alias}: this capability needs the user's consent to {claims}. "
                "Call it when the user's request needs this capability — the call "
                "raises a consent request in chat for the user to approve and "
                "returns the consent status. After the user grants, the real "
                f"{alias} tools become available on the next turn."
            ),
            args_schema=_ConsentRequestArgs,
        ))
    return tools


def mcp_connections(
    connections: List[Dict[str, Any]],
    disabled_map: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """The `kind: mcp` entries of the agent's declared tool-connection list, minus
    the servers the user turned OFF whole this turn — the same admin-ceiling ∩
    user-enabled narrowing the plain/code-exec tools get, so MCP servers are
    governed too ("which agent").

    ``disabled_map`` is the capabilities picker's ``disabled.mcp`` category:
    ``{server_id: true | [tool names]}``. The KEY IS THE SERVER ID — the id the
    picker's catalog rows carry, which a connection may declare separately from
    its model-facing ``alias``. The narrowing itself is the shared seam's
    (``narrow_mcp_connections``), so this app and every other wrapped runtime drop
    a server on exactly the same rule, BEFORE resolution: a server turned off is
    never dialled, no grant token is read for it, and it can raise no consent card.

    A LIST value is a partial denial: the connection stays and its surviving tools
    are named after the bind (``narrow_bound_mcp_tools``)."""
    conns = [
        c for c in connections or []
        if isinstance(c, dict) and str(c.get("kind") or "").strip().lower() == "mcp"
    ]
    return narrow_mcp_connections(conns, disabled_map)


def narrow_bound_mcp_tools(
    tools: List[Any],
    disabled_map: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    """The bound MCP tools minus the individual ones the user turned off.

    The second half of the ``disabled.mcp`` deny map: a LIST value under a server
    id names the tools opted out of a server that stays on. Whole-server denials
    never reach here (the connection is dropped before resolution) — this covers
    exactly the per-tool rows the picker offers under a server.

    Each tool bound from a server map carries its origin in
    ``metadata["mcp_server_id"]``, which is what makes the match server-scoped:
    two servers publishing the same tool name are narrowed independently. A tool
    with no such metadata is never dropped."""
    disabled = disabled_map or {}
    denied = {
        str(server_id): {str(name).strip() for name in entry if str(name or "").strip()}
        for server_id, entry in disabled.items()
        if isinstance(entry, (list, tuple))
    }
    if not denied:
        return list(tools or [])
    kept: List[Any] = []
    for tool in tools or []:
        server_id = str((getattr(tool, "metadata", None) or {}).get("mcp_server_id") or "")
        if str(getattr(tool, "name", "") or "") in denied.get(server_id, ()):
            continue
        kept.append(tool)
    return kept


async def load_mcp_tools_for_connections(
    connections: List[Dict[str, Any]],
    *,
    user_sub: Optional[str] = None,
    disabled_map: Optional[Mapping[str, Any]] = None,
    application: str = "",
    agent_id: str = "",
    bearer_provider: Optional[Any] = None,
    instructions_sink: Optional[Dict[str, str]] = None,
) -> tuple[List[Any], List[MCPConsentRequired]]:
    """Bind the agent's declared, user-enabled `kind: mcp` connections as LangChain
    tools for THIS turn's user, AS this agent.

    ``disabled_map`` is the picker's ``disabled.mcp`` category (keyed by SERVER
    ID): a server turned off whole is dropped before resolution, and a server
    kept with individual tools turned off binds only its survivors.

    The agent is a "Delegated By KDCube" entity keyed by `application` + `agent_id`,
    so consent is per-agent. When ``bearer_provider`` is supplied (the recommended
    path), a delegated connection uses the token the user's per-agent grant already
    bound — so the KDCube `@mcp` guard passes; a connection with NO consented grant
    is dropped and surfaces as a consent demand. Without a provider the resolver
    falls back to a fresh mint (unbound → the guard denies until consent exists),
    which still yields the same consent demand.

    Returns ``(tools, consent_demands)``: when a KDCube `@mcp` load is denied for
    missing consent (a 403 at connect time), the tools are absent and a
    ``MCPConsentRequired`` is returned for each delegated connection so the caller
    can bubble it into chat and explain it to the agent. Never raises."""
    conns = mcp_connections(connections, disabled_map)
    if not conns:
        return [], []
    client_id = delegated_client_id_for_agent(application, agent_id)
    # The server map is resolved directly (not via the foreign-runtime seam's
    # `resolve_turn_mcp` wrapper): this function's contract takes an INJECTED
    # `bearer_provider` + explicit `user_sub` and no entrypoint, while the
    # wrapper derives both from an entrypoint and accepts no provider override.
    drop_sink: Dict[str, str] = {}
    server_map = await resolve_mcp_server_map(
        conns, user_sub=user_sub, client_id=client_id, bearer_provider=bearer_provider,
        drop_sink=drop_sink,
    )
    error_sink: Dict[str, Any] = {}
    # The shared loader applies chat consent+delivery post-processing to every
    # bound tool (driven by the surface's self-describing result), so this thin
    # bundle adapter carries none of that logic.
    tools = await load_mcp_tools_from_server_map(server_map, error_sink=error_sink)
    # The per-tool half of the same deny map (a server kept, some of its tools
    # turned off) — applied on the bound tools, since only the bind knows which
    # tools a server actually publishes.
    tools = narrow_bound_mcp_tools(tools, disabled_map)

    # An MCP server may publish an operating guide during protocol negotiation —
    # what MCP-native clients (e.g. Claude connectors) show their model. Tool
    # schemas do not carry it, so recover it here for the system prompt (via the
    # foreign-runtime seam's fail-open wrapper — any failure yields {}).
    # Only when tools actually loaded (a consent-denied door would just 403).
    if instructions_sink is not None and tools:
        instructions_sink.update(await load_mcp_server_instructions_safe(server_map))

    # A delegated connection the user hasn't granted THIS agent surfaces as a
    # consent demand, whichever way the block manifested:
    #   * dropped BEFORE any server contact (the consented-token path returned no
    #     bearer -> DROP_CONSENT_PENDING in drop_sink) — no transport error exists;
    #   * denied AT connect time (an unbound bearer met the @mcp guard's 403).
    server_errors = error_sink.get("_server_errors") or {}
    consents: List[MCPConsentRequired] = []
    for c in conns:
        if not is_delegated_connection(c):
            continue
        server_id = connection_server_id(c)
        dropped_pending = drop_sink.get(server_id) == DROP_CONSENT_PENDING
        server_error = server_errors.get(server_id)
        denied_at_load = load_error_looks_like_denial(server_error)
        if not dropped_pending and not (denied_at_load and server_id in server_map):
            continue
        claims = c.get("scopes") or c.get("claims") or []
        if isinstance(claims, str):
            claims = [claims]
        consents.append(mcp_consent_from_denial(
            {"status": 403, "reason": "authority_mismatch"},
            # The connection's declared delegated-resource id (its `resource`,
            # falling back to the url) — the SAME key the grant is created and
            # looked up under. A deployment whose configured resource is a
            # wildcard pattern declares it via `resource`, so the demand's
            # one-click grant validates against the catalog.
            resource=connection_resource(c),
            claims=claims,
            tool_name=str(c.get("alias") or c.get("name") or ""),
            agent_client_id=client_id,
        ))
    return tools, consents
