# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── platform/named_services.py ── what lg-react is told about namespaces ──
#
# lg-react reaches KDCube's named-service door through a `kind: mcp` connection
# and declares WHICH namespaces it consumes in a companion `kind: named_service`
# entry, each with the operations the administrator allows. Two texts come out
# of that declaration, and this module composes both for one turn:
#
#   1. HOW to work a namespace — the agent-neutral SDK teaching block
#      (`named_services_bridge_instructions`), transport-neutral and named with
#      the exact tools this turn actually bound when the door is among them.
#   2. WHICH namespaces are available and what may be done in each — the
#      foreign-runtime seam's roster block (`named_service_roster_block`). That
#      block is ONE text shared with every other wrapped runtime, so an agent
#      hosted here and an agent hosted on a CLI runtime read the same words.
#
# The roster is narrowed the same way the tools are: the administrator's
# declaration is the ceiling and the chatting user's pick in the capabilities
# widget subtracts from it (`disabled.named_services`). A namespace the user
# turned off is not a row, and nothing anywhere says it once existed.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.named_services import (
    named_service_door_servers,
    named_service_door_tools,
    named_service_roster_block,
    named_service_rosters,
    narrow_named_service_rosters,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "named_service_roster_rows",
    "narrow_bound_door_tools",
    "build_named_services_block",
]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().rstrip(":")


def named_service_roster_rows(
    connections: Sequence[Mapping[str, Any]] | None,
    disabled_namespaces: Optional[Mapping[str, Any]] = None,
    *,
    connected: Sequence[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``(roster rows for this turn, what the user removed)``.

    The rows are the declared `kind: named_service` namespaces with their allowed
    operations, minus the user's pick. ``connected`` adds the namespaces this
    agent reaches by another declaration (an event source, a canvas resolver, the
    app's own namespace config) — they exist for the agent, so they are rows too;
    with no operation list declared they carry none, and the roster says only
    their name.

    A namespace the user turned off whole is dropped from BOTH sources."""
    rosters = named_service_rosters(connections)
    kept, removed = narrow_named_service_rosters(rosters, disabled_namespaces)
    off = {
        _norm(ns) for ns, entry in (disabled_namespaces or {}).items() if entry is True
    } | {_norm(ns) for ns, entry in removed.items() if entry is True}
    named = {_norm(row.get("namespace")) for row in kept}
    for namespace in connected or []:
        ns = _norm(namespace)
        if not ns or ns in named or ns in off:
            continue
        kept.append({"alias": "", "namespace": ns, "operations": []})
        named.add(ns)
    return kept, removed


def narrow_bound_door_tools(
    tools: Sequence[Any] | None,
    connections: Sequence[Mapping[str, Any]] | None,
    disabled_namespaces: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    """The bound tools with the door's operations narrowed to the surviving roster.

    The door publishes one tool per operation plus the generic
    ``named_services_call``, and every namespace rides the SAME tools — the
    namespace is an argument. So this narrows what the roster no longer allows
    ANYWHERE: an operation still allowed in one surviving namespace keeps its
    tool, because that namespace needs it (the per-namespace half is the door's
    own gate to enforce; the roster block states it in words). The generic tool
    goes as soon as the pick removed anything, since it takes its operation as an
    argument and would otherwise reach what was just removed.

    Tools from other servers are never touched, and with no `kind: named_service`
    roster declared nothing is narrowed at all — an app that declares only the
    door connection has stated no ceiling to narrow against."""
    bound = list(tools or [])
    rosters = named_service_rosters(connections)
    if not bound or not rosters:
        return bound
    door_servers = {sid for sid in named_service_door_servers(connections, rosters).values() if sid}
    if not door_servers:
        return bound
    kept, removed = narrow_named_service_rosters(rosters, disabled_namespaces)
    allowed = set(named_service_door_tools(kept, narrowed=bool(removed)))
    out: List[Any] = []
    withheld: List[str] = []
    for tool in bound:
        metadata = getattr(tool, "metadata", None) or {}
        server_id = _norm(metadata.get("mcp_server_id") if isinstance(metadata, Mapping) else "")
        name = _norm(getattr(tool, "name", ""))
        if server_id and server_id in door_servers and name not in allowed:
            withheld.append(name)
            continue
        out.append(tool)
    if withheld:
        LOGGER.info(
            "[ported-langgraph] door tools withheld by the pick: %s (surviving namespaces=%s)",
            sorted(withheld), [row.get("namespace") for row in kept],
        )
    return out


async def build_named_services_block(
    *,
    bundle_props: Mapping[str, Any] | None,
    agent_id: str,
    connections: Sequence[Mapping[str, Any]] | None,
    disabled_namespaces: Optional[Mapping[str, Any]] = None,
    tool_names: Sequence[str] = (),
    redis: Any = None,
    tenant: str = "",
    project: str = "",
    pull_tool: str = "pull_files",
    read_tool: str = "read_file",
) -> Tuple[str, List[str]]:
    """``(the block for this turn's system prompt, the namespaces it names)``.

    Empty on both halves when this agent has no namespaces this turn — with
    nothing to name, the prompt carries nothing. Never raises: any failure
    reading the declaration or the published blurbs costs the block, never the
    turn."""
    try:
        from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
            connected_named_service_namespaces,
        )
        from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.instructions import (
            NAMED_SERVICES_MCP_DOOR_TOOL_NAMES,
            named_services_bridge_instructions,
        )
    except Exception:
        LOGGER.info("[ported-langgraph] named-services instructions unavailable", exc_info=True)
        return "", []

    try:
        connected = connected_named_service_namespaces(bundle_props or {}, client_id=agent_id)
    except Exception:
        LOGGER.info("[ported-langgraph] connected-namespace lookup failed", exc_info=True)
        connected = []

    try:
        rows, removed = named_service_roster_rows(
            connections, disabled_namespaces, connected=connected
        )
    except Exception:
        LOGGER.info("[ported-langgraph] namespace roster narrowing failed", exc_info=True)
        return "", []
    if not rows:
        return "", []
    namespaces = [str(row.get("namespace")) for row in rows]
    LOGGER.info(
        "[ported-langgraph] namespaces agent=%s roster=%s picked_off=%s",
        agent_id, namespaces, dict(removed),
    )

    # The published blurb per namespace — an optional detail of the roster's own
    # entry grammar, so a deployment that publishes intros says more in the same
    # shape and one that does not still renders the same block.
    intros: Mapping[str, Mapping[str, Any]] = {}
    try:
        from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.discovery import (
            RedisNamedServiceDiscovery,
            fetch_namespace_intros,
        )

        if redis is not None and str(tenant).strip() and str(project).strip():
            intros = await fetch_namespace_intros(
                RedisNamedServiceDiscovery(redis, tenant=str(tenant), project=str(project)),
                namespaces,
            ) or {}
    except Exception:
        LOGGER.info("[ported-langgraph] namespace intro fetch failed", exc_info=True)
        intros = {}

    # When the door itself is bound, teach with its EXACT tool names — a model
    # does not reliably map an operation name onto a differently named tool.
    names = {str(name or "") for name in tool_names}
    door_bound = any(name in names for name in NAMED_SERVICES_MCP_DOOR_TOOL_NAMES.values())
    try:
        teaching = named_services_bridge_instructions(
            pull_tool=pull_tool,
            read_tool=read_tool,
            operations=dict(NAMED_SERVICES_MCP_DOOR_TOOL_NAMES) if door_bound else None,
        )
    except Exception:
        LOGGER.info("[ported-langgraph] named-services teaching block failed", exc_info=True)
        teaching = ""

    roster = named_service_roster_block(rows, intros=intros)
    block = "\n\n".join(part for part in (teaching.strip(), roster) if part)
    return block, namespaces
