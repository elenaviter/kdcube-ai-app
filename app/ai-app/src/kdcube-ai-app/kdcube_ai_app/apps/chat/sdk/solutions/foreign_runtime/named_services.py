# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── named_services.py ── the declared namespace roster, narrowed per turn ──
#
# An agent that reaches KDCube's named-service door declares TWO things in its
# `surfaces.as_consumer.agents.<id>.tools` list:
#
#   - a `kind: mcp` connection to the door endpoint (the transport + the consent
#     grant), and
#   - a `kind: named_service` entry that binds no tools of its own and instead
#     names the NAMESPACES this agent consumes, each with the operations the
#     administrator allows:
#
#       - name: named_services_roster
#         kind: named_service
#         alias: named_services            # ties it to the door connection above
#         namespaces:
#           linkedin: { allowed: [provider.about, object.search, object.get] }
#           conv:     { allowed: [provider.about, object.search, object.get] }
#
# That declaration is the ADMIN CEILING and is exactly what the capabilities
# picker renders as its Services group: one row per namespace, expandable to its
# operations, each toggleable. What the user turns off comes back as the
# `disabled.named_services` deny map (`{namespace: true | [operation keys]}`).
#
# This module is the read side for a WRAPPED runtime — a runtime that has no ReAct
# tool config to narrow. It answers three questions with no I/O:
#
#   1. what did the admin declare?                 `named_service_rosters`
#   2. what survives the user's pick this turn?    `narrow_named_service_rosters`
#   3. what does the surviving set mean for the door's own MCP tools and for the
#      agent's own reading?                        `named_service_door_tools`,
#                                                  `named_service_roster_lines`
#
# HONEST LIMIT (why (3) has two halves): the door publishes GENERIC tools —
# `named_services_search`, `named_services_get`, … — and takes the namespace as an
# ARGUMENT. A tool-name permission grammar can therefore enforce the OPERATION
# half of a pick exactly (a denied operation's tool is removed), but not the
# NAMESPACE half: no tool name distinguishes `linkedin` from `conv`. So the
# surviving roster is also stated to the agent in words, and per-namespace
# enforcement stays with the door's own consumer gate, which knows the namespace
# because it reads the call.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "NAMED_SERVICE_OPERATION_TO_DOOR_TOOL",
    "DOOR_DISCOVERY_TOOLS",
    "DOOR_GENERIC_TOOL",
    "named_service_rosters",
    "named_service_door_servers",
    "narrow_named_service_rosters",
    "named_service_door_tools",
    "named_service_roster_lines",
]

# Declared operation → the door's MCP tool that performs it. The door's tool names
# are its own (`kdcube-services@1-0/surfaces/mcp/named_services.py`); the ReAct
# client tools of the same operations are named differently, which is why this map
# is not the one in `runtime/tool_config.py`.
NAMED_SERVICE_OPERATION_TO_DOOR_TOOL: Dict[str, str] = {
    "provider.about": "named_services_about",
    "provider.capabilities": "named_services_capabilities",
    "object.list": "named_services_list",
    "object.search": "named_services_search",
    "object.get": "named_services_get",
    "object.schema": "named_services_schema",
    "object.upsert": "named_services_upsert",
    "object.host_file": "named_services_host_file",
    "object.action": "named_services_action",
    "object.delete": "named_services_delete",
}

# Contract reading, always available while ANY namespace survives: the roster of
# what this connection serves, and the schema browse the door's own operating
# guide tells the model to read before calling anything.
DOOR_DISCOVERY_TOOLS: Tuple[str, ...] = (
    "named_services_list",
    "named_services_capabilities",
    "named_services_schema",
)

# The door's escape hatch: it takes the operation as an argument, so it reaches
# operations the surviving roster excludes. It is offered only when the roster is
# whole — a narrowed roster that left this in place would not be narrowed at all.
DOOR_GENERIC_TOOL = "named_services_call"


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _operations(namespace_cfg: Mapping[str, Any]) -> List[str]:
    raw = (
        namespace_cfg.get("allowed")
        or namespace_cfg.get("allowed_operations")
        or namespace_cfg.get("operations")
        or []
    )
    if isinstance(raw, str):
        raw = [raw]
    out: List[str] = []
    for item in raw:
        text = _norm(item)
        if text and text not in out:
            out.append(text)
    return out


def named_service_rosters(connections: Sequence[Mapping[str, Any]] | None) -> List[Dict[str, Any]]:
    """The agent's declared namespace roster — one row per namespace.

    ``[{"alias", "namespace", "operations": [...]}]`` in declaration order, read
    from the ``kind: named_service`` entries of the agent's tool-connection list.
    Rows with no namespace map, or with no allowed operations, contribute nothing:
    a namespace an admin declared empty grants nothing and should not appear as a
    capability."""
    out: List[Dict[str, Any]] = []
    for conn in connections or []:
        if not isinstance(conn, Mapping):
            continue
        if _norm(conn.get("kind")).lower() != "named_service":
            continue
        namespaces = conn.get("namespaces")
        if not isinstance(namespaces, Mapping):
            continue
        alias = _norm(conn.get("alias") or conn.get("name"))
        for namespace, namespace_cfg in namespaces.items():
            ns = _norm(namespace).lower()
            if not ns or not isinstance(namespace_cfg, Mapping):
                continue
            operations = _operations(namespace_cfg)
            if not operations:
                continue
            out.append({"alias": alias, "namespace": ns, "operations": operations})
    return out


def named_service_door_servers(
    connections: Sequence[Mapping[str, Any]] | None,
    rosters: Sequence[Mapping[str, Any]] | None,
) -> Dict[str, str]:
    """``{alias: server_id}`` for the door connections the roster rows ride.

    The tie between a roster entry and its transport is the ALIAS: a
    ``kind: named_service`` row and the ``kind: mcp`` door connection that serves
    it carry the same one. An alias with no matching door connection yields no
    entry — the roster then teaches namespaces the agent has no way to reach, which
    the caller can log."""
    wanted = {_norm(row.get("alias")) for row in rosters or [] if _norm(row.get("alias"))}
    out: Dict[str, str] = {}
    for conn in connections or []:
        if not isinstance(conn, Mapping):
            continue
        if _norm(conn.get("kind")).lower() != "mcp":
            continue
        alias = _norm(conn.get("alias") or conn.get("name"))
        if alias not in wanted or alias in out:
            continue
        server_id = _norm(conn.get("server_id") or conn.get("server") or conn.get("name"))
        if server_id:
            out[alias] = server_id
    return out


def narrow_named_service_rosters(
    rosters: Sequence[Mapping[str, Any]] | None,
    disabled_namespaces: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``(surviving rows, what the user removed)`` for this turn.

    ``disabled_namespaces`` is the picker's ``disabled.named_services`` category:
    ``true`` removes a whole namespace, a list removes individual entry keys —
    operations (``object.search``) or named actions (``object.action.<name>``).
    A named action denial narrows the ACTION, not the operation, so
    ``object.action`` survives it and only the door's own gate can enforce the
    action name; it is reported in the removed map so a lane can say so.

    A namespace whose operations are all denied disappears entirely — an empty
    row would advertise a capability that can do nothing."""
    disabled = disabled_namespaces or {}
    kept: List[Dict[str, Any]] = []
    removed: Dict[str, Any] = {}
    for row in rosters or []:
        ns = _norm(row.get("namespace")).lower()
        entry = disabled.get(ns)
        if entry is True:
            removed[ns] = True
            continue
        denied = (
            {_norm(item) for item in entry if _norm(item)}
            if isinstance(entry, (list, tuple))
            else set()
        )
        if not denied:
            kept.append({**dict(row), "namespace": ns})
            continue
        operations = [op for op in (row.get("operations") or []) if _norm(op) not in denied]
        dropped = [op for op in (row.get("operations") or []) if _norm(op) in denied]
        actions = sorted(item for item in denied if item.startswith("object.action."))
        if dropped or actions:
            removed[ns] = sorted(dropped) + actions
        if not operations:
            removed[ns] = True
            continue
        kept.append({**dict(row), "namespace": ns, "operations": operations})
    return kept, removed


def named_service_door_tools(
    rosters: Sequence[Mapping[str, Any]] | None,
    *,
    narrowed: bool = False,
) -> List[str]:
    """The door MCP tools the surviving roster needs, sorted.

    The union of the surviving operations' door tools, plus the discovery tools
    (contract reading is what the door's own guide asks the model to do first).
    ``narrowed=True`` — the user removed something — withholds the generic
    ``named_services_call``, which takes its operation as an argument and would
    walk straight around the narrowing.

    An empty surviving roster yields ``[]``: no namespace, no door."""
    if not rosters:
        return []
    tools = set(DOOR_DISCOVERY_TOOLS)
    for row in rosters:
        for operation in row.get("operations") or []:
            tool = NAMED_SERVICE_OPERATION_TO_DOOR_TOOL.get(_norm(operation))
            if tool:
                tools.add(tool)
    if not narrowed:
        tools.add(DOOR_GENERIC_TOOL)
    return sorted(tools)


def named_service_roster_lines(
    rosters: Sequence[Mapping[str, Any]] | None,
) -> List[str]:
    """One line per surviving namespace, for the runtime's in-band instructions.

    The namespace half of a pick cannot be enforced by tool name (the door takes
    the namespace as an argument), so the surviving roster is also STATED: the
    agent is told which namespaces it works with this turn and what it may do in
    each, in the operation vocabulary the door itself speaks."""
    return [
        f"- {row.get('namespace')}: " + ", ".join(row.get("operations") or [])
        for row in rosters or []
        if _norm(row.get("namespace"))
    ]
