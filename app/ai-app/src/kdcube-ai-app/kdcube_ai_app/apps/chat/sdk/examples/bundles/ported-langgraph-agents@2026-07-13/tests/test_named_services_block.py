"""What lg-react is told about the namespaces it consumes (platform/named_services.py).

The declared `kind: named_service` inventory is the ceiling, the conversation's
capabilities pick subtracts from it, and what survives is stated to the agent in
the SEAM's block — the same text every other wrapped runtime renders. Asserts:
the block is the shared one (not a bundle wording), the pick reaches it, a
namespace that is off is not mentioned at all, and the teaching half names the
door's exact tools only when the door is actually bound. Fully offline.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.runtime.dynamic_module_loader import load_dynamic_module_for_path
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.named_services import (
    named_service_roster_block,
)

BUNDLE_ROOT = Path(__file__).resolve().parents[1]

AGENT_ID = "lg-react"
DOOR = {
    "name": "named_services", "kind": "mcp", "server_id": "named_services",
    "alias": "named_services", "url": "https://h/mcp/named_services", "delegated": True,
}
INVENTORY = {
    "name": "named_services_roster", "kind": "named_service", "alias": "named_services",
    "namespaces": {
        "linkedin": {"allowed": ["provider.about", "object.search", "object.get", "object.action"]},
        "conv": {"allowed": ["provider.about", "object.search", "object.get"]},
    },
}
CONNECTIONS = [DOOR, INVENTORY]
PROPS = {"surfaces": {"as_consumer": {"agents": {AGENT_ID: {"tools": CONNECTIONS}}}}}
DOOR_TOOL_NAMES = ("named_services_list", "named_services_search", "named_services_get")


def _module():
    _name, module = load_dynamic_module_for_path(BUNDLE_ROOT / "platform" / "named_services.py")
    return module


def _build(disabled=None, tool_names=DOOR_TOOL_NAMES, connections=CONNECTIONS, props=PROPS):
    return asyncio.run(_module().build_named_services_block(
        bundle_props=props,
        agent_id=AGENT_ID,
        connections=connections,
        disabled_namespaces=disabled,
        tool_names=tool_names,
    ))


def test_the_prompt_carries_the_SHARED_roster_block() -> None:
    # One block, one wording, both runtimes: the bundle composes the seam's text
    # verbatim rather than a prompt phrasing of its own.
    block, namespaces = _build()
    assert namespaces == ["linkedin", "conv"]
    rows = [
        {"namespace": "linkedin", "operations": ["provider.about", "object.search", "object.get", "object.action"]},
        {"namespace": "conv", "operations": ["provider.about", "object.search", "object.get"]},
    ]
    assert named_service_roster_block(rows) in block


def test_the_conversation_pick_reaches_the_prompt() -> None:
    # REGRESSION: the inventory used to be composed from the declared config alone,
    # so a namespace the user turned off was still announced to the model.
    block, namespaces = _build({"conv": True})
    assert namespaces == ["linkedin"]
    assert "conv" not in block.replace("conversation", "")
    # a narrowed namespace keeps its surviving operations, loses the others
    block, _ns = _build({"linkedin": ["object.action"]})
    assert "- linkedin\n  operations: provider.about, object.search, object.get\n" in block


def test_every_namespace_off_leaves_no_block() -> None:
    block, namespaces = _build({"linkedin": True, "conv": True})
    assert block == "" and namespaces == []
    # ... and so does an agent that declares none at all
    assert _build(None, connections=[], props={}) == ("", [])


def test_the_teaching_half_names_the_door_tools_only_when_they_are_bound() -> None:
    bound, _ns = _build(tool_names=DOOR_TOOL_NAMES)
    assert "`named_services_search`" in bound
    unbound, _ns = _build(tool_names=("web_search",))
    assert "named_services_search" not in unbound
    assert "search_objects" in unbound  # taught by operation name instead


def test_a_namespace_reached_without_an_operation_list_is_still_a_row() -> None:
    # `task` is connected by another declaration (an event source, a canvas
    # resolver) — it exists for the agent, so it is a row; with no declared
    # operations it is named alone. A namespace the user turned off is neither.
    module = _module()
    rows, _removed = module.named_service_inventory_rows(
        [INVENTORY], {"linkedin": True}, connected=["conv", "linkedin", "task"]
    )
    assert [row["namespace"] for row in rows] == ["conv", "task"]
    assert rows[0]["operations"] == ["provider.about", "object.search", "object.get"]
    assert rows[1]["operations"] == []


# ── the pick reaches the BINDING, not only the prompt ──────────────────────────
#
# The door publishes one tool per operation plus the generic `named_services_call`,
# and every namespace rides the SAME tools (the namespace is an argument). So the
# inventory narrows the door's tools exactly where an operation survives NOWHERE, and
# the per-namespace half stays with the door's own gate — stated in the block's
# words, asserted honestly below rather than claimed.

class _BoundTool:
    def __init__(self, name: str, server_id: str) -> None:
        self.name = name
        self.metadata = {"mcp_server_id": server_id}

    def __repr__(self) -> str:  # readable assertion failures
        return f"{self.name}@{self.metadata['mcp_server_id']}"


def _door_tools(*names: str):
    return [_BoundTool(name, "named_services") for name in names]


def _names(tools):
    return [t.name for t in tools]


def test_an_operation_denied_everywhere_loses_its_door_tool() -> None:
    # REGRESSION: lg-react bound every tool the door published, so a user who
    # unchecked an operation kept an agent holding it — the roster block said one
    # thing and the tool list allowed another.
    module = _module()
    bound = _door_tools(
        "named_services_search", "named_services_get", "named_services_action",
        "named_services_list", "named_services_call",
    ) + [_BoundTool("web_search", "web")]
    kept = module.narrow_bound_door_tools(
        bound, CONNECTIONS,
        {"linkedin": ["object.get", "object.action"], "conv": ["object.get"]},
    )
    assert "named_services_get" not in _names(kept)      # denied in BOTH namespaces
    assert "named_services_action" not in _names(kept)   # only linkedin had it
    assert "named_services_search" in _names(kept)       # still allowed
    assert "named_services_list" in _names(kept)         # discovery always survives
    assert "web_search" in _names(kept)                  # another server, untouched


def test_the_generic_call_tool_goes_as_soon_as_the_pick_removes_anything() -> None:
    # It takes its operation as an ARGUMENT, so leaving it bound would reach
    # exactly what was just removed — including operations the ADMIN never declared.
    module = _module()
    bound = _door_tools("named_services_search", "named_services_call")
    assert "named_services_call" in _names(
        module.narrow_bound_door_tools(bound, CONNECTIONS, None)
    )
    assert "named_services_call" not in _names(
        module.narrow_bound_door_tools(bound, CONNECTIONS, {"conv": ["object.get"]})
    )


def test_an_operation_kept_in_ONE_namespace_keeps_its_tool() -> None:
    # THE HONEST LIMIT, pinned so nobody reads the narrowing as more than it is:
    # `object.get` denied in linkedin but allowed in conv still binds
    # `named_services_get`, because conv needs it and one tool serves both. The
    # namespace argument is enforced by the door's gate, not by a tool name.
    module = _module()
    bound = _door_tools("named_services_get")
    kept = module.narrow_bound_door_tools(bound, CONNECTIONS, {"linkedin": ["object.get"]})
    assert _names(kept) == ["named_services_get"]


def test_every_namespace_off_leaves_no_door_tool_at_all() -> None:
    module = _module()
    bound = _door_tools("named_services_search", "named_services_list", "named_services_call")
    kept = module.narrow_bound_door_tools(bound, CONNECTIONS, {"linkedin": True, "conv": True})
    assert kept == []


def test_no_declared_inventory_narrows_nothing() -> None:
    # An app that declares only the door connection has stated no ceiling, so
    # there is nothing to narrow against and the tools pass through untouched.
    module = _module()
    bound = _door_tools("named_services_search", "named_services_call")
    assert module.narrow_bound_door_tools(bound, [DOOR], {"linkedin": True}) == bound
    assert module.narrow_bound_door_tools(bound, CONNECTIONS, None) == bound
