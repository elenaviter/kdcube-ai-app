"""What lg-react is told about the namespaces it consumes (platform/named_services.py).

The declared `kind: named_service` roster is the ceiling, the conversation's
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
ROSTER = {
    "name": "named_services_roster", "kind": "named_service", "alias": "named_services",
    "namespaces": {
        "linkedin": {"allowed": ["provider.about", "object.search", "object.get", "object.action"]},
        "conv": {"allowed": ["provider.about", "object.search", "object.get"]},
    },
}
CONNECTIONS = [DOOR, ROSTER]
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
    # REGRESSION: the roster used to be composed from the declared config alone,
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
    rows, _removed = module.named_service_roster_rows(
        [ROSTER], {"linkedin": True}, connected=["conv", "linkedin", "task"]
    )
    assert [row["namespace"] for row in rows] == ["conv", "task"]
    assert rows[0]["operations"] == ["provider.about", "object.search", "object.get"]
    assert rows[1]["operations"] == []
