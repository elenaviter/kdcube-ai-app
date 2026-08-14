# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""The declared namespace roster, narrowed per turn (foreign_runtime/named_services.py).

Pure shaping, no I/O. What is proven: the admin's `kind: named_service`
declaration becomes the roster a wrapped runtime reads, the user's
`disabled.named_services` pick subtracts from it, and the surviving set turns
into the door's own MCP tool list plus the words the agent is told — the two
halves that exist because the door takes its namespace as an argument.
"""
from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import named_services as ns


DOOR = {
    "name": "named_services", "kind": "mcp", "server_id": "kdcube_services",
    "alias": "named_services", "url": "https://host/mcp/named_services",
}
ROSTER = {
    "name": "named_services_roster", "kind": "named_service", "alias": "named_services",
    "namespaces": {
        "linkedin": {"allowed": ["provider.about", "object.search", "object.get", "object.action"]},
        "conv": {"allowed": ["provider.about", "object.search", "object.get"]},
    },
}
CONNECTIONS = [DOOR, ROSTER]


def test_the_declaration_becomes_the_roster() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    assert [row["namespace"] for row in rosters] == ["linkedin", "conv"]
    assert rosters[0]["alias"] == "named_services"
    assert rosters[0]["operations"] == [
        "provider.about", "object.search", "object.get", "object.action",
    ]


def test_a_namespace_with_no_allowed_operations_is_not_a_capability() -> None:
    rosters = ns.named_service_rosters(
        [{"kind": "named_service", "alias": "a", "namespaces": {"empty": {}, "ok": {"allowed": ["object.get"]}}}]
    )
    assert [row["namespace"] for row in rosters] == ["ok"]


def test_a_roster_row_finds_its_door_by_alias() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    assert ns.named_service_door_servers(CONNECTIONS, rosters) == {
        "named_services": "kdcube_services"
    }


def test_a_roster_with_no_door_yields_no_server() -> None:
    rosters = ns.named_service_rosters([ROSTER])
    assert ns.named_service_door_servers([ROSTER], rosters) == {}


# ── the user's pick ─────────────────────────────────────────────────────────

def test_nothing_picked_keeps_the_whole_declared_roster() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    for disabled in ({}, None):
        kept, removed = ns.narrow_named_service_rosters(rosters, disabled)
        assert [row["namespace"] for row in kept] == ["linkedin", "conv"]
        assert removed == {}


def test_a_namespace_turned_off_whole_leaves_the_roster() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, removed = ns.narrow_named_service_rosters(rosters, {"conv": True})
    assert [row["namespace"] for row in kept] == ["linkedin"]
    assert removed == {"conv": True}


def test_denied_operations_narrow_a_namespace_that_stays_on() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, removed = ns.narrow_named_service_rosters(
        rosters, {"linkedin": ["object.action", "object.get"]}
    )
    assert kept[0]["operations"] == ["provider.about", "object.search"]
    assert removed["linkedin"] == ["object.action", "object.get"]


def test_a_namespace_whose_operations_are_all_denied_disappears() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, removed = ns.narrow_named_service_rosters(
        rosters, {"conv": ["provider.about", "object.search", "object.get"]}
    )
    assert [row["namespace"] for row in kept] == ["linkedin"]
    assert removed["conv"] is True


def test_a_named_action_denial_keeps_the_operation_and_is_reported() -> None:
    """`object.action.<name>` narrows one action, not the operation — no tool
    name can express it, so the door's own gate enforces it and the lane can say
    so from the removed map."""
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, removed = ns.narrow_named_service_rosters(
        rosters, {"linkedin": ["object.action.download"]}
    )
    assert "object.action" in kept[0]["operations"]
    assert removed["linkedin"] == ["object.action.download"]


# ── what the surviving set means downstream ─────────────────────────────────

def test_surviving_operations_pick_the_door_tools() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    tools = ns.named_service_door_tools(rosters)
    assert "named_services_search" in tools
    assert "named_services_get" in tools
    assert "named_services_action" in tools
    # never declared by either namespace
    assert "named_services_upsert" not in tools
    assert "named_services_delete" not in tools
    # contract reading is always available
    for discovery in ns.DOOR_DISCOVERY_TOOLS:
        assert discovery in tools


def test_a_narrowed_roster_withholds_the_generic_call_tool() -> None:
    """`named_services_call` takes its operation as an argument, so leaving it in
    would walk straight around the narrowing."""
    rosters = ns.named_service_rosters(CONNECTIONS)
    assert ns.DOOR_GENERIC_TOOL in ns.named_service_door_tools(rosters)
    kept, removed = ns.narrow_named_service_rosters(rosters, {"conv": True})
    assert ns.DOOR_GENERIC_TOOL not in ns.named_service_door_tools(
        kept, narrowed=bool(removed)
    )


def test_an_empty_surviving_roster_yields_no_door_tools() -> None:
    assert ns.named_service_door_tools([]) == []
    assert ns.named_service_door_tools(None) == []


def test_the_surviving_roster_is_also_stated_in_words() -> None:
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, _removed = ns.narrow_named_service_rosters(rosters, {"conv": True})
    lines = ns.named_service_roster_lines(kept)
    assert lines == [
        "- linkedin\n  operations: provider.about, object.search, object.get, object.action"
    ]
    assert ns.named_service_roster_lines([]) == []


def test_a_published_intro_is_a_detail_of_the_same_entry() -> None:
    """A deployment that publishes blurbs says more in the SAME grammar, so a
    runtime with intros and one without still render the same shape."""
    rosters = ns.named_service_rosters([ROSTER])
    lines = ns.named_service_roster_lines(
        rosters, intros={"linkedin": {"intro": "the publications realm"}}
    )
    assert lines[0].startswith("- linkedin — the publications realm\n  operations: ")
    assert lines[1].startswith("- conv\n  operations: ")


def test_a_row_with_no_declared_operations_still_names_its_namespace() -> None:
    assert ns.named_service_roster_lines([{"namespace": "conv", "operations": []}]) == ["- conv"]


# ── the block both runtimes render ──────────────────────────────────────────

def test_one_block_one_wording_for_every_runtime() -> None:
    """The whole point of this builder: a LangGraph system prompt and a Claude
    Code instructions file carry the SAME text, so the agent reads the same
    words wherever it runs. Pinned verbatim — a wording change here is a change
    to every hosted runtime's standing instructions."""
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, _removed = ns.narrow_named_service_rosters(
        rosters, {"linkedin": ["object.action"]}
    )
    assert ns.named_service_roster_block(kept) == (
        "## Service namespaces\n"
        "These namespaces are available to you this turn, each with the operations "
        "you may run in it. Pass a namespace exactly as written, and read its schema "
        "before you call it.\n"
        "\n"
        "- linkedin\n"
        "  operations: provider.about, object.search, object.get\n"
        "- conv\n"
        "  operations: provider.about, object.search, object.get"
    )


def test_the_block_says_only_what_is_available() -> None:
    """Positive framing is structural, not stylistic: what the user turned off
    has no row and no mention, so nothing suggests it ever existed."""
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, _removed = ns.narrow_named_service_rosters(rosters, {"conv": True})
    block = ns.named_service_roster_block(kept)
    assert "conv" not in block
    assert "object.action" in block          # what survives IS named
    for absent in ("not ", "disabled", "denied", "unavailable", "no longer"):
        assert absent not in block.lower()


def test_no_namespaces_no_block() -> None:
    assert ns.named_service_roster_block([]) == ""
    assert ns.named_service_roster_block(None) == ""
    rosters = ns.named_service_rosters(CONNECTIONS)
    kept, _removed = ns.narrow_named_service_rosters(
        rosters, {"linkedin": True, "conv": True}
    )
    assert ns.named_service_roster_block(kept) == ""
