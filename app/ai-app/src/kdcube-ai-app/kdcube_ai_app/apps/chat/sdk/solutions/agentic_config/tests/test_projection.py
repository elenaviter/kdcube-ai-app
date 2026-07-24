# SPDX-License-Identifier: MIT

"""The forge's cost meter: the projection composes the REAL system text and
its breakdown reacts to the draft's levers (catalog detail, roster,
gallery, subagents); unloadable tool kinds are reported, never silently
dropped."""

import asyncio

from kdcube_ai_app.apps.chat.sdk.solutions.agentic_config.projection import (
    project_agent_config,
)

WEB_TOOLS = {
    "name": "web",
    "alias": "web_tools",
    "kind": "python",
    "module": "kdcube_ai_app.apps.chat.sdk.tools.web_tools",
}


def _project(draft):
    return asyncio.run(project_agent_config(draft, store=None))


def test_projection_breakdown_and_roster():
    result = _project({
        "react": {"instructions": {"blocks": ["instr:profile:extra-lite"], "tool_catalog_detail": "compact"}},
        "consumer": {"tools": [WEB_TOOLS, {"name": "mymcp", "kind": "mcp"}]},
    })
    tokens = result["tokens"]
    assert tokens["total"] > 0
    # the breakdown comes from diffs of real compositions and adds up
    assert tokens["total"] == (
        tokens["protocol_and_instructions"] + tokens["tool_catalog"] + tokens["skill_gallery"]
    )
    # the draft roster is loaded and projected
    assert "web_tools.web_search" in result["tools"]["included_ids"]
    # unloadable kinds are reported, not silently dropped
    assert any("mymcp" in s for s in result["tools"]["skipped"])


def test_projection_reacts_to_catalog_detail_and_subagents():
    base = {
        "react": {"instructions": {"blocks": ["instr:profile:extra-lite"], "tool_catalog_detail": "compact"}},
        "consumer": {"tools": [WEB_TOOLS]},
    }
    compact = _project(base)

    full = _project({
        "react": {"instructions": {"blocks": ["instr:profile:extra-lite"], "tool_catalog_detail": "full"}},
        "consumer": {"tools": [WEB_TOOLS]},
    })
    assert full["tokens"]["tool_catalog"] > compact["tokens"]["tool_catalog"]

    with_subagents = _project({
        "react": {
            "instructions": {"blocks": ["instr:profile:extra-lite"], "tool_catalog_detail": "compact"},
            "subagents": {"enabled": True},
        },
        "consumer": {"tools": [WEB_TOOLS]},
    })
    # subagents add the delegation tool to the projected catalog
    assert with_subagents["tokens"]["tool_catalog"] > compact["tokens"]["tool_catalog"]
    assert with_subagents["facets"]["subagents"] is True


def test_projection_allowed_narrows_roster():
    narrowed = _project({
        "react": {},
        "consumer": {"tools": [{**WEB_TOOLS, "allowed": ["web_search"]}]},
    })
    assert narrowed["tools"]["included_ids"] == ["web_tools.web_search"]
