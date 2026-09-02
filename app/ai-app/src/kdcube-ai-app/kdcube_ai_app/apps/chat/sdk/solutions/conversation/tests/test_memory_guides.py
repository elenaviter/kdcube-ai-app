# SPDX-License-Identifier: MIT
"""The summary-writing guide and the query guide are two halves of one
mechanism. Every surface that teaches one of them must carry the shared core,
so the halves cannot drift apart per host (native ReAct, hosted LangGraph,
hosted Claude Code, external MCP clients)."""

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.conversation.instructions import (
    CONVERSATION_QUERY_GUIDE,
    QUERY_GUIDE_SENTINEL,
    SUMMARY_GUIDE_SENTINEL,
    turn_summary_writing_guide,
)
from kdcube_ai_app.apps.chat.sdk.skills.instructions.react_summary_channel import (
    react_summary_channel_details,
)
from kdcube_ai_app.apps.chat.sdk.skills.instructions.workspace_agent_instructions import (
    conversation_recovery_guide,
    turn_summary_contribution_guide,
)
from kdcube_ai_app.apps.chat.sdk.skills.instructions.shared_instructions_lite import (
    REACT_LITE_MEMORY_SEARCH_RECOVERY,
)
from kdcube_ai_app.apps.chat.sdk.skills.instructions.instructions_extra_lite import (
    REACT_XLITE_RECOVERY,
)
from kdcube_ai_app.apps.chat.sdk.skills.instructions import shared_instructions
from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.memsearch import TOOL_SPEC
from kdcube_ai_app.apps.chat.sdk.solutions.conversation.named_service import (
    CONVERSATION_SEARCH_FILTERS,
    CONVERSATION_SEARCH_SCOPES,
    SERVICE_ABOUT,
)


# --- the two cores ------------------------------------------------------------

def test_query_guide_teaches_the_mechanics_the_index_actually_has():
    g = CONVERSATION_QUERY_GUIDE
    assert QUERY_GUIDE_SENTINEL in g
    # AND semantics, quoting, OR, exclusion, anchors weight, fuzzy, recency
    assert "every unquoted word must occur" in g
    assert '"quoted phrase" must occur verbatim' in g
    assert "OR separates alternatives" in g
    assert "-word excludes" in g
    assert "retrieval anchors carry the highest weight" in g
    assert "typos yes, synonyms no" in g
    assert "about 2x" in g
    # the failing shape by name, and where time belongs
    assert "last time I worked with" in g
    assert "time goes to from/to" in g
    # host-neutral: no tool named inside the core
    assert "react." not in g
    assert "channel:" not in g
    assert "named_services" not in g


def test_summary_guide_binds_writing_to_reading():
    g = turn_summary_writing_guide()
    assert SUMMARY_GUIDE_SENTINEL in g
    assert "Name things by their searchable names" in g
    assert "the conversation search's `summary` target matches THIS text" in g
    assert "Forecast-Q2-2026.xlsx" in g
    assert "react." not in g and "channel:" not in g
    native = turn_summary_writing_guide('`react.memsearch(targets=["summary"])`')
    assert '`react.memsearch(targets=["summary"])` matches THIS text' in native


# --- carriers: reading side ---------------------------------------------------

def test_memsearch_spec_carries_the_query_guide_and_documents_rank_weights():
    assert QUERY_GUIDE_SENTINEL in TOOL_SPEC["purpose"]
    assert "your own channel:summary texts" in TOOL_SPEC["purpose"]
    assert "Content words only" in TOOL_SPEC["args"]["query"]
    assert "rank_weights" in TOOL_SPEC["args"]
    assert "never in `query`" in TOOL_SPEC["args"]["from"]


def test_named_service_carries_the_query_guide_and_the_catalog_filters():
    assert QUERY_GUIDE_SENTINEL in CONVERSATION_SEARCH_SCOPES[0].description
    assert QUERY_GUIDE_SENTINEL in SERVICE_ABOUT["description"]
    for key in ("ordinal", "order", "top_k", "from", "to", "days", "targets", "scope"):
        assert key in CONVERSATION_SEARCH_FILTERS, key


def test_hosted_agent_recovery_guide_carries_the_query_guide():
    block = conversation_recovery_guide(namespace="conv", pull_tool="pull_files")
    assert "[CONVERSATION RECOVERY — `conv` namespace]" in block
    assert QUERY_GUIDE_SENTINEL in block
    assert "react." not in block and "channel:" not in block


def test_lite_and_full_react_blocks_carry_the_query_shape():
    for text in (REACT_LITE_MEMORY_SEARCH_RECOVERY, REACT_XLITE_RECOVERY):
        assert QUERY_GUIDE_SENTINEL in text
        assert "channel:summary texts" in text
    full = shared_instructions.__dict__
    joined = "\n".join(v for v in full.values() if isinstance(v, str))
    assert QUERY_GUIDE_SENTINEL in joined
    assert "It supports semantic search plus ordinal/temporal turn lookup" not in joined


# --- carriers: writing side ---------------------------------------------------

def test_react_channel_wrapper_names_memsearch_only_when_bound():
    bound = react_summary_channel_details(memsearch_bound=True)
    assert SUMMARY_GUIDE_SENTINEL in bound
    assert "<channel:summary>" in bound
    assert 'react.memsearch(targets=["summary"])' in bound
    unbound = react_summary_channel_details(memsearch_bound=False)
    assert SUMMARY_GUIDE_SENTINEL in unbound
    assert "react.memsearch" not in unbound
    assert "the conversation search's `summary` target" in unbound
    v2 = react_summary_channel_details(memsearch_bound=True, tick_style=True)
    assert "`channel:summary`" in v2 and "<channel:summary>" not in v2
    assert not bound.endswith("\n")
    # the multi-action protocol keeps its own wording for tool-only rounds
    multi = react_summary_channel_details(memsearch_bound=True, multi_action=True)
    assert "For call_tool-only rounds, omit <channel:summary> entirely" in multi
    assert "For call_tool rounds, omit <channel:summary> entirely" in bound


def test_hosted_agent_summary_guide_carries_the_writing_rules_tool_neutrally():
    block = turn_summary_contribution_guide()
    assert "[SHARED TURN CONTEXT — `record_turn_summary`]" in block
    assert SUMMARY_GUIDE_SENTINEL in block
    assert "the conversation search's `summary` target" in block
    assert "react." not in block and "channel:" not in block
    named = turn_summary_contribution_guide(
        tool_name="record_turn_summary",
        search_ref="the `conv` namespace search with targets summary",
    )
    assert "the `conv` namespace search with targets summary matches THIS text" in named
