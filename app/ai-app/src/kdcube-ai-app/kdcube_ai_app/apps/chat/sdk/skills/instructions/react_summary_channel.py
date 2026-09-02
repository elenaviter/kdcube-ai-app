# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The native ReAct wrapper around the shared turn-summary writing guide.

The ReAct decision protocol emits the turn summary as ``<channel:summary>``
(v3) or ``channel:summary`` in backticks (v2). The rules for what to put in it
are the host-neutral ``turn_summary_writing_guide`` from the conversation
realm; this module only adds the channel mechanics and names the native search
tool when it is bound.
"""

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.conversation.instructions import (
    turn_summary_writing_guide,
)

MEMSEARCH_TOOL_ID = "react.memsearch"


def react_summary_channel_details(
    *,
    memsearch_bound: bool,
    tick_style: bool = False,
    multi_action: bool = False,
) -> str:
    """Channel mechanics plus the shared writing rules, without a trailing newline.

    ``memsearch_bound`` names ``react.memsearch`` as the reader of the summary
    only when that tool is in the effective roster; otherwise the text speaks
    of the conversation search in class vocabulary. ``tick_style`` renders the
    channel name as ```channel:summary``` (v2) instead of ``<channel:summary>``
    (v3). ``multi_action`` uses the multi-action protocol's wording, where a
    round holding only tool calls is a "call_tool-only" round.
    """
    channel = "`channel:summary`" if tick_style else "<channel:summary>"
    search_ref = f"`{MEMSEARCH_TOOL_ID}(targets=[\"summary\"])`" if memsearch_bound else None
    rounds = "call_tool-only rounds" if multi_action else "call_tool rounds"
    header = (
        f"For {rounds}, omit {channel} entirely. For complete/exit rounds, include exactly one {channel} with: Goal, Outcome, Key facts, Refs, Retrieval-anchors (phrases and entities as JSON lists). "
        "Scale it to the turn: trivial exchanges (greeting, acknowledgment, tiny answer) get one line or a few words per field. "
        "This summary is for future continuity, not for the user-facing final_answer.\n"
    )
    return header + turn_summary_writing_guide(search_ref)


__all__ = ["MEMSEARCH_TOOL_ID", "react_summary_channel_details"]
