# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# stop_repair.py - leaving the checkpoint in a state the next turn can send
#
# A run that is cancelled mid-stream can be cancelled between two halves of one
# exchange: the model has already asked for a tool and the checkpoint has that
# request, but the result never arrives. The graph's own iteration is fine with
# it — nothing crashes, the turn ends, the answer is written — and the next turn
# then replays that history to the provider, which rejects the whole request:
#
#   400 invalid_request_error: 'tool_use' ids were found without 'tool_result'
#   blocks immediately after: toolu_… Each 'tool_use' block must have a
#   corresponding 'tool_result' block in the next message.
#
# The conversation is then wedged: every later turn replays the same history and
# gets the same 400, so a single stop costs the conversation rather than the
# turn. Seen live on lg-react, one turn after a stop.
#
# The repair is to answer the unanswered call — truthfully. A tool result saying
# the run was stopped before it ran is both what the provider requires and what
# the model should read: the next turn knows the tool did not happen, instead of
# inferring a result that never existed.
#
# Written against ANY interruption, not just a steer. A timeout kill, a worker
# dying, a provider error mid-loop all leave the same shape, so the next turn
# repairs whatever it finds rather than trusting that only stops can cause it.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger("ported-langgraph.stop_repair")

#: What the model reads in place of the result it never got. Plain language: a
#: tagged token here would name a concept the model was never taught.
UNANSWERED_TOOL_RESULT = (
    "This tool was not run. The user stopped the previous turn before it "
    "started, so there is no result. Do not treat this as a failure of the "
    "tool itself; call it again if you still need it."
)


def _tool_calls(message: Any) -> List[Dict[str, Any]]:
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return []
    out: List[Dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict) and str(call.get("id") or "").strip():
            out.append(call)
    return out


def unanswered_tool_calls(messages: Any) -> List[Tuple[str, str]]:
    """The ``(id, name)`` of every tool call left without a result.

    Answered means a later message carries that ``tool_call_id`` — the shape the
    provider checks, and the only one that matters here.
    """
    requested: List[Tuple[str, str]] = []
    answered = set()
    for message in list(messages or []):
        for call in _tool_calls(message):
            requested.append((str(call.get("id")), str(call.get("name") or "")))
        call_id = getattr(message, "tool_call_id", None)
        if call_id:
            answered.add(str(call_id))
    return [(call_id, name) for call_id, name in requested if call_id not in answered]


async def repair_unanswered_tool_calls(graph: Any, run_config: Dict[str, Any]) -> int:
    """Answer every dangling tool call in this thread's checkpoint.

    Returns how many were repaired (0 when the state is already sendable).
    Best-effort by contract: a repair that cannot be written leaves the turn
    exactly as it was — the provider error it prevents is not made worse by
    failing here, and it is logged either way.
    """
    try:
        snapshot = await graph.aget_state(run_config)
    except Exception:
        LOGGER.debug("[stop-repair] could not read the graph state", exc_info=True)
        return 0
    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return 0
    pending = unanswered_tool_calls(values.get("messages"))
    if not pending:
        return 0
    try:
        from langchain_core.messages import ToolMessage

        repairs = [
            ToolMessage(
                content=UNANSWERED_TOOL_RESULT,
                tool_call_id=call_id,
                name=name or "tool",
                status="error",
            )
            for call_id, name in pending
        ]
        await graph.aupdate_state(run_config, {"messages": repairs})
    except Exception:
        LOGGER.warning(
            "[stop-repair] could not answer %d dangling tool call(s); the next "
            "turn will be refused by the provider until this thread is repaired",
            len(pending), exc_info=True,
        )
        return 0
    LOGGER.info(
        "[stop-repair] answered %d tool call(s) left unanswered by an "
        "interrupted run: %s",
        len(pending), [call_id for call_id, _ in pending],
    )
    return len(repairs)
