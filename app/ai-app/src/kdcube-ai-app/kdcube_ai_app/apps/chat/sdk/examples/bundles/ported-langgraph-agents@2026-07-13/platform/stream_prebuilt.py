# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── stream_adapter.py ── the streaming seam (the create_agent ReAct shape) ──
#
# This is the stream-policy file that differs from a linear-graph port. The
# standalone agent is `langchain.agents.create_agent`, whose graph has a LOOPING
# `model` node (the model node) and a `tools` node:
#
#     START ─▶ model ─┬─(tool calls?)─▶ tools ─▶ model ...   (loops)
#                     └────── no tool calls ─────▶ END        (final message = answer)
#
# The `model` node fires ONCE PER TOOL-DECISION CYCLE, not once per turn. There is
# NO dedicated `answer` node (unlike the lg-solution port, whose linear graph had
# one). So "stream the answer" cannot mean "stream every token the model node
# emits" — that would stream the model's intermediate tool-deciding turns too.
#
# THE RULE (why this file exists):
#   Only the LAST model turn — the one that returns a message with NO tool call —
#   is the answer. So:
#     • Stream a model token as answer text ONLY when it carries visible content
#       and NO tool-call chunk. In the standard ReAct loop a tool-deciding turn
#       emits empty content + a tool call, so this naturally suppresses it. (A
#       model that emits "preamble" text before a tool call in the same turn is
#       the one caveat; the ReAct loop's tool turns emit no visible text.)
#     • Surface each `tools` run as a step (tool start -> running, end ->
#       completed), so the user sees the loop working.
#     • The authoritative final answer is the last model turn's message content
#       when it makes no tool call — used to emit a single delta on the offline /
#       non-streaming path, and as the returned value.
#
# Compaction (SummarizationMiddleware) runs in its OWN before_model middleware node,
# not the `model` node, so its summarization tokens never reach this streaming path.
#
# A different agent shape supplies an AgentSpec with its own build function, input
# mapper, model role, and stream adapter. Identity, storage, economics, capabilities,
# and conversation integration remain shared; looping-node interpretation lives here.

from __future__ import annotations

import logging
from typing import Any, Dict

from kdcube_ai_app.apps.chat.sdk.runtime import comm_ctx
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.stream_contract import (
    content_text,
    tool_result_view,
    tool_call_views,
)

LOGGER = logging.getLogger("kdcube.ported_langgraph_agents.stream_react")


# ── tool-call rendering for the Steps view ───────────────────────────────────
# A step row that says just "run_python / running" hides the one thing that
# matters when a call misbehaves: WHAT the model actually passed. Each tool
# invocation therefore gets its own step whose title is a compact call
# signature (`run_python(code=<2.4 KB>, prog_name='news')`) and whose body
# shows the arguments. The rendering (and the chunk-content normalizer) is the
# shared foreign-runtime seam's; local aliases for readability.

_tool_call_views = tool_call_views
_tool_result_view = tool_result_view
_content_text = content_text


def _is_agent_node_event(name: Any, node: Any, agent_node: str) -> bool:
    """LangGraph event names have drifted across versions, but
    ``metadata.langgraph_node`` remains the semantic node id. Accept either so a
    harmless tracing/name change cannot make answer deltas disappear."""
    agent = str(agent_node or "")
    return str(node or "") == agent or str(name or "") == agent


async def _emit_answer_delta(text: str, index: int) -> bool:
    comm = comm_ctx.get_comm()
    if comm is None:
        LOGGER.warning(
            "[ported-langgraph] lg-react answer delta skipped: no communicator bound "
            "index=%d text_len=%d",
            index, len(text or ""),
        )
        return False
    await comm.delta(text=text, index=index, marker="answer")
    return True


async def _emit_complete(answer: str) -> bool:
    comm = comm_ctx.get_comm()
    if comm is None:
        LOGGER.warning(
            "[ported-langgraph] lg-react complete skipped: no communicator bound "
            "answer_len=%d",
            len(answer or ""),
        )
        return False
    await comm.complete(data={"final_answer": answer})
    return True


async def stream_react_turn(
    graph: Any,
    inputs: Dict[str, Any],
    run_config: Dict[str, Any],
    *,
    agent_node: str = "model",
) -> str:
    """Run one turn of a create_agent ReAct ``graph`` and stream it through the
    current communicator. Returns the final answer text (also set on the platform
    state by the caller, so the turn is streamed live AND recorded for reload).

    ``agent_node`` is the looping model node whose FINAL (no-tool-call) turn is the
    user-visible answer.
    """
    idx = 0
    answer = ""
    # Track one model turn at a time. The normal path streams visible model
    # chunks immediately; the node-end content is only a fallback for models that
    # do not emit token stream chunks.
    model_turn_has_tool_call = False
    model_turn_active = False
    model_turn_streamed_answer = False

    def _reset_model_turn() -> None:
        nonlocal model_turn_has_tool_call, model_turn_active, model_turn_streamed_answer
        model_turn_has_tool_call = False
        model_turn_active = True
        model_turn_streamed_answer = False
    # Steps are keyed by their `step` string client-side, so every tool INVOCATION
    # gets its own key (`run_python`, `run_python (2)`, …) — a retry loop shows as
    # N rows with their actual arguments, not one row silently overwritten.
    tool_call_seq: Dict[str, int] = {}
    tool_run_step: Dict[str, tuple] = {}  # run_id -> (step_key, title, markdown)
    answer_delta_count = 0

    async for event in graph.astream_events(inputs, run_config, version="v2"):
        kind = event.get("event")
        name = event.get("name")
        node = (event.get("metadata") or {}).get("langgraph_node")

        is_agent_node = _is_agent_node_event(name, node, agent_node)

        if kind in {"on_chain_start", "on_chat_model_start"} and is_agent_node:
            # A new model turn begins — until proven otherwise it might be final.
            _reset_model_turn()

        elif kind == "on_chat_model_stream" and is_agent_node:
            chunk = (event.get("data") or {}).get("chunk")
            # A tool-call chunk marks this agent turn as a tool-deciding turn: it
            # is NOT the answer, so never stream it as answer text.
            if getattr(chunk, "tool_call_chunks", None):
                if not model_turn_active:
                    _reset_model_turn()
                model_turn_has_tool_call = True
            token = _content_text(getattr(chunk, "content", ""))
            if token and not model_turn_has_tool_call:
                if not model_turn_active:
                    _reset_model_turn()
                answer += token
                if await _emit_answer_delta(token, idx):
                    answer_delta_count += 1
                idx += 1
                model_turn_streamed_answer = True

        elif kind == "on_tool_start":
            # Surface each tool run as a progress step showing HOW it was called:
            # title = the call signature, body = the arguments (large values as
            # fenced blocks; empty args stated explicitly).
            tool_args = (event.get("data") or {}).get("input")
            title, markdown = _tool_call_views(str(name), tool_args)
            seq = tool_call_seq.get(str(name), 0) + 1
            tool_call_seq[str(name)] = seq
            step_key = str(name) if seq == 1 else f"{name} ({seq})"
            run_id = str(event.get("run_id") or "")
            if run_id:
                tool_run_step[run_id] = (step_key, title, markdown)
            LOGGER.info("[ported-langgraph] lg-react tool START: %s", title)
            await comm_ctx.step(step=step_key, status="running", title=title, markdown=markdown)

        elif kind == "on_tool_end":
            run_id = str(event.get("run_id") or "")
            step_key, title, markdown = tool_run_step.pop(
                run_id, (str(name), str(name), "")
            )
            result_markdown = _tool_result_view((event.get("data") or {}).get("output"))
            if result_markdown:
                markdown = (
                    f"{markdown}\n\n---\n\n{result_markdown}"
                    if str(markdown or "").strip()
                    else result_markdown
                )
            LOGGER.info("[ported-langgraph] lg-react tool END: %s", title)
            await comm_ctx.step(step=step_key, status="completed", title=title, markdown=markdown)

        elif kind == "on_chain_end" and is_agent_node:
            # The agent turn just finished. Read its last message (authoritative):
            #   - has tool_calls  -> intermediate turn; the next cycle continues.
            #   - no tool_calls   -> the FINAL answer. If token streaming already
            #     produced answer deltas, leave them alone; otherwise emit the
            #     returned content as the offline/non-streaming fallback.
            out = (event.get("data") or {}).get("output") or {}
            msgs = out.get("messages") if isinstance(out, dict) else None
            last = msgs[-1] if msgs else None
            has_tool_calls = bool(getattr(last, "tool_calls", None)) or model_turn_has_tool_call
            if last is not None and not has_tool_calls and not model_turn_streamed_answer:
                content = _content_text(getattr(last, "content", ""))
                if content:
                    answer += content
                    if await _emit_answer_delta(content, idx):
                        answer_delta_count += 1
                    idx += 1
            model_turn_has_tool_call = False
            model_turn_active = False
            model_turn_streamed_answer = False

    complete_emitted = await _emit_complete(answer)
    LOGGER.info(
        "[ported-langgraph] lg-react turn complete: answer_len=%d "
        "answer_delta_count=%d complete_emitted=%s",
        len(answer), answer_delta_count, complete_emitted,
    )
    return answer
