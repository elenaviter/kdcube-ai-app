# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── stream_adapter.py ── the streaming seam ──
#
# The standalone CLI already consumed the graph through
# `graph.astream_events(..., version="v2")` (see poc/lg-solution/lg_solution/
# cli.py): answer-node token events became printed text, node starts became
# stderr step lines. The platform layer keeps that exact loop but redirects it at KDCube's
# comm primitives instead of stdout, so the existing chat component renders the
# turn live with zero UI work.
#
# The translation is 1:1:
#   on_chain_start   (a graph node)          -> comm_ctx.step(node, "running")
#   on_chain_end     (a graph node)          -> comm_ctx.step(node, "completed")
#   on_chat_model_stream (in the answer node) -> comm_ctx.delta(token, marker="answer")
#   final answer                              -> return value + comm_ctx.complete
#
# Nothing about the graph changes. This module is framework-shaped
# (astream_events is LangChain/LangGraph); a different framework's port swaps
# this file for its own event loop and keeps everything else.

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from kdcube_ai_app.apps.chat.sdk.runtime import comm_ctx
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.stream_contract import (
    content_text,
)

LOGGER = logging.getLogger("kdcube.ported_langgraph_agents.stream_solution")

# The generic chunk-content normalizer (content may stream as a LIST of content
# blocks) lives in the shared foreign-runtime seam; local alias for readability.
_content_text = content_text


async def stream_graph_turn(
    graph: Any,
    inputs: Dict[str, Any],
    run_config: Dict[str, Any],
    *,
    answer_node: str = "answer",
    step_nodes: Optional[Iterable[str]] = None,
) -> str:
    """Run one turn of ``graph`` and stream it through the current communicator.

    Returns the final answer text (also set on the platform state by the caller,
    so the turn is both streamed live and recorded for reload).

    ``step_nodes`` are the graph node names surfaced as progress steps; default
    to the answer node plus none (caller passes the real set). ``answer_node`` is
    the node whose model tokens are the user-visible answer — other nodes'
    tokens (planner, subagent) are intentionally not streamed as the answer.
    """
    steps = set(step_nodes or {answer_node})

    idx = 0
    answer = ""
    async for event in graph.astream_events(inputs, run_config, version="v2"):
        kind = event.get("event")
        name = event.get("name")
        node = (event.get("metadata") or {}).get("langgraph_node")

        if kind == "on_chain_start" and name in steps:
            LOGGER.info("[ported-langgraph] lg-solution node START: %s", name)
            await comm_ctx.step(step=name, status="running")

        elif kind == "on_chain_end" and name in steps and name != answer_node:
            await comm_ctx.step(step=name, status="completed")

        elif kind == "on_chat_model_stream" and node == answer_node:
            chunk = (event.get("data") or {}).get("chunk")
            token = _content_text(getattr(chunk, "content", ""))
            if token:
                await comm_ctx.delta(text=token, index=idx, marker="answer")
                idx += 1
                answer += token

        elif kind == "on_chain_end" and name == answer_node:
            # Offline/stub mode emits no token stream — take the node's returned
            # answer and stream it as a single delta so the UI still renders it.
            if not answer:
                out = (event.get("data") or {}).get("output") or {}
                node_answer = out.get(answer_node) if isinstance(out, dict) else ""
                if node_answer:
                    answer = _content_text(node_answer) or str(node_answer)
                    await comm_ctx.delta(text=answer, index=idx, marker="answer")
                    idx += 1
            await comm_ctx.step(step=answer_node, status="completed")

    LOGGER.info("[ported-langgraph] lg-solution turn complete: answer_len=%d", len(answer))
    await comm_ctx.complete(data={"final_answer": answer})
    return answer
