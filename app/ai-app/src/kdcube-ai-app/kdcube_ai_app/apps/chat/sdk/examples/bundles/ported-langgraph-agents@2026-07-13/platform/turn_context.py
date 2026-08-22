# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Model-facing binding for deliberate shared turn-context contributions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.contributions import (
    stage_turn_summary,
)


def build_record_turn_summary_tool(state: Dict[str, Any]) -> Any:
    """Bind trusted turn state behind one ordinary LangChain tool."""

    @tool
    async def record_turn_summary(
        summary: str,
        refs: Optional[List[str]] = None,
        phrases: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ) -> str:
        """Stage one searchable summary of this turn's reusable result.

        Capture the outcome, durable facts and decisions, and relevant object
        or file refs. ``phrases`` are exact ways a person may search for it;
        ``entities`` are names or identifiers. A later call replaces the first.
        The final draft becomes durable only if this turn completes successfully.
        This does not alter the LangGraph checkpoint or its private summaries.
        """
        try:
            receipt = stage_turn_summary(
                state,
                summary=summary,
                refs=refs,
                phrases=phrases,
                entities=entities,
                contributor="langgraph",
            )
        except ValueError as error:
            return f"Turn summary was not staged: {error}"
        action = "replaced" if receipt["replaced"] else "staged"
        return (
            f"Turn summary {action}; it becomes durable and searchable only "
            "after this turn completes successfully."
        )

    return record_turn_summary


__all__ = ["build_record_turn_summary_tool"]
