# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/tools/web/filter_segmenter_fast.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import re

from langchain_core.messages import HumanMessage

from kdcube_ai_app.apps.chat.sdk.util import _today_str, _now_up_to_minutes, _json_loads_loose
from kdcube_ai_app.infra.accounting import with_accounting
from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase, create_cached_system_message
from kdcube_ai_app.apps.chat.sdk.streaming.streaming import stream_agent_to_json
import kdcube_ai_app.apps.chat.sdk.tools.web.content_filters as content_filters


def _strip_thinking_guidance(text: str) -> str:
    if not text:
        return text
    # Remove the dedicated THINKING section block
    text = re.sub(
        r"\n*════════{3,}\nTHINKING OUTPUT REQUIREMENT.*?(?=\n════════{3,}\nPHASE 1)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Remove explicit "Your thinking for Phase X" blocks
    text = re.sub(
        r"\n\*\*Your thinking for Phase 1:\*\*.*?(?=\n\s*\*\*Phase 1 output:\*\*|\n\s*════════{3,})",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"\n\*\*Your thinking for Phase 2:\*\*.*?(?=\n\s*════════{3,}|\n\s*\*\*Phase 2 output:\*\*)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Replace residual thinking references with neutral phrasing
    replacements = {
        "Signal this transition in your thinking:": "Signal this transition:",
        "After Phase 1 thinking, explicitly write:": "After Phase 1, explicitly write:",
        "Stop thinking about": "Stop considering",
        "Start thinking about": "Start focusing on",
        "Users see your thinking to understand your progress.": "",
        "DO NOT mention in Phase 2 thinking:": "DO NOT mention in Phase 2:",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _get_single_channel_protocol_filter_segmenter(json_shape_hint: str) -> str:
    return (
        "\n\n[CRITICAL OUTPUT PROTOCOL]:\n"
        "• Output ONLY a JSON object matching the schema below.\n"
        "• Do NOT include any thinking, commentary, or extra text.\n"
        "• Do NOT use any markers or additional code fences.\n"
        "• The response must be valid JSON that matches the structure exactly.\n\n"
        "EXPECTED JSON SHAPE:\n"
        f"{json_shape_hint}\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY JSON (no markdown, no prose, no prefixes).\n"
        "2. Structure: {\"<sid>\": [{\"s\": \"...\", \"e\": \"...\"}], ...}\n"
        "3. Use {} if all pages are dropped.\n"
    )


async def filter_and_segment_stream(
        svc: ModelServiceBase,
        *,
        objective: str,
        queries: List[str],
        sources_with_content: List[Dict[str, Any]],
        mode: str = "balanced",  # "balanced", "precision", "recall"
        on_progress_delta: Optional[Any] = None,
        thinking_budget: int = 180,
        max_tokens: int = 700,
        role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Filter and segment sources using single-channel JSON output (no thinking channel).
    """
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()

    if mode == "precision":
        core_instruction = content_filters.FILTER_AND_SEGMENT_HIGH_PRECISION(now_iso)
    elif mode == "recall":
        core_instruction = content_filters.FILTER_AND_SEGMENT_HIGH_RECALL(now_iso)
    else:
        core_instruction = content_filters.FILTER_AND_SEGMENT_BALANCED(now_iso)

    core_instruction = _strip_thinking_guidance(core_instruction)

    schema = (
        "{\n"
        "  \"<sid_1>\": [{\"s\": \"start anchor text\", \"e\": \"end anchor text or empty\"}],\n"
        "  \"<sid_2>\": [{\"s\": \"...\", \"e\": \"...\"}]\n"
        "}\n"
        "or {} if all pages dropped"
    )

    protocol = _get_single_channel_protocol_filter_segmenter(schema)
    from kdcube_ai_app.infra.accounting import _get_context

    context = _get_context()
    context_snapshot = context.to_dict()

    timezone = context_snapshot.get("timezone")
    today = _today_str()
    now = _now_up_to_minutes()

    time_evidence = (
        "AUTHORITATIVE TEMPORAL CONTEXT (GROUND TRUTH)\n"
        f"Current UTC date: {today}\n"
        "All relative dates (today/yesterday/last year/next month) MUST be "
        "interpreted against this context. Freshness must be estimated based on this context.\n"
    )
    time_evidence_reminder = (
        f"Very important: The user's timezone is {timezone or 'Europe/Berlin'}. "
        f"Current UTC timestamp: {now}. Current UTC date: {today}. "
        "Any dates before this are in the past, and any dates after this are in the future. "
        "When dealing with modern entities/companies/people, and the user asks for the "
        "'latest', 'most recent', 'today's', etc. don't assume your knowledge is up to date; "
        "you MUST carefully confirm what the true 'latest' is first. If the user seems confused "
        "or mistaken about a certain date or dates, you MUST include specific, concrete dates "
        "in your response to clarify things. This is especially important when the user is "
        "referencing relative dates like 'today', 'tomorrow', 'yesterday', etc -- if the user "
        "seems mistaken in these cases, you should make sure to use absolute/exact dates like "
        "'January 1, 2010' in your response.\n"
    )

    system_msg = create_cached_system_message([
        {"text": time_evidence + "\n" + core_instruction, "cache": True},
        {"text": protocol, "cache": True},
        {"text": time_evidence_reminder, "cache": False},
    ])

    input_ctx = {
        "objective": (objective or "").strip(),
        "queries": queries or [],
    }

    prepared_sources = []
    for row in sources_with_content:
        try:
            sid = int(row.get("sid"))
        except Exception:
            continue
        content = (row.get("content") or "").strip()
        if not (sid and content):
            continue

        prepared_sources.append({
            "sid": sid,
            "url": row.get("url"),
            "content": content,
            "published_time_iso": row.get("published_time_iso"),
            "modified_time_iso": row.get("modified_time_iso"),
        })

    if not prepared_sources:
        return {"agent_response": {}}

    user_msg = (
        "INPUT CONTEXT:\n" + json.dumps(input_ctx, ensure_ascii=False) + "\n\n"
        "SOURCES:\n" + json.dumps(prepared_sources, ensure_ascii=False) + "\n\n"
        "Return ONLY the JSON object."
    )
    role = role or "tool.sources.filter.by.content.and.segment"

    context = _get_context()
    context_snapshot = context.to_dict()
    track_id = context_snapshot.get("track_id")
    bundle_id = context_snapshot.get("app_bundle_id")

    async with with_accounting(
            bundle_id,
            track_id=track_id,
            agent=role,
            metadata={
                "track_id": track_id,
                "agent": role,
                "agent_name": "Filter Segmenter LLM (fast)",
            }
    ):
        out = await stream_agent_to_json(
            svc,
            client_name=role,
            client_role=role,
            sys_prompt=system_msg,
            messages=[HumanMessage(content=user_msg)],
            schema_model=None,
            on_progress_delta=on_progress_delta,
            max_tokens=max_tokens,
        )

    if not out:
        return {"agent_response": {}}

    try:
        raw_text = (out.get("agent_response") or "").strip()
        parsed = _json_loads_loose(raw_text) if raw_text else {}
        out["agent_response"] = parsed if isinstance(parsed, dict) else {}
        return out
    except Exception:
        return {"agent_response": {}}
