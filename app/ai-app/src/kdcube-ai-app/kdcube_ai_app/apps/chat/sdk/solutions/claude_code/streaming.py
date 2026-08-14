# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

from typing import Any

from kdcube_ai_app.infra.accounting.usage import _norm_usage_dict

CLAUDE_CODE_PROVIDER = "anthropic"

_CLAUDE_CODE_MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "best": "claude-opus-4-6",
    "claude-opus": "claude-opus-4-6",
    "opus-4.6": "claude-opus-4-6",
    "opus-4-6": "claude-opus-4-6",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-opus-4-6": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
}


#: Content blocks that are the runtime's own machinery rather than the agent's
#: words. A ``tool_result`` block carries whatever the tool returned — a file
#: read comes back as the whole file, line-numbered — and a ``tool_use`` block
#: carries the call's JSON arguments. Streaming either into the answer prints
#: the agent's workings on top of its answer; the reader wants what it CONCLUDED.
_NON_ANSWER_BLOCK_TYPES = frozenset({"tool_result", "tool_use", "thinking", "redacted_thinking"})


def extract_text_from_claude_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part for part in (extract_text_from_claude_content(item) for item in value) if part)
    if not isinstance(value, dict):
        return ""

    if str(value.get("type") or "") in _NON_ANSWER_BLOCK_TYPES:
        return ""

    if isinstance(value.get("text"), str):
        return value["text"]

    for key in ("content", "message", "delta", "result"):
        if key in value:
            text = extract_text_from_claude_content(value[key])
            if text:
                return text
    return ""


#: Stream events that never carry answer text. A ``user`` event is how Claude
#: Code reports a TOOL RESULT back into its own conversation — the file it read,
#: the command output — and a ``system`` event is session bookkeeping. Both have
#: a ``content`` key, so a generic extractor happily streams them to the reader.
_NON_ANSWER_EVENT_TYPES = frozenset({"user", "system"})


def extract_tool_uses_from_claude_event(value: Any) -> list[dict[str, Any]]:
    """``[{"name", "input"}, …]`` for the tool calls an assistant event makes.

    The reader still deserves to see that the agent read a file or searched the
    store — just as an activity row, not as the file's contents pasted into the
    answer. The caller renders these as steps."""
    if not isinstance(value, dict) or str(value.get("type") or "") != "assistant":
        return []
    message = value.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or str(block.get("type") or "") != "tool_use":
            continue
        name = str(block.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "id": str(block.get("id") or ""),
            "name": name,
            "input": block.get("input") if isinstance(block.get("input"), dict) else {},
        })
    return out


def extract_text_from_claude_event(value: Any) -> str:
    if not isinstance(value, dict):
        return ""

    if str(value.get("type") or "") in _NON_ANSWER_EVENT_TYPES:
        return ""

    for key in ("text", "completion", "message", "delta", "result", "content"):
        if key in value:
            text = extract_text_from_claude_content(value[key])
            if text:
                return text
    return ""


def compute_incremental_chunk(previous_snapshot: str, new_text: str) -> tuple[str, str]:
    if not new_text:
        return previous_snapshot, ""
    if not previous_snapshot:
        return new_text, new_text
    if new_text.startswith(previous_snapshot):
        return new_text, new_text[len(previous_snapshot):]

    common_prefix = 0
    for prev_char, next_char in zip(previous_snapshot, new_text):
        if prev_char != next_char:
            break
        common_prefix += 1
    return new_text, new_text[common_prefix:]


def accumulate_transcript(
    transcript: str,
    previous_snapshot: str,
    new_text: str,
    *,
    separator: str = "\n\n",
) -> tuple[str, str, str]:
    """
    Maintain a full transcript across Claude Code partial snapshots.

    Claude Code often emits cumulative snapshots for one logical assistant
    message. Sometimes it emits a new logical message whose text no longer
    extends the previous snapshot. In that case we keep the previous snapshot
    in the transcript and start a new live snapshot instead of replacing the
    whole output.

    Returns:
    - updated transcript
    - updated live snapshot
    - incremental chunk to emit to the UI
    """
    if not new_text:
        return transcript, previous_snapshot, ""

    if not previous_snapshot:
        return transcript, new_text, new_text

    if new_text.startswith(previous_snapshot):
        return transcript, new_text, new_text[len(previous_snapshot):]

    base = transcript
    if previous_snapshot:
        base = f"{base}{separator}{previous_snapshot}" if base else previous_snapshot

    emit_prefix = separator if base else ""
    return base, new_text, f"{emit_prefix}{new_text}"


def normalize_claude_code_model(value: Any) -> str | None:
    if not value:
        return None
    model = str(value).strip()
    if not model:
        return None
    lowered = model.lower()
    return _CLAUDE_CODE_MODEL_ALIASES.get(lowered, model)


def extract_model_from_claude_event(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    candidates = [value.get("model")]
    message = value.get("message")
    if isinstance(message, dict):
        candidates.append(message.get("model"))
    result = value.get("result")
    if isinstance(result, dict):
        candidates.append(result.get("model"))

    for candidate in candidates:
        normalized = normalize_claude_code_model(candidate)
        if normalized:
            return normalized
    return None


def extract_usage_from_claude_event(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    candidates: list[Any] = [value.get("usage")]
    message = value.get("message")
    if isinstance(message, dict):
        candidates.append(message.get("usage"))
    result = value.get("result")
    if isinstance(result, dict):
        candidates.append(result.get("usage"))

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return None


def extract_result_metrics_from_claude_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    result_dict = value.get("result") if isinstance(value.get("result"), dict) else None
    container = result_dict if result_dict is not None else value

    out: dict[str, Any] = {}
    for key in ("duration_ms", "duration_api_ms", "api_duration_ms"):
        metric = container.get(key)
        if metric is not None:
            try:
                out[key] = int(metric)
            except Exception:
                pass

    for key in ("total_cost_usd", "cost_usd", "cost"):
        raw = container.get(key)
        if raw is None:
            continue
        try:
            out["cost_usd"] = float(raw)
            break
        except Exception:
            continue

    return out


def is_usage_bearing_message_event(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    event_type = str(value.get("type") or "").strip().lower()
    return event_type in {"assistant", "user"}


def is_result_event(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    event_type = str(value.get("type") or "").strip().lower()
    return event_type == "result"


def accumulate_usage(
    current: dict[str, Any] | None,
    usage_payload: dict[str, Any],
    *,
    default_requests: int = 1,
) -> dict[str, Any]:
    current = dict(current or {})
    normalized = _norm_usage_dict(usage_payload or {})

    current["input_tokens"] = int(current.get("input_tokens", 0) or 0) + int(normalized.get("input_tokens", 0) or 0)
    current["output_tokens"] = int(current.get("output_tokens", 0) or 0) + int(normalized.get("output_tokens", 0) or 0)
    current["thinking_tokens"] = int(current.get("thinking_tokens", 0) or 0) + int(normalized.get("thinking_tokens", 0) or 0)
    current["cache_creation_tokens"] = int(current.get("cache_creation_tokens", 0) or 0) + int(normalized.get("cache_creation_input_tokens", 0) or 0)
    current["cache_read_tokens"] = int(current.get("cache_read_tokens", 0) or 0) + int(normalized.get("cache_read_input_tokens", 0) or 0)
    current["total_tokens"] = int(current.get("total_tokens", 0) or 0) + int(normalized.get("total_tokens", 0) or 0)

    current_cache_creation = current.get("cache_creation")
    if not isinstance(current_cache_creation, dict):
        current_cache_creation = {}
    new_cache_creation = normalized.get("cache_creation")
    if isinstance(new_cache_creation, dict):
        for key, raw in new_cache_creation.items():
            try:
                current_cache_creation[key] = int(current_cache_creation.get(key, 0) or 0) + int(raw or 0)
            except Exception:
                continue
    if current_cache_creation:
        current["cache_creation"] = current_cache_creation

    if "cost_usd" in usage_payload and usage_payload.get("cost_usd") is not None:
        try:
            current["cost_usd"] = float(current.get("cost_usd", 0.0) or 0.0) + float(usage_payload.get("cost_usd") or 0.0)
        except Exception:
            pass

    requests = usage_payload.get("requests")
    try:
        requests_int = int(requests) if requests is not None else int(default_requests)
    except Exception:
        requests_int = int(default_requests)
    current["requests"] = int(current.get("requests", 0) or 0) + max(requests_int, 0)

    return current

#: How much of a tool's output the activity row carries. The row exists so the
#: reader (and whoever is debugging a turn) can see WHAT came back; the whole
#: file is already on disk and in the agent's context, so the row shows the head
#: of it and says how much it stood for.
TOOL_RESULT_PREVIEW_CHARS = 1600


def extract_tool_results_from_claude_event(value: Any) -> list[dict[str, Any]]:
    """``[{"tool_use_id", "text", "is_error", "truncated", "total_chars"}, …]``
    for the tool results a ``user`` event reports back.

    Claude Code narrates its own tool calls by feeding the results into its
    conversation as `user` events. They are not the agent's answer — but they
    are exactly what a reader wants to see as activity, and what a debugger
    needs when a turn goes sideways."""
    if not isinstance(value, dict) or str(value.get("type") or "") != "user":
        return []
    message = value.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or str(block.get("type") or "") != "tool_result":
            continue
        raw = block.get("content")
        text = raw if isinstance(raw, str) else _plain_text_of_blocks(raw)
        total = len(text)
        preview = text[:TOOL_RESULT_PREVIEW_CHARS]
        out.append({
            "tool_use_id": str(block.get("tool_use_id") or ""),
            "text": preview,
            "truncated": total > len(preview),
            "total_chars": total,
            "is_error": bool(block.get("is_error")),
        })
    return out


def _plain_text_of_blocks(value: Any) -> str:
    """The text of a tool result given as blocks rather than a string."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


#: How a tool call reads in the activity row. A signature like
#: ``Bash(command=<252 chars>, description=…)`` tells a reader nothing about what
#: the agent just did; ``Bash · git status --porcelain`` does. One line per tool
#: family, keyed by the argument that carries the meaning.
_TOOL_SUBJECT_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bash", ("command",)),
    ("read", ("file_path", "path", "notebook_path")),
    ("write", ("file_path", "path")),
    ("edit", ("file_path", "path")),
    ("glob", ("pattern", "path")),
    ("grep", ("pattern", "query")),
    ("webfetch", ("url",)),
    ("websearch", ("query",)),
    ("task", ("description", "prompt")),
)
TOOL_TITLE_SUBJECT_CHARS = 96


def claude_tool_activity_title(name: str, args: Any) -> str:
    """``<tool> · <what it acted on>`` for one tool call.

    Reads as a sentence in the Steps list — the file, the command, the pattern —
    instead of a truncated argument dump. An MCP tool keeps its server so two
    services publishing the same tool name stay distinguishable."""
    label = str(name or "tool").strip() or "tool"
    if label.startswith("mcp__"):
        parts = label.split("__")
        label = " · ".join(part for part in parts[1:] if part) or label
    args = args if isinstance(args, dict) else {}
    subject = ""
    key = label.split("·")[0].strip().lower()
    for family, keys in _TOOL_SUBJECT_KEYS:
        if key == family:
            for candidate in keys:
                value = args.get(candidate)
                if isinstance(value, str) and value.strip():
                    subject = value.strip()
                    break
            break
    if not subject:
        for value in args.values():
            if isinstance(value, str) and value.strip():
                subject = value.strip()
                break
    if not subject:
        return label
    subject = " ".join(subject.split())
    if len(subject) > TOOL_TITLE_SUBJECT_CHARS:
        subject = subject[: TOOL_TITLE_SUBJECT_CHARS - 1] + "…"
    return f"{label} · {subject}"
