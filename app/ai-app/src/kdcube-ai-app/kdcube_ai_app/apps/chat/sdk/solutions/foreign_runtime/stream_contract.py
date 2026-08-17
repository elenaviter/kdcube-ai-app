# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── stream_contract.py ── the generic pieces of a stream adapter ──
#
# Stream adapters are runtime-specific (a looping ReAct model node streams
# differently from a linear graph's dedicated answer node), but two pieces are
# pure and shared by every adapter:
#
#   * ``content_text``    — normalize a message chunk's ``content`` to text
#     (newer chat models stream ``content`` as a LIST of content blocks).
#   * ``tool_call_views`` — render one tool invocation as a Steps row: a compact
#     call signature title + an arguments body. A step row that says just
#     "run_python / running" hides the one thing that matters when a call
#     misbehaves: WHAT the model actually passed. Large string values (e.g. a
#     program text) render as their own fenced block, truncated. Empty
#     arguments are stated explicitly: a call that arrives with NO usable args
#     (e.g. truncated upstream) must be visible as such, not rendered like a
#     healthy call.
#
# Pure functions; no framework imports.

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

_SIG_STR_CAP = 48       # inline string preview inside the signature
_SIG_TOTAL_CAP = 160    # whole signature line
_BODY_STR_CAP = 1500    # per-argument body preview
_BODY_INLINE_CAP = 120  # strings up to this render inline in the args list
_RESULT_STR_CAP = 1800


def _sig_value(value: Any) -> str:
    if isinstance(value, str):
        flat = " ".join(value.split())
        if len(flat) <= _SIG_STR_CAP:
            return repr(flat)
        return f"<{len(value):,} chars>"
    if isinstance(value, (list, tuple)):
        return f"<list:{len(value)}>"
    if isinstance(value, dict):
        return f"<dict:{len(value)}>"
    return repr(value)


def tool_call_views(name: str, args: Any) -> tuple[str, str]:
    """(title, markdown) for one tool invocation: a compact signature line and
    an arguments body the Steps row expands to."""
    if not isinstance(args, dict) or not args:
        return f"{name}()", "_No arguments received._"
    parts = []
    for key, value in args.items():
        parts.append(f"{key}={_sig_value(value)}")
    signature = f"{name}({', '.join(parts)})"
    if len(signature) > _SIG_TOTAL_CAP:
        signature = signature[: _SIG_TOTAL_CAP - 2] + "…)"

    inline: Dict[str, Any] = {}
    blocks: list[str] = []
    for key, value in args.items():
        if isinstance(value, str) and (len(value) > _BODY_INLINE_CAP or "\n" in value):
            shown = value[:_BODY_STR_CAP]
            tail = f"\n… ({len(value):,} chars total)" if len(value) > _BODY_STR_CAP else ""
            lang = "python" if key == "code" else ""
            blocks.append(f"**{key}**\n```{lang}\n{shown}\n```{tail}")
        else:
            inline[key] = value
    md_parts: list[str] = []
    if inline:
        try:
            import json
            md_parts.append("```json\n" + json.dumps(inline, ensure_ascii=False, default=str, indent=2)[:_BODY_STR_CAP] + "\n```")
        except Exception:
            md_parts.append("```\n" + str(inline)[:_BODY_STR_CAP] + "\n```")
    md_parts.extend(blocks)
    return signature, "\n\n".join(md_parts)


def _extract_tool_result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if content is not None:
        text = content_text(content)
        if text:
            return text
    if isinstance(result, Mapping):
        content_value = result.get("content")
        if isinstance(content_value, list):
            text = content_text(content_value)
            if text:
                return text
        try:
            return json.dumps(dict(result), ensure_ascii=False, default=str)
        except Exception:
            return str(result)
    return str(result)


def _payload_from_result(result: Any) -> Mapping[str, Any] | None:
    if isinstance(result, Mapping):
        payload = _payload_from_content_blocks(result.get("content"))
        if payload is not None:
            return payload
        if _looks_like_result_payload(result):
            return result
    content = getattr(result, "content", None)
    if content is not None:
        payload = _payload_from_content_blocks(content)
        if payload is not None:
            return payload
    return _parse_result_payload(_extract_tool_result_text(result))


def _looks_like_result_payload(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "ok",
            "ret",
            "error",
            "consent",
            "missing_grants",
            "code",
        )
    )


def _payload_from_content_blocks(content: Any) -> Mapping[str, Any] | None:
    if not isinstance(content, list) or len(content) != 1:
        return None
    block = content[0]
    text = ""
    if isinstance(block, Mapping):
        text = str(block.get("text") or "")
    else:
        text = str(getattr(block, "text", "") or "")
    return _parse_result_payload(text)


def _parse_result_payload(text: str) -> Mapping[str, Any] | None:
    raw = str(text or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _payload_error_code(payload: Mapping[str, Any]) -> str:
    err = payload.get("error")
    if isinstance(err, Mapping):
        return str(err.get("code") or err.get("error") or "").strip()
    return str(payload.get("code") or err or "").strip()


def _payload_message(payload: Mapping[str, Any]) -> str:
    err = payload.get("error")
    if isinstance(err, Mapping):
        for key in ("message", "error_description", "description"):
            value = str(err.get(key) or "").strip()
            if value:
                return value
    for key in ("message", "error_description", "description"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _payload_consent(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    consent = payload.get("consent")
    if isinstance(consent, Mapping):
        return consent
    err = payload.get("error")
    details = err.get("details") if isinstance(err, Mapping) and isinstance(err.get("details"), Mapping) else {}
    consent = details.get("consent") if isinstance(details, Mapping) else None
    return consent if isinstance(consent, Mapping) else {}


def _payload_ok_summary(payload: Mapping[str, Any]) -> str:
    lines = ["**Tool result:** ok"]
    status = payload.get("status")
    if status is not None:
        lines.append(f"- Status: `{status}`")

    ret = payload.get("ret")
    if isinstance(ret, Mapping):
        items = ret.get("items")
        if isinstance(items, list):
            lines.append(f"- Items: {len(items)}")
            for item in items[:3]:
                label = _item_label(item)
                if label:
                    lines.append(f"  - {label}")
            if len(items) > 3:
                lines.append(f"  - ... {len(items) - 3} more")
        extra = ret.get("extra")
        if isinstance(extra, Mapping):
            kind = str(extra.get("kind") or "").strip()
            count = extra.get("count")
            if kind:
                lines.append(f"- Kind: `{kind}`")
            if count is not None and not isinstance(items, list):
                lines.append(f"- Count: `{count}`")
        obj = ret.get("object")
        if isinstance(obj, Mapping):
            label = _item_label(obj)
            if label:
                lines.append(f"- Object: {label}")

    message = _payload_message(payload)
    if message:
        lines.append(f"- Message: {message}")
    return "\n".join(lines)


def _item_label(item: Any) -> str:
    if not isinstance(item, Mapping):
        return str(item or "").strip()
    for key in (
        "label",
        "display_name",
        "name",
        "title",
        "email",
        "object_ref",
        "ref",
        "id",
    ):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _as_display_list(value: Any, *, cap: int = 8) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value if str(item or "").strip()]
    else:
        items = []
    items = items[:cap]
    return items


def tool_result_view(result: Any) -> str:
    """Markdown summary for a completed tool step.

    The start row already shows the arguments. The completion row must also
    show the result when it carries a failure/consent reason; otherwise the
    Steps tab says only that a tool completed and hides the actionable cause.
    """
    payload = _payload_from_result(result)
    if payload is not None:
        code = _payload_error_code(payload)
        consent = _payload_consent(payload)
        ok = payload.get("ok")
        is_error = payload.get("is_error")
        if ok is False or is_error is True or code or consent:
            claims = (
                payload.get("missing_grants")
                or payload.get("claims")
                or consent.get("claims")
                or consent.get("account_claim")
                or []
            )
            candidates = consent.get("candidates")
            candidate_labels: list[str] = []
            if isinstance(candidates, list):
                for item in candidates[:5]:
                    if isinstance(item, Mapping):
                        label = str(item.get("label") or item.get("email") or item.get("account_id") or "").strip()
                        if label:
                            candidate_labels.append(label)
            lines = ["**Tool result:** action needed"]
            if code:
                lines.append(f"- Code: `{code}`")
            message = _payload_message(payload)
            if message:
                lines.append(f"- Message: {message}")
            display_claims = _as_display_list(claims)
            if display_claims:
                lines.append("- Claims: " + ", ".join(f"`{claim}`" for claim in display_claims))
            if candidate_labels:
                lines.append("- Candidate accounts: " + ", ".join(candidate_labels))
            action_label = str(consent.get("action_label") or "").strip()
            if action_label:
                lines.append(f"- Action: {action_label}")
            return "\n".join(lines)
        if ok is True or "ret" in payload:
            return _payload_ok_summary(payload)

    text = _extract_tool_result_text(result)
    if not text:
        return ""
    shown = text.strip()
    if len(shown) > _RESULT_STR_CAP:
        shown = shown[:_RESULT_STR_CAP] + f"\n... ({len(text):,} chars total)"
    return "**Tool result:**\n```text\n" + shown + "\n```"


def content_text(content: Any) -> str:
    """Normalize a streamed message chunk's ``content`` to text.

    Newer chat models (e.g. OpenAI's Responses API) stream ``content`` as a LIST
    of content blocks, not a plain str — so ``answer += chunk.content`` would
    raise ``TypeError: can only concatenate str (not "list") to str``, and a
    ``str(content)`` fallback would render the raw block list. Join the text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""
