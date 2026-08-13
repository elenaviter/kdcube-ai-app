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

from typing import Any, Dict

_SIG_STR_CAP = 48       # inline string preview inside the signature
_SIG_TOTAL_CAP = 160    # whole signature line
_BODY_STR_CAP = 1500    # per-argument body preview
_BODY_INLINE_CAP = 120  # strings up to this render inline in the args list


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
