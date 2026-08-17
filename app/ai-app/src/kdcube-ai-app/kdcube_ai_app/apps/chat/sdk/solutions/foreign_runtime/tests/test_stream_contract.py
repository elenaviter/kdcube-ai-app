# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""The generic stream-adapter pieces (foreign_runtime/stream_contract.py).

Expectations copied from how the ported bundle's stream adapters use these
helpers: content normalization over string / list-of-blocks shapes, and the
tool-call Steps rendering (signature title + arguments body; empty args
stated, never rendered like a healthy call).
"""
from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.stream_contract import (
    content_text,
    tool_call_views,
    tool_result_view,
)


# ── content_text ─────────────────────────────────────────────────────────────

def test_content_text_passes_a_plain_string_through() -> None:
    assert content_text("hello") == "hello"
    assert content_text("") == ""


def test_content_text_joins_list_of_blocks() -> None:
    # Newer chat models stream `content` as a LIST of content blocks.
    blocks = [
        {"type": "text", "text": "Hello, "},
        {"type": "text", "text": "world"},
        {"type": "tool_use", "id": "t1"},  # no "text" -> skipped
        "!",
    ]
    assert content_text(blocks) == "Hello, world!"


def test_content_text_non_text_shapes_yield_empty() -> None:
    assert content_text(None) == ""
    assert content_text({"text": "x"}) == ""
    assert content_text(42) == ""
    assert content_text([{"type": "image", "data": "..."}]) == ""


# ── tool_call_views ──────────────────────────────────────────────────────────

def test_tool_call_views_signature_and_args_body() -> None:
    code = "print('x')\n" * 200
    title, md = tool_call_views("run_python", {"code": code, "prog_name": "news"})
    # Signature: long values as size, short ones inline.
    assert title.startswith("run_python(code=<")
    assert "prog_name='news'" in title
    # Body: the code as a fenced python block, truncated with the total stated.
    assert "```python" in md
    assert "chars total" in md


def test_tool_call_views_empty_args_are_stated() -> None:
    t_empty, md_empty = tool_call_views("run_python", {})
    assert t_empty == "run_python()"
    assert "No arguments received" in md_empty
    t_none, md_none = tool_call_views("run_python", None)
    assert t_none == "run_python()"
    assert "No arguments received" in md_none


def test_tool_call_views_short_args_render_inline_json() -> None:
    title, md = tool_call_views("web_search", {"query": "kdcube", "limit": 3})
    assert title == "web_search(query='kdcube', limit=3)"
    assert "```json" in md
    assert '"query": "kdcube"' in md


def test_tool_call_views_collection_values_render_as_sizes() -> None:
    title, _md = tool_call_views("t", {"items": [1, 2, 3], "opts": {"a": 1}})
    assert "items=<list:3>" in title
    assert "opts=<dict:1>" in title


def test_tool_call_views_signature_line_is_capped() -> None:
    args = {f"k{i}": f"value-{i}" for i in range(40)}
    title, _md = tool_call_views("many", args)
    assert len(title) <= 160
    assert title.endswith("…)")


# ── tool_result_view ─────────────────────────────────────────────────────────

def test_tool_result_view_summarizes_json_text_mcp_success() -> None:
    result = {
        "content": [{
            "type": "text",
            "text": (
                '{"ok": true, "status": 200, "ret": {"items": ['
                '{"label": "elena.viter @ NestLogic", "object_ref": "slack:account"}'
                '], "extra": {"kind": "accounts", "count": 1}}}'
            ),
        }],
        "is_error": False,
    }

    md = tool_result_view(result)

    assert "**Tool result:** ok" in md
    assert "Items: 1" in md
    assert "elena.viter @ NestLogic" in md
    assert '"content"' not in md


def test_tool_result_view_summarizes_direct_error_payload() -> None:
    md = tool_result_view({
        "ok": False,
        "status": 400,
        "error": {
            "code": "file_path_required",
            "message": "Pass the conv:fi artifact path as file_path.",
        },
    })

    assert "**Tool result:** action needed" in md
    assert "`file_path_required`" in md
    assert "conv:fi artifact path" in md
