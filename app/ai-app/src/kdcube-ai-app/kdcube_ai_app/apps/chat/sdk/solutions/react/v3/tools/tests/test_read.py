# SPDX-License-Identifier: MIT

import pytest
import json
import random

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import artifact_outdir_for
from kdcube_ai_app.apps.chat.sdk.solutions.react.proto import RuntimeCtx
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools import read as v2_read
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.read import handle_react_read
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.tests.helpers import FakeBrowser


@pytest.mark.asyncio
async def test_read_missing_paths_notice(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {"paths": ["conv:fi:turn_read.files/missing.md"]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r1")
    assert any(b.get("type") == "react.notice" for b in ctx.timeline.blocks)


@pytest.mark.asyncio
async def test_read_requires_pull_for_external_namespace_ref(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {"paths": ["demo:item-1"]}}}}

    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_owner")

    assert any(
        block.get("type") == "react.notice" and "read_paths_missing" in (block.get("text") or "")
        for block in ctx.timeline.blocks
    )
    assert not any(
        block.get("type") == "react.tool.result" and block.get("path") == "demo:item-1"
        for block in ctx.timeline.blocks
    )


@pytest.mark.asyncio
async def test_read_returns_latest_version(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    path = "conv:fi:turn_read.files/report.md"
    # older version
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "application/json",
        "text": '{"artifact_path":"conv:fi:turn_read.files/report.md","physical_path":"turn_read/files/report.md"}',
        "turn_id": "turn_read",
    })
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "text/markdown",
        "path": path,
        "text": "old",
        "turn_id": "turn_read",
    })
    # newer version
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "application/json",
        "text": '{"artifact_path":"conv:fi:turn_read.files/report.md","physical_path":"turn_read/files/report.md"}',
        "turn_id": "turn_read",
    })
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "text/markdown",
        "path": path,
        "text": "new",
        "turn_id": "turn_read",
    })
    state = {"last_decision": {"tool_call": {"params": {"paths": [path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r2")
    assert any(b.get("text") == "new" for b in ctx.timeline.blocks if b.get("type") == "react.tool.result")


@pytest.mark.asyncio
async def test_read_supports_current_turn_file_refs(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    logs_dir = artifact_outdir_for(tmp_path) / "turn_read" / "files" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "docker.err.log").write_text("boom", encoding="utf-8")

    source_path = "conv:fi:turn_read.files/logs/docker.err.log"
    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r3")

    assert any(
        b.get("path") == source_path and b.get("text") == "boom"
        for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
    )


@pytest.mark.asyncio
async def test_read_external_followup_svg_attachment_as_text(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    physical = (
        artifact_outdir_for(tmp_path)
        / "turn_read"
        / "external"
        / "external_event"
        / "attachments"
        / "evt_1"
        / "diagram.svg"
    )
    physical.parent.mkdir(parents=True, exist_ok=True)
    physical.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>diagram</text></svg>', encoding="utf-8")
    path = "conv:fi:turn_read.external.external_event.attachments/evt_1/diagram.svg"
    state = {"last_decision": {"tool_call": {"params": {"paths": [path]}}}}

    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_svg")

    assert any(
        b.get("path") == path and "<svg" in (b.get("text") or "") and "diagram" in (b.get("text") or "")
        for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
    )


@pytest.mark.asyncio
async def test_read_tc_result_prefers_inline_payload_over_meta(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    source_path = "conv:tc:turn_src.pref1.result"

    ctx.timeline.blocks.extend([
        {
            "type": "react.tool.result",
            "mime": "application/json",
            "path": source_path,
            "text": (
                '{"artifact_path":"conv:tc:turn_src.pref1.result","mime":"application/json",'
                '"kind":"file","visibility":"internal","tool_call_id":"pref1"}'
            ),
            "turn_id": "turn_src",
            "call_id": "pref1",
            "meta": {"tool_call_id": "pref1"},
        },
        {
            "type": "react.tool.result",
            "mime": "application/json",
            "path": source_path,
            "text": (
                '{"ok": true, "current": {"location": {"value": "Wuppertal"}}, '
                '"summary": "Current preferences:\\n- location: Wuppertal"}'
            ),
            "turn_id": "turn_src",
            "call_id": "pref1",
            "meta": {"tool_call_id": "pref1"},
        },
    ])

    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r4")

    assert any(
        b.get("call_id") == "r4"
        and b.get("path") == source_path
        and '"location": {"value": "Wuppertal"}' in (b.get("text") or "")
        for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
    )


@pytest.mark.asyncio
async def test_read_sources_pool_prefers_fetched_content_over_preview(tmp_path):
    runtime = RuntimeCtx(
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        max_tokens=80_000,
        read_visible_max_text_symbols=120,
    )
    ctx = FakeBrowser(runtime)
    full_content = "full fetched article body with CVE evidence " * 40
    ctx.timeline.sources_pool = [
        {
            "sid": 1,
            "url": "https://example.com/security",
            "title": "Security Article",
            "text": "short search preview only",
            "content": full_content,
            "content_length": len(full_content),
        }
    ]

    state = {"last_decision": {"tool_call": {"params": {"paths": ["conv:so:sources_pool[1]"]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_source")

    block = next(
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
        and b.get("path") == "conv:so:sources_pool[1]"
        and b.get("call_id") == "r_source"
    )
    assert block.get("mime") == "application/json"
    rows = json.loads(block.get("text") or "[]")
    assert rows[0]["text"] == "short search preview only"
    assert rows[0]["content"] == full_content
    assert rows[0]["content"].endswith("CVE evidence ")

    status_block = next(
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
        and b.get("path") == "conv:tc:turn_read.r_source.result"
        and b.get("call_id") == "r_source"
    )
    status = json.loads(status_block["text"])
    assert status["paths"][0]["kind"] == "sources_pool"
    assert status["paths"][0]["items_stats"]["content_rows"] == 1
    assert status["paths"][0]["content_policy"] == "full_source_rows"


@pytest.mark.asyncio
async def test_read_sources_pool_max_text_symbols_preserves_json_items(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path), max_tokens=80_000)
    ctx = FakeBrowser(runtime)
    ctx.timeline.sources_pool = [
        {"sid": 1, "url": "https://example.com/1", "title": "One", "text": "snippet one", "content": "A" * 100},
        {"sid": 2, "url": "https://example.com/2", "title": "Two", "text": "snippet two", "content": "B" * 100},
    ]

    state = {"last_decision": {"tool_call": {"params": {"paths": ["conv:so:sources_pool[1,2]"], "max_text_symbols": 50}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_source_cap")

    block = next(
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
        and b.get("path") == "conv:so:sources_pool[1,2]"
        and b.get("call_id") == "r_source_cap"
    )
    rows = json.loads(block.get("text") or "[]")
    assert [r["sid"] for r in rows] == [1, 2]
    assert rows[0]["content"] == "A" * 50
    assert rows[0]["content_truncated_for_visible_context"] is True
    assert rows[1]["content"] == ""
    assert rows[1]["content_truncated_for_visible_context"] is True


@pytest.mark.asyncio
async def test_read_cross_conversation_sources_pool_uses_ctx_client(tmp_path):
    class FakeCtxClient:
        def __init__(self):
            self.calls = []

        async def fetch_conversation_sources_pool(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "conversation_id": kwargs["conversation_id"],
                "sources_pool": [
                    {"sid": 1, "url": "https://example.com/one", "title": "One", "content": "first"},
                    {"sid": 2, "url": "https://example.com/two", "title": "Two", "content": "second"},
                ],
            }

    runtime = RuntimeCtx(
        user_id="user-1",
        conversation_id="current-conv",
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        max_tokens=80_000,
    )
    ctx = FakeBrowser(runtime)
    ctx.ctx_client = FakeCtxClient()
    path = "conv:so:conv_other-conv.sources_pool[2]"

    state = {"last_decision": {"tool_call": {"params": {"paths": [path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_cross_source")

    assert ctx.ctx_client.calls
    assert ctx.ctx_client.calls[0]["user_id"] == "user-1"
    assert ctx.ctx_client.calls[0]["conversation_id"] == "other-conv"

    block = next(
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
        and b.get("path") == path
        and b.get("call_id") == "r_cross_source"
    )
    rows = json.loads(block.get("text") or "[]")
    assert rows == [{"sid": 2, "url": "https://example.com/two", "title": "Two", "content": "second"}]
    assert block["meta"]["source_conversation_id"] == "other-conv"

    status = next(
        json.loads(b["text"])
        for b in ctx.timeline.blocks
        if b.get("path") == "conv:tc:turn_read.r_cross_source.result"
    )
    assert status["paths"][0]["conversation_id"] == "other-conv"


@pytest.mark.asyncio
async def test_read_large_tc_result_returns_configured_preview_not_full_payload(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path), max_tokens=80_000)
    ctx = FakeBrowser(runtime)
    source_path = "conv:tc:turn_src.tc_big.result"
    large_text = json.dumps({"ok": True, "messages": ["email body " * 2000 for _ in range(80)]})

    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "application/json",
        "path": source_path,
        "text": large_text,
        "turn_id": "turn_src",
        "call_id": "tc_big",
        "meta": {"tool_call_id": "tc_big", "tool_id": "email.process_user_emails"},
    })

    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_large")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_large"
    ]
    assert not any(b.get("text") == large_text for b in read_blocks)
    preview = next(b for b in read_blocks if b.get("path") == source_path)
    assert "READ PREVIEW TRUNCATED" in preview["text"]
    assert "react.read stats_only" in preview["text"]
    assert "continuation: if shape metadata is missing use react.read stats_only" in preview["text"]

    status = next(
        json.loads(b["text"])
        for b in read_blocks
        if b.get("path") == "conv:tc:turn_read.r_large.result" and b.get("mime") == "application/json"
    )
    assert status["paths"][0]["status"] == "truncated_for_visible_context"
    assert status["paths"][0]["recovery"]["strategy"] == "bounded_inspection"
    assert status["paths"][0]["recovery"]["use"] == [
        "react.rg then its bounded read_items",
        "react.read bounded line or symbol items",
        "programmatic inspection of the materialized physical file into small derived artifacts",
    ]
    assert status["truncated_paths"][0]["path"] == source_path


@pytest.mark.asyncio
async def test_read_ev_event_path(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path), max_tokens=80_000)
    ctx = FakeBrowser(runtime)
    event_path = "conv:ev:turn_src.events/task-tracker/canvas/review/evt_1"
    ctx.timeline.blocks.append({
        "type": "event.external",
        "mime": "application/json",
        "path": event_path,
        "text": json.dumps({
            "event_id": "evt_1",
            "event_source_id": "task_tracker.canvas.review.requested",
            "ok": True,
            "ret": {"prompt": "Review selected text"},
        }),
        "turn_id": "turn_src",
        "meta": {
            "event_id": "evt_1",
            "event_source_id": "task_tracker.canvas.review.requested",
            "event_type": "event.external",
        },
    })

    state = {"last_decision": {"tool_call": {"params": {"paths": [event_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_ev")

    assert any(
        b.get("type") == "react.tool.result"
        and b.get("call_id") == "r_ev"
        and b.get("path") == event_path
        and "Review selected text" in (b.get("text") or "")
        for b in ctx.timeline.blocks
    )


@pytest.mark.asyncio
async def test_read_mid_sized_payload_uses_configured_preview(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path), max_tokens=80_000)
    ctx = FakeBrowser(runtime)
    source_path = "conv:tc:turn_src.tc_mid.result"
    raw_text = "email body with enough detail\n" * 9000
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "text/plain",
        "path": source_path,
        "text": raw_text,
        "turn_id": "turn_src",
        "call_id": "tc_mid",
        "meta": {"tool_call_id": "tc_mid", "tool_id": "exec_tools.execute_code_python"},
    })

    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_mid_large")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_mid_large"
    ]
    assert not any(b.get("text") == raw_text for b in read_blocks)
    preview = next(b for b in read_blocks if b.get("path") == source_path)
    assert "READ PREVIEW TRUNCATED" in preview["text"]


@pytest.mark.asyncio
async def test_read_large_payload_with_max_text_symbols_returns_bounded_preview(tmp_path):
    runtime = RuntimeCtx(
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        max_tokens=80_000,
        read_visible_max_text_symbols=1_200,
        read_visible_max_tokens=800,
    )
    ctx = FakeBrowser(runtime)
    source_path = "conv:tc:turn_src.tc_preview.result"
    raw_text = "0123456789abcdef\n" * 1000
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "text/plain",
        "path": source_path,
        "text": raw_text,
        "turn_id": "turn_src",
        "call_id": "tc_preview",
        "meta": {"tool_call_id": "tc_preview", "tool_id": "email.process_user_emails"},
    })

    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path], "max_text_symbols": 500}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_preview")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_preview"
    ]
    preview = next(b for b in read_blocks if b.get("path") == source_path)
    assert "READ PREVIEW TRUNCATED" in preview["text"]
    assert "omitted_text_symbols:" in preview["text"]
    assert raw_text not in preview["text"]
    status = next(
        json.loads(b["text"])
        for b in read_blocks
        if b.get("path") == "conv:tc:turn_read.r_preview.result" and b.get("mime") == "application/json"
    )
    assert status["paths"][0]["status"] == "truncated_for_visible_context"
    assert status["requested_text_symbols"] == 500


@pytest.mark.asyncio
async def test_read_text_payload_honors_raw_byte_cap(tmp_path):
    runtime = RuntimeCtx(
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        max_tokens=80_000,
        read_visible_max_text_symbols=10_000,
        read_visible_max_tokens=4_000,
        read_visible_max_bytes=64,
    )
    ctx = FakeBrowser(runtime)
    source_path = "conv:tc:turn_src.tc_bytes.result"
    raw_text = "0123456789abcdef" * 20
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "text/plain",
        "path": source_path,
        "text": raw_text,
        "turn_id": "turn_src",
        "call_id": "tc_bytes",
        "meta": {"tool_call_id": "tc_bytes", "tool_id": "email.process_user_emails"},
    })

    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_bytes")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_bytes"
    ]
    marker = next(b for b in read_blocks if b.get("path") == source_path)
    assert "READ PREVIEW TRUNCATED" in marker["text"]
    assert "bytes:" in marker["text"]
    status = next(
        json.loads(b["text"])
        for b in read_blocks
        if b.get("path") == "conv:tc:turn_read.r_bytes.result" and b.get("mime") == "application/json"
    )
    assert status["paths"][0]["status"] == "truncated_for_visible_context"
    assert status["paths"][0]["bytes"] > status["visible_read_limit_bytes"]
    assert status["truncated_paths"][0]["bytes"] > status["visible_read_limit_bytes"]


@pytest.mark.asyncio
async def test_read_large_text_file_returns_configured_preview(tmp_path):
    runtime = RuntimeCtx(
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        max_tokens=80_000,
        read_visible_max_text_symbols=120,
        read_visible_max_tokens=200,
        read_visible_max_bytes=512,
    )
    ctx = FakeBrowser(runtime)
    logs_dir = artifact_outdir_for(tmp_path) / "turn_read" / "files" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    full_text = "large file line with details\n" * 200
    (logs_dir / "large.txt").write_text(full_text, encoding="utf-8")

    source_path = "conv:fi:turn_read.files/logs/large.txt"
    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_file_large")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_file_large"
    ]
    preview = next(b for b in read_blocks if b.get("path") == source_path)
    assert "READ PREVIEW TRUNCATED" in preview["text"]
    assert "large file line" in preview["text"]
    assert full_text not in preview["text"]
    status = next(
        json.loads(b["text"])
        for b in read_blocks
        if b.get("path") == "conv:tc:turn_read.r_file_large.result" and b.get("mime") == "application/json"
    )
    assert status["paths"][0]["status"] == "truncated_for_visible_context"
    assert status["paths"][0]["bytes"] > status["visible_read_limit_bytes"]


@pytest.mark.asyncio
async def test_read_stats_only_returns_metadata_without_materializing_tc_text(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path), max_tokens=80_000)
    ctx = FakeBrowser(runtime)
    source_path = "conv:tc:turn_src.tc_stats.result"
    raw_text = json.dumps({"ok": True, "messages": ["email body " * 100 for _ in range(3)]})
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "mime": "application/json",
        "path": source_path,
        "text": raw_text,
        "turn_id": "turn_src",
        "call_id": "tc_stats",
        "meta": {"tool_call_id": "tc_stats", "tool_id": "email.process_user_emails"},
    })

    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path], "stats_only": True}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_stats")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_stats"
    ]
    assert not any(b.get("path") == source_path for b in read_blocks)
    status = json.loads(next(
        b["text"]
        for b in read_blocks
        if b.get("path") == "conv:tc:turn_read.r_stats.result" and b.get("mime") == "application/json"
    ))
    assert status["stats_only"] is True
    assert status["paths"][0]["path"] == source_path
    assert status["paths"][0]["status"] == "stats_only"
    assert status["paths"][0]["mime"] == "application/json"
    assert status["paths"][0]["tokens"] > 0
    assert status["paths"][0]["text_symbols"] == len(raw_text)
    assert status["paths"][0]["bytes"] == len(raw_text.encode("utf-8"))


@pytest.mark.asyncio
async def test_read_stats_only_marks_surfaced_ten_line_large_object_for_bounded_inspection(tmp_path):
    runtime = RuntimeCtx(
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        max_tokens=80_000,
        read_visible_max_text_symbols=48_000,
        read_visible_max_tokens=12_000,
    )
    ctx = FakeBrowser(runtime)
    logs_dir = artifact_outdir_for(tmp_path) / "turn_read" / "files" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Same decisive shape as the surfaced canvas object: very few lines, but
    # almost all content in one line. ASCII keeps byte and symbol counts equal.
    raw_text = ("meta\n" * 8) + ("B" * 270) + "\n" + ("A" * 131_835)
    assert len(raw_text) == 132_146
    (logs_dir / "long-line.txt").write_text(raw_text, encoding="utf-8")
    source_path = "conv:fi:turn_read.files/logs/long-line.txt"

    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path], "stats_only": True}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_long_line_stats")

    status = json.loads(next(
        block["text"]
        for block in ctx.timeline.blocks
        if block.get("path") == "conv:tc:turn_read.r_long_line_stats.result"
        and block.get("mime") == "application/json"
    ))
    stats = status["paths"][0]
    assert stats["fits_visible_context"] is False
    assert "text_symbols" in stats["over_limit_dimensions"]
    assert stats["text_symbols"] == 132_146
    assert stats["bytes"] == 132_146
    assert stats["tokens"] > 0
    assert stats["token_count_kind"] == "chunk_sum_estimate"
    assert stats["line_count"] == 10
    assert stats["max_line_text_symbols"] == 131_835
    assert stats["recommended_inspection"] == (
        "react.rg_or_bounded_line_or_symbol_items_or_programmatic_file_inspection"
    )
    assert "read_items" not in stats


@pytest.mark.asyncio
async def test_read_stats_only_returns_file_metadata_without_base64(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    logs_dir = artifact_outdir_for(tmp_path) / "turn_read" / "files" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = b"%PDF-1.7\n" + (b"x" * 2048)
    (logs_dir / "report.pdf").write_bytes(payload)

    source_path = "conv:fi:turn_read.files/logs/report.pdf"
    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path], "stats_only": True}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_file_stats")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_file_stats"
    ]
    assert not any(b.get("path") == source_path for b in read_blocks)
    assert not any(b.get("base64") for b in read_blocks)
    status = json.loads(next(
        b["text"]
        for b in read_blocks
        if b.get("path") == "conv:tc:turn_read.r_file_stats.result" and b.get("mime") == "application/json"
    ))
    assert status["stats_only"] is True
    assert status["paths"][0]["path"] == source_path
    assert status["paths"][0]["status"] == "stats_only"
    assert status["paths"][0]["kind"] == "binary"
    assert status["paths"][0]["mime"] == "application/pdf"
    assert status["paths"][0]["bytes"] == len(payload)


@pytest.mark.asyncio
async def test_read_large_image_file_returns_downscaled_multimodal_preview(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    runtime = RuntimeCtx(
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        read_visible_max_bytes=90_000,
    )
    ctx = FakeBrowser(runtime)
    out_file = artifact_outdir_for(tmp_path) / "turn_read" / "files" / "large.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(123)
    width = height = 900
    payload = rng.randbytes(width * height * 3)
    Image.frombytes("RGB", (width, height), payload).save(out_file, "PNG")
    assert out_file.stat().st_size > runtime.read_visible_max_bytes

    source_path = "conv:fi:turn_read.files/large.png"
    state = {"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_img")

    read_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("call_id") == "r_img"
    ]
    image_block = next(b for b in read_blocks if b.get("path") == source_path and b.get("base64"))
    assert image_block["mime"] == "image/png"
    assert image_block["meta"]["image_view"]["view_kind"] == "image_downscaled"
    assert image_block["meta"]["image_view"]["original_size_bytes"] == out_file.stat().st_size
    assert image_block["meta"]["image_view"]["visible_size_bytes"] <= runtime.read_visible_max_bytes

    status = json.loads(next(
        b["text"]
        for b in read_blocks
        if b.get("path") == "conv:tc:turn_read.r_img.result" and b.get("mime") == "application/json"
    ))
    assert status["paths"][0]["status"] == "image_downscaled_for_visible_context"
    assert status["paths"][0]["image_view"]["visible_size_bytes"] <= runtime.read_visible_max_bytes


@pytest.mark.asyncio
async def test_read_turn_index_reconstructs_semantic_inventory_from_turn_log(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_current", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    ctx._turn_logs["turn_prev"] = {
        "blocks": [
            {
                "type": "conv.working.summary",
                "author": "assistant",
                "turn_id": "turn_prev",
                "ts": "2026-05-05T19:37:00Z",
                "path": "conv:ws:turn_prev.conv.working.summary.attempt.1",
                "text": "Goal: Create ZIP with Anthropic April invoices.\nOutcome: ZIP failed at hosted artifact boundary.",
                "mime": "text/markdown",
                "meta": {"kind": "working_summary", "summary_scope": "completion_attempt"},
            },
            {
                "type": "user.followup",
                "author": "user",
                "turn_id": "turn_prev",
                "ts": "2026-05-05T19:27:38Z",
                "path": "conv:ar:turn_prev.external.followup.msg_1",
                "text": "Retry the Anthropic April invoice ZIP workflow.",
                "meta": {"event_kind": "followup", "message_id": "msg_1", "sequence": 12, "source": "telegram"},
            },
            {
                "type": "react.tool.call",
                "turn_id": "turn_prev",
                "call_id": "tc_scan",
                "path": "conv:tc:turn_prev.tc_scan.call",
                "mime": "application/json",
                "text": json.dumps({"tool_id": "email.process_user_emails", "params": {"query": "Anthropic April invoices"}}),
            },
            {
                "type": "react.tool.result",
                "turn_id": "turn_prev",
                "call_id": "tc_scan",
                "path": "conv:tc:turn_prev.tc_scan.result",
                "mime": "application/json",
                "text": json.dumps({"ok": True, "message": "found 10 Anthropic April emails and current attachment IDs"}),
            },
            {
                "type": "react.tool.result",
                "turn_id": "turn_prev",
                "call_id": "tc_mat",
                "path": "conv:tc:turn_prev.tc_mat.result",
                "mime": "application/json",
                "text": json.dumps({
                    "artifact_path": "conv:fi:turn_prev.files/email-attachments/Invoice_1.pdf",
                    "mime": "application/pdf",
                    "kind": "file",
                    "description": "Anthropic invoice PDF materialized from Gmail",
                }),
                "meta": {"tool_call_id": "tc_mat", "tool_id": "email.materialize_email_attachments"},
            },
            {
                "type": "react.tool.result",
                "turn_id": "turn_prev",
                "call_id": "tc_mat",
                "path": "conv:fi:turn_prev.files/email-attachments/Invoice_1.pdf",
                "mime": "application/pdf",
                "meta": {"tool_call_id": "tc_mat", "tool_id": "email.materialize_email_attachments"},
            },
            {
                "type": "assistant.completion",
                "author": "assistant",
                "turn_id": "turn_prev",
                "ts": "2026-05-05T19:37:19Z",
                "path": "conv:ar:turn_prev.assistant.completion",
                "text": "All 20 Anthropic invoice PDFs were materialized, but ZIP failed. [[S:1]]",
                "meta": {"sources_used": [1]},
            },
        ],
        "sources_pool": [
            {"sid": 1, "title": "Anthropic invoice source", "url": "https://example.test/invoice", "text": "invoice source row"}
        ],
    }

    index_path = "conv:ar:turn_prev.react.turn.index"
    state = {"last_decision": {"tool_call": {"params": {"paths": [index_path]}}}}
    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_index")

    index_block = next(
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("path") == index_path
    )
    text = index_block.get("text") or ""
    assert "[TURN INDEX]" in text
    assert "latest working summary: conv:ws:turn_prev.conv.working.summary" in text
    assert "user followup: conv:ar:turn_prev.external.followup.msg_1" in text
    assert "Retry the Anthropic April invoice ZIP workflow" in text
    assert "tool: email.process_user_emails" in text
    assert "found 10 Anthropic April emails" in text
    assert "conv:fi:turn_prev.files/email-attachments/Invoice_1.pdf" in text
    assert "Anthropic invoice PDF materialized from Gmail" in text
    assert "source: conv:so:sources_pool[1]" in text

    status_block = next(
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("path") == "conv:tc:turn_current.r_index.result"
    )
    status = json.loads(status_block["text"])
    assert status["paths"][0]["path"] == index_path
    assert status["paths"][0]["source_turn_id"] == "turn_prev"


@pytest.mark.asyncio
async def test_read_items_materializes_multiple_line_ranges(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_read", outdir=str(tmp_path), workdir=str(tmp_path), max_tokens=80_000)
    ctx = FakeBrowser(runtime)
    out_file = artifact_outdir_for(tmp_path) / "turn_read" / "files" / "page.html"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join([
        "<html>",
        "<body>",
        "<section id=\"hero\">Hero</section>",
        "<section id=\"pricing\">Pricing</section>",
        "<section id=\"checkout\">Checkout</section>",
        "</body>",
        "</html>",
    ]), encoding="utf-8")

    source_path = "conv:fi:turn_read.files/page.html"
    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "items": [
                        {"path": source_path, "line_start": 3, "line_count": 2},
                        {"path": source_path, "line_start": 5, "line_count": 1},
                    ]
                }
            }
        }
    }

    await handle_react_read(ctx_browser=ctx, state=state, tool_call_id="r_ranges")

    range_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result"
        and b.get("call_id") == "r_ranges"
        and b.get("path") == source_path
        and "[READ RANGE]" in (b.get("text") or "")
    ]
    assert len(range_blocks) == 2
    assert "lines: [3-4]/7" in range_blocks[0]["text"]
    assert "     3\t<section id=\"hero\">Hero</section>" in range_blocks[0]["text"]
    assert "     4\t<section id=\"pricing\">Pricing</section>" in range_blocks[0]["text"]
    assert "lines: [5-5]/7" in range_blocks[1]["text"]

    status = next(
        json.loads(b["text"])
        for b in ctx.timeline.blocks
        if b.get("path") == "conv:tc:turn_read.r_ranges.result" and b.get("mime") == "application/json"
    )
    assert len(status["paths"]) == 2
    assert status["paths"][0]["read_range"]["line_start"] == 3
    assert status["paths"][1]["read_range"]["line_start"] == 5


@pytest.mark.asyncio
async def test_read_truncated_line_range_preserves_source_coordinates_after_full_preview(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(v2_read, "_count_tokens", lambda text: max(1, len(text) // 4))
    runtime = RuntimeCtx(
        turn_id="turn_read",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        max_tokens=80_000,
        read_visible_max_text_symbols=1_200,
        read_visible_max_tokens=800,
        tool_result_preview_max_text_symbols=8_000,
    )
    ctx = FakeBrowser(runtime)
    out_file = artifact_outdir_for(tmp_path) / "turn_read" / "files" / "mixed-card.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    source_lines = [f"source-line-{line_number}" for line_number in range(1, 76)]
    source_lines[74] = ""
    image_prefix = "![pasted image](data:image/png;base64,"
    image_suffix = ")"
    source_lines.append(
        image_prefix + ("A" * (574_907 - len(image_prefix) - len(image_suffix))) + image_suffix
    )
    assert len(source_lines[-1]) == 574_907
    out_file.write_text("\n".join(source_lines), encoding="utf-8")
    source_path = "conv:fi:turn_read.files/mixed-card.md"

    await handle_react_read(
        ctx_browser=ctx,
        state={"last_decision": {"tool_call": {"params": {"paths": [source_path]}}}},
        tool_call_id="r_full_preview",
    )
    await handle_react_read(
        ctx_browser=ctx,
        state={
            "last_decision": {
                "tool_call": {
                    "params": {
                        "items": [{"path": source_path, "line_start": 64, "line_count": 12}]
                    }
                }
            }
        },
        tool_call_id="r_safe_range",
    )
    await handle_react_read(
        ctx_browser=ctx,
        state={
            "last_decision": {
                "tool_call": {
                    "params": {
                        "items": [{"path": source_path, "line_start": 64, "line_count": 13}]
                    }
                }
            }
        },
        tool_call_id="r_truncated_range",
    )
    await handle_react_read(
        ctx_browser=ctx,
        state={
            "last_decision": {
                "tool_call": {
                    "params": {
                        "items": [{"path": source_path, "line_start": 72, "line_count": 10}]
                    }
                }
            }
        },
        tool_call_id="r_past_eof_range",
    )

    safe_range = next(
        block
        for block in ctx.timeline.blocks
        if block.get("type") == "react.tool.result"
        and block.get("call_id") == "r_safe_range"
        and block.get("path") == source_path
    )
    assert "lines: [64-75]/76" in safe_range["text"]
    assert "    64\tsource-line-64" in safe_range["text"]
    assert "    74\tsource-line-74" in safe_range["text"]
    assert "source-line-57" not in safe_range["text"]
    assert "[READ PREVIEW" not in safe_range["text"]

    truncated_range = next(
        block
        for block in ctx.timeline.blocks
        if block.get("type") == "react.tool.result"
        and block.get("call_id") == "r_truncated_range"
        and block.get("path") == source_path
    )
    assert "requested_lines: 64-76" in truncated_range["text"]
    assert "available_lines: [64-76]/76" in truncated_range["text"]
    assert "visible_lines: [64-75]/76" in truncated_range["text"]
    assert "partial_line: 76" in truncated_range["text"]
    assert "    64\tsource-line-64" in truncated_range["text"]
    assert "    75\t\n" in truncated_range["text"]
    assert "    76\t![pasted image](data:image/png;base64," in truncated_range["text"]
    assert "source-line-57" not in truncated_range["text"]
    assert "[READ RANGE TRUNCATED]" in truncated_range["text"]
    assert "[READ PREVIEW" not in truncated_range["text"]

    past_eof_range = next(
        block
        for block in ctx.timeline.blocks
        if block.get("type") == "react.tool.result"
        and block.get("call_id") == "r_past_eof_range"
        and block.get("path") == source_path
    )
    assert "requested_lines: 72-81" in past_eof_range["text"]
    assert "available_lines: [72-76]/76" in past_eof_range["text"]
    assert "visible_lines: [72-75]/76" in past_eof_range["text"]
    assert "partial_line: 76" in past_eof_range["text"]
    assert "/81" not in past_eof_range["text"]

    status = next(
        json.loads(block["text"])
        for block in ctx.timeline.blocks
        if block.get("path") == "conv:tc:turn_read.r_truncated_range.result"
        and block.get("mime") == "application/json"
    )
    assert status["paths"][0]["status"] == "truncated_for_visible_context"
    assert status["paths"][0]["read_range"]["line_start"] == 64
    assert status["paths"][0]["read_range"]["line_end"] == 76
    assert status["paths"][0]["read_range"]["total_line_count"] == 76

    rendered = await ctx.timeline.render(cache_last=False)
    rendered_text = "\n".join(
        block.get("text") or ""
        for block in rendered
        if block.get("type") == "text"
    )
    assert "requested_lines: 64-76" in rendered_text
    assert "visible_lines: [64-75]/76" in rendered_text
    assert rendered_text.rfind("requested_lines: 64-76") > rendered_text.find("[READ PREVIEW]")
