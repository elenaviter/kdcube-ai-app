# SPDX-License-Identifier: MIT

import json

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.react.proto import RuntimeCtx
from kdcube_ai_app.apps.chat.sdk.solutions.react.events import block_event_id, block_event_source_id
from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.turn_view import (
    extract_assistant_files_from_blocks,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.write import handle_react_write
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.tests.helpers import FakeBrowser, FakeReact
from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.patch import TOOL_SPEC as PATCH_TOOL_SPEC
from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.write import TOOL_SPEC as WRITE_TOOL_SPEC


def test_write_and_patch_tool_specs_expose_create_once_vs_in_place_edit_contract():
    write_purpose = str(WRITE_TOOL_SPEC.get("purpose") or "")
    patch_purpose = str(PATCH_TOOL_SPEC.get("purpose") or "")

    assert "path may be created by react.write once" in write_purpose
    assert "Intentionally edit" in patch_purpose
    assert "preserving its path" in patch_purpose
    assert "instead of calling react.write again" in patch_purpose


@pytest.mark.asyncio
async def test_write_blocks_resolve_tool_event_identity_when_pipeline_enabled(tmp_path):
    runtime = RuntimeCtx(
        turn_id="turn_cur",
        outdir=str(tmp_path),
        workdir=str(tmp_path),
        event_source_pipeline_enabled=True,
    )
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {
        "path": "turn_cur/files/report.md",
        "channel": "canvas",
        "content": "# Report",
        "kind": "display",
    }}}, "outdir": str(tmp_path)}

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c_evt")

    occurrence_blocks = [
        b for b in ctx.timeline.blocks
        if b.get("call_id") == "c_evt" and b.get("type") != "react.notice"
    ]
    assert occurrence_blocks
    assert all("event_source_id" not in b for b in occurrence_blocks)
    assert all("event_id" not in b for b in occurrence_blocks)
    call_meta = {"c_evt": {"tool_id": "react.write"}}
    assert all(block_event_source_id(b, call_meta=call_meta) == "react.write" for b in occurrence_blocks)
    assert all(block_event_id(b) == "c_evt" for b in occurrence_blocks)


@pytest.mark.asyncio
async def test_write_rewrites_old_turn_path_and_notice(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {"path": "turn_old/files/draft.md", "content": "hi", "kind": "display"}}},
             "outdir": str(tmp_path)}

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c1")

    assert any(b.get("type") == "react.notice" for b in ctx.timeline.blocks)
    assert any("turn_cur/files/draft.md" in (b.get("text") or "") for b in ctx.timeline.blocks)


@pytest.mark.asyncio
async def test_write_rewrites_logical_path(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {"path": "conv:fi:turn_old.files/draft.md", "content": "hi", "kind": "display"}}},
             "outdir": str(tmp_path)}

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c1")

    assert any(b.get("type") == "react.notice" for b in ctx.timeline.blocks)
    assert any("turn_cur/files/draft.md" in (b.get("text") or "") for b in ctx.timeline.blocks)


@pytest.mark.asyncio
async def test_write_internal_channel_creates_internal_file_by_default(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {"path": "turn_cur/files/note.md", "channel": "internal", "content": "keep this", "kind": "file"}}},
             "outdir": str(tmp_path)}

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c2")

    note_blocks = [b for b in ctx.timeline.blocks if b.get("type") == "react.note"]
    assert not note_blocks, "internal channel should not inline file content unless scratchpad=true"
    assert (tmp_path / "workdir" / "turn_cur" / "files" / "note.md").read_text() == "keep this"
    assert any("\"visibility\": \"internal\"" in (b.get("text") or "") for b in ctx.timeline.blocks)
    assert any("\"kind\": \"file\"" in (b.get("text") or "") for b in ctx.timeline.blocks)


@pytest.mark.asyncio
async def test_write_internal_scratchpad_creates_note_block(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {"path": "turn_cur/files/note.md", "channel": "internal", "content": "keep this", "kind": "display", "scratchpad": True}}},
             "outdir": str(tmp_path)}

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c2")

    note_blocks = [b for b in ctx.timeline.blocks if b.get("type") == "react.note"]
    assert note_blocks, "internal scratchpad writes should create react.note block"
    assert any((b.get("meta") or {}).get("channel") == "internal" for b in note_blocks)
    assert any((b.get("text") or "") == "keep this" for b in note_blocks)


@pytest.mark.asyncio
async def test_write_rejects_generic_outdir_fi_path(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"params": {"path": "fi:logs/docker.err.log", "content": "hi", "kind": "display"}}},
             "outdir": str(tmp_path)}

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c3")

    assert state["error"]["error"] == "invalid_write_logical_path"


@pytest.mark.asyncio
async def test_write_resolves_ref_content_before_materializing(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    source_path = "conv:fi:turn_prev.files/b1-german-knowledge.mmd"
    # ref:fi bindings consume materialized bytes, so the source must be present on disk
    # (produced this turn, or pulled). Materialize it and point the block at its physical_path.
    source_rel = "turn_prev/files/b1-german-knowledge.mmd"
    source_file = tmp_path / "workdir" / "turn_prev" / "files" / "b1-german-knowledge.mmd"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("graph TD\nA-->B\n")
    ctx.timeline.blocks.append({
        "type": "react.tool.result",
        "turn_id": "turn_prev",
        "path": source_path,
        "mime": "text/markdown",
        "text": "graph TD\nA-->B\n",
        "meta": {"physical_path": source_rel},
    })
    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "path": "turn_cur/files/b1-german-knowledge-resent.mmd",
                    "channel": "canvas",
                    "content": f"ref:{source_path}",
                    "kind": "display",
                }
            }
        },
        "outdir": str(tmp_path),
    }

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c4")

    out_file = tmp_path / "workdir" / "turn_cur" / "files" / "b1-german-knowledge-resent.mmd"
    assert out_file.read_text() == "graph TD\nA-->B\n"
    result_blocks = [b for b in ctx.timeline.blocks if b.get("path") == "conv:fi:turn_cur.files/b1-german-knowledge-resent.mmd"]
    assert any((b.get("text") or "") == "graph TD\nA-->B\n" for b in result_blocks)


@pytest.mark.asyncio
async def test_write_relative_files_path_stays_in_single_files_namespace(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "path": "files/demo_proj/README.md",
                    "channel": "canvas",
                    "content": "# Demo\n",
                    "kind": "file",
                }
            }
        },
        "outdir": str(tmp_path),
    }

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c5")

    assert (tmp_path / "workdir" / "turn_cur" / "files" / "demo_proj" / "README.md").read_text() == "# Demo\n"
    assert not (tmp_path / "workdir" / "turn_cur" / "files" / "files" / "demo_proj" / "README.md").exists()


@pytest.mark.asyncio
async def test_write_rejects_removed_outputs_namespace(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "path": "outputs/demo_proj/test_results.txt",
                    "channel": "canvas",
                    "content": "all tests passed\n",
                    "kind": "file",
                }
            }
        },
        "outdir": str(tmp_path),
    }

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c6")

    assert state["error"]["error"] == "unsafe_path"
    assert not (tmp_path / "workdir" / "turn_cur" / "files" / "outputs").exists()


@pytest.mark.asyncio
async def test_write_unqualified_path_defaults_to_files_namespace(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "path": "demo_proj/report.md",
                    "channel": "canvas",
                    "content": "# Report\n",
                    "kind": "file",
                }
            }
        },
        "outdir": str(tmp_path),
    }

    await handle_react_write(react=FakeReact(), ctx_browser=ctx, state=state, tool_call_id="c7")

    assert (tmp_path / "workdir" / "turn_cur" / "files" / "demo_proj" / "report.md").read_text() == "# Report\n"
    assert any(b.get("path") == "conv:fi:turn_cur.files/demo_proj/report.md" for b in ctx.timeline.blocks)


@pytest.mark.asyncio
async def test_write_same_path_is_rejected_and_keeps_original_version(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"outdir": str(tmp_path)}

    def _decision(content: str) -> dict:
        return {
            "tool_call": {
                "params": {
                    "path": "turn_cur/files/report.md",
                    "channel": "canvas",
                    "content": content,
                    "kind": "file",
                }
            }
        }

    state["last_decision"] = _decision("first version\n")
    await handle_react_write(
        react=FakeReact(),
        ctx_browser=ctx,
        state=state,
        tool_call_id="write_first",
    )
    state["last_decision"] = _decision("corrected final version\n")
    await handle_react_write(
        react=FakeReact(),
        ctx_browser=ctx,
        state=state,
        tool_call_id="write_second",
    )

    path = tmp_path / "workdir" / "turn_cur" / "files" / "report.md"
    assert path.read_text() == "first version\n"
    assert state["retry_decision"] is True

    artifact_ref = "conv:fi:turn_cur.files/report.md"
    resolved = ctx.timeline.resolve_artifact(artifact_ref)
    assert resolved is not None
    assert resolved["text"] == "first version\n"
    assert resolved["tool_call_id"] == "write_first"

    write_meta = []
    for block in ctx.timeline.blocks:
        if block.get("mime") != "application/json":
            continue
        try:
            meta = json.loads(block.get("text") or "")
        except json.JSONDecodeError:
            continue
        if meta.get("artifact_path") == artifact_ref:
            write_meta.append(meta)
    assert [row["tool_call_id"] for row in write_meta] == ["write_first"]
    assert any(
        block.get("type") == "react.notice"
        and "protocol_violation.write_path_already_exists" in str(block.get("text") or "")
        and block.get("call_id") == "write_second"
        for block in ctx.timeline.blocks
    )

    assistant_files = extract_assistant_files_from_blocks(ctx.timeline.blocks)
    assert len(assistant_files) == 1
    assert assistant_files[0]["artifact_path"] == artifact_ref
    assert assistant_files[0]["tool_call_id"] == "write_first"


@pytest.mark.asyncio
async def test_write_rejects_materialized_path_missing_from_timeline(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_cur", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {
        "outdir": str(tmp_path),
        "last_decision": {
            "tool_call": {
                "params": {
                    "path": "turn_cur/files/report.md",
                    "channel": "canvas",
                    "content": "replacement\n",
                    "kind": "file",
                }
            }
        },
    }
    path = tmp_path / "workdir" / "turn_cur" / "files" / "report.md"
    path.parent.mkdir(parents=True)
    path.write_text("created by another current-turn tool\n")

    await handle_react_write(
        react=FakeReact(),
        ctx_browser=ctx,
        state=state,
        tool_call_id="write_duplicate",
    )

    assert path.read_text() == "created by another current-turn tool\n"
    assert state["retry_decision"] is True
    assert any(
        "protocol_violation.write_path_already_exists" in str(block.get("text") or "")
        for block in ctx.timeline.blocks
    )
