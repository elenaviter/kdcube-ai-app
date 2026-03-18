# SPDX-License-Identifier: MIT

import json

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.proto import RuntimeCtx
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.search_files import handle_react_search_files
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.tests.helpers import FakeBrowser


def _latest_hits(ctx: FakeBrowser) -> list[dict]:
    blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("mime") == "application/json"
    ]
    assert blocks
    payload = json.loads(blocks[-1]["text"])
    return payload["hits"]


def _latest_payload(ctx: FakeBrowser) -> dict:
    blocks = [
        b for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("mime") == "application/json"
    ]
    assert blocks
    return json.loads(blocks[-1]["text"])


@pytest.mark.asyncio
async def test_search_files_finds_file_under_outdir_root(tmp_path):
    outdir = tmp_path / "out"
    workdir = tmp_path / "work"
    runtime = RuntimeCtx(turn_id="turn_search", outdir=str(outdir), workdir=str(workdir))
    ctx = FakeBrowser(runtime)

    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    target = logs_dir / "docker.err.log"
    target.write_text("boom", encoding="utf-8")

    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "root": "outdir",
                    "name_regex": r"docker\.err\.log$",
                    "max_hits": 5,
                }
            }
        },
        "outdir": str(outdir),
    }

    await handle_react_search_files(ctx_browser=ctx, state=state, tool_call_id="sf1")

    payload = _latest_payload(ctx)
    assert payload["root"] == "outdir"
    assert payload["hits"] == [{
        "path": "logs/docker.err.log",
        "size_bytes": 4,
        "logical_path": "fi:logs/docker.err.log",
    }]


@pytest.mark.asyncio
async def test_search_files_supports_workdir_subdir_root(tmp_path):
    outdir = tmp_path / "out"
    workdir = tmp_path / "work"
    runtime = RuntimeCtx(turn_id="turn_search", outdir=str(outdir), workdir=str(workdir))
    ctx = FakeBrowser(runtime)

    target_dir = workdir / "runtime" / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "docker.err.log"
    target.write_text("boom", encoding="utf-8")

    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "root": "workdir/runtime",
                    "name_regex": r"docker\.err\.log$",
                    "max_hits": 5,
                }
            }
        },
        "outdir": str(outdir),
    }

    await handle_react_search_files(ctx_browser=ctx, state=state, tool_call_id="sf2")

    payload = _latest_payload(ctx)
    assert payload["root"] == "workdir/runtime"
    assert payload["hits"] == [{
        "path": "logs/docker.err.log",
        "size_bytes": 4,
    }]


@pytest.mark.asyncio
async def test_search_files_includes_logical_path_for_turn_files(tmp_path):
    outdir = tmp_path / "out"
    workdir = tmp_path / "work"
    runtime = RuntimeCtx(turn_id="turn_search", outdir=str(outdir), workdir=str(workdir))
    ctx = FakeBrowser(runtime)

    target = outdir / "turn_prev" / "files" / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello", encoding="utf-8")

    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "root": "outdir",
                    "name_regex": r"report\.md$",
                }
            }
        },
        "outdir": str(outdir),
    }

    await handle_react_search_files(ctx_browser=ctx, state=state, tool_call_id="sf_logic")

    hits = _latest_hits(ctx)
    assert hits == [{
        "path": "turn_prev/files/report.md",
        "size_bytes": 5,
        "logical_path": "fi:turn_prev.files/report.md",
    }]


@pytest.mark.asyncio
async def test_search_files_rejects_removed_fi_root_syntax(tmp_path):
    outdir = tmp_path / "out"
    workdir = tmp_path / "work"
    runtime = RuntimeCtx(turn_id="turn_search", outdir=str(outdir), workdir=str(workdir))
    ctx = FakeBrowser(runtime)

    state = {
        "last_decision": {
            "tool_call": {
                "params": {
                    "root": "fi:logs",
                    "name_regex": r"docker\.err\.log$",
                }
            }
        },
        "outdir": str(outdir),
    }

    await handle_react_search_files(ctx_browser=ctx, state=state, tool_call_id="sf3")

    assert state["last_tool_result"] == []
    assert any(b.get("type") == "react.notice" for b in ctx.timeline.blocks)
