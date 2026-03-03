# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

from typing import Any, Dict, List

import json
import pathlib

from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.common import (
    tool_call_block,
    notice_block,
    add_block,
    tc_result_path,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools_.search_files import search_files
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.solution_workspace import _safe_relpath

TOOL_SPEC = {
    "id": "react.search_files",
    "purpose": "Search local files under OUT_DIR (default) or workdir for filenames or content regexes.",
    "args": {
        "root": "optional root selector: outdir (default), fi:<relpath>, or workdir/wd:<relpath>",
        "name_regex": "optional regex for file name",
        "content_regex": "optional regex for file content",
        "max_files": "int limit",
        "max_bytes": "int per file",
        "max_hits": "int limit",
    },
    "returns": "list of matching file paths",
}


async def handle_react_search_files(*, ctx_browser: Any, state: Dict[str, Any], tool_call_id: str) -> Dict[str, Any]:
    last_decision = state.get("last_decision") or {}
    tool_call = last_decision.get("tool_call") or {}
    tool_id = "react.search_files"
    params = tool_call.get("params") or {}
    root_sel = (params.get("root") or "").strip()
    name_regex = params.get("name_regex")
    content_regex = params.get("content_regex")
    max_files = int(params.get("max_files") or 2000)
    max_bytes = int(params.get("max_bytes") or 1_000_000)
    max_hits = int(params.get("max_hits") or 200)

    turn_id = (ctx_browser.runtime_ctx.turn_id or "")
    tool_call_block(
        ctx_browser=ctx_browser,
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        payload={
            "tool_id": tool_id,
            "tool_call_id": tool_call_id,
            "params": tool_call.get("params") or {},
        },
    )

    outdir = pathlib.Path(state["outdir"])
    root_dir = outdir
    if root_sel:
        if root_sel.startswith("fi:"):
            rel = root_sel[len("fi:"):].lstrip("/")
            if rel:
                if not _safe_relpath(rel):
                    notice_block(
                        ctx_browser=ctx_browser,
                        tool_call_id=tool_call_id,
                        code="search_files_invalid_root",
                        message="invalid fi: root selector.",
                        extra={"tool_id": tool_id, "root": root_sel},
                        rel="result",
                    )
                    state["last_tool_result"] = []
                    return state
                root_dir = outdir / rel
        elif root_sel.lower() in {"workdir", "wd", "work"} or root_sel.startswith(("wd:", "workdir:")):
            workdir_raw = getattr(getattr(ctx_browser, "runtime_ctx", None), "workdir", "") or ""
            if not workdir_raw:
                notice_block(
                    ctx_browser=ctx_browser,
                    tool_call_id=tool_call_id,
                    code="search_files_no_workdir",
                    message="workdir is not configured.",
                    extra={"tool_id": tool_id},
                    rel="result",
                )
                state["last_tool_result"] = []
                return state
            rel = ""
            if root_sel.startswith("wd:"):
                rel = root_sel[len("wd:"):].lstrip("/")
            elif root_sel.startswith("workdir:"):
                rel = root_sel[len("workdir:"):].lstrip("/")
            root_dir = pathlib.Path(workdir_raw)
            if rel:
                if not _safe_relpath(rel):
                    notice_block(
                        ctx_browser=ctx_browser,
                        tool_call_id=tool_call_id,
                        code="search_files_invalid_root",
                        message="invalid workdir root selector.",
                        extra={"tool_id": tool_id, "root": root_sel},
                        rel="result",
                    )
                    state["last_tool_result"] = []
                    return state
                root_dir = root_dir / rel
    try:
        hits = search_files(
            root=str(root_dir),
            name_regex=name_regex,
            content_regex=content_regex,
            max_files=max_files,
            max_bytes=max_bytes,
            max_hits=max_hits,
        )
    except Exception as exc:
        notice_block(
            ctx_browser=ctx_browser,
            tool_call_id=tool_call_id,
            code="search_files_failed",
            message=f"search_files failed: {exc}",
            extra={"tool_id": tool_id},
            rel="result",
        )
        state["last_tool_result"] = []
        return state

    add_block(ctx_browser, {
        "turn": turn_id,
        "type": "react.tool.result",
        "call_id": tool_call_id,
        "mime": "application/json",
        "path": tc_result_path(turn_id=turn_id, call_id=tool_call_id),
        "text": json.dumps({"hits": hits}, ensure_ascii=False, indent=2),
        "meta": {
            "tool_call_id": tool_call_id,
        },
    })
    state["last_tool_result"] = hits
    return state
