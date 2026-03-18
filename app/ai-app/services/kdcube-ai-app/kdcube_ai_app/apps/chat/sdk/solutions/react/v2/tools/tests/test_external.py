# SPDX-License-Identifier: MIT

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.proto import RuntimeCtx
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.external import handle_external_tool
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.tests.helpers import FakeBrowser, FakeReact


class _FakeExecStreamer:
    def __init__(self, code: str):
        self._code = code
        self.subsystem_language = "python"

    def get_code(self):
        return self._code

    def set_code(self, code: str):
        self._code = code


@pytest.mark.asyncio
async def test_external_exec_path_rewrite_notice(monkeypatch, tmp_path):
    runtime = RuntimeCtx(turn_id="turn_exec", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {"last_decision": {"tool_call": {"tool_id": "exec_tools.execute_code_python", "params": {
        "contract": [{"filename": "turn_exec/files/out.txt", "description": "test output"}],
        "prog_name": "snippet.py",
    }}},
             "outdir": str(tmp_path),
             "workdir": str(tmp_path),
             "exec_code_streamer": _FakeExecStreamer("open('files/x.txt').read()")}

    captured = {}

    async def _fake_execute_tool(**kwargs):
        captured["params"] = kwargs["tool_execution_context"]["params"]
        return {"items": []}

    monkeypatch.setattr("kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.external.execute_tool", _fake_execute_tool)

    react = FakeReact()
    react.tools_subsystem = None

    await handle_external_tool(react=react, ctx_browser=ctx, state=state, tool_call_id="e1")
    assert "turn_exec/files/x.txt" in captured["params"]["code"]


@pytest.mark.asyncio
async def test_rendering_tool_accepts_generic_outdir_fi_path(monkeypatch, tmp_path):
    runtime = RuntimeCtx(turn_id="turn_exec", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    state = {
        "last_decision": {
            "tool_call": {
                "tool_id": "rendering_tools.write_pdf",
                "params": {"path": "fi:logs/out.pdf", "content": "<html><body>x</body></html>"},
            }
        },
        "outdir": str(tmp_path),
        "workdir": str(tmp_path),
    }
    captured = {}

    async def _fake_execute_tool(**kwargs):
        captured["params"] = kwargs["tool_execution_context"]["params"]
        outdir = kwargs["outdir"]
        target = outdir / kwargs["tool_execution_context"]["params"]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"%PDF-1.4\n")
        return {"output": kwargs["tool_execution_context"]["params"]["path"], "summary": ""}

    monkeypatch.setattr("kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.external.execute_tool", _fake_execute_tool)

    react = FakeReact()
    react.tools_subsystem = None

    await handle_external_tool(react=react, ctx_browser=ctx, state=state, tool_call_id="e2")

    assert captured["params"]["path"] == "logs/out.pdf"
    assert any(
        "\"artifact_path\": \"fi:logs/out.pdf\"" in (b.get("text") or "")
        for b in ctx.timeline.blocks
        if b.get("type") == "react.tool.result" and b.get("mime") == "application/json"
    )
    assert not any(
        b.get("type") == "react.notice" and "path_rewritten" in (b.get("text") or "")
        for b in ctx.timeline.blocks
    )
