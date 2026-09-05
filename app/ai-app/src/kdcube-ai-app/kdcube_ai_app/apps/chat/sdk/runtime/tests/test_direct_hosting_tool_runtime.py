from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting import tool_runtime as module
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.tool_runtime import (
    DirectToolRuntime,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.workspace import (
    DirectTurnWorkspace,
)


class _ToolSubsystem:
    def __init__(self, **kwargs):
        self.comm = kwargs["comm"]
        self.bundle_root = Path(kwargs["bundle_spec"].path)
        self.prebind_for_in_memory = AsyncMock()


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DirectToolRuntime:
    monkeypatch.setattr(module, "ToolSubsystem", _ToolSubsystem)
    return DirectToolRuntime(
        service=object(),
        comm=SimpleNamespace(),
        workspace=DirectTurnWorkspace(tmp_path, "turn_demo"),
        exec_runtime={"mode": "docker", "image": "exec:test"},
        bundle_id="example@1-0",
        bundle_root=tmp_path,
        bundle_module="agent",
    )


def test_direct_turn_workspace_uses_the_canonical_artifact_layout(tmp_path: Path) -> None:
    workspace = DirectTurnWorkspace(tmp_path, "turn_demo")

    assert workspace.current_file("research/data.xlsx") == (
        tmp_path / "turn_demo" / "out" / "workdir" / "turn_demo" / "files" / "research" / "data.xlsx"
    )
    assert workspace.current_attachment("request.md") == (
        tmp_path / "turn_demo" / "out" / "workdir" / "turn_demo" / "attachments" / "request.md"
    )
    with pytest.raises(ValueError, match="canonical turn_"):
        DirectTurnWorkspace(tmp_path, "turn-demo")


@pytest.mark.asyncio
async def test_execute_python_normalizes_paths_and_reports_generated_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    execute = AsyncMock(return_value={"ok": True, "items": [], "report_text": "done"})
    monkeypatch.setattr(module, "run_exec_tool", execute)

    result = await runtime.execute_python(
        code="from pathlib import Path\nPath('files/report.txt').write_text('done')",
        artifacts=[
            {
                "filepath": "files/report.txt",
                "description": "Generated report",
                "visibility": "external",
            }
        ],
        program_name="Report program",
    )

    assert result["ok"] is True
    assert result["generated_source"]["archive_path"] == "pkg/user_code.py"
    assert result["path_rewrites"]["contract"] == [
        {
            "original": "files/report.txt",
            "rewritten": "turn_demo/files/report.txt",
        }
    ]
    call = execute.await_args.kwargs
    assert "turn_demo/files/report.txt" in call["code"]
    assert call["contract"][0]["filepath"] == "turn_demo/files/report.txt"
    runtime.tool_subsystem.prebind_for_in_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_renderer_reads_current_turn_source_and_writes_current_turn_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    source = runtime.workspace.current_file("research/brief.html")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("<html><body><h1>Brief</h1></body></html>", encoding="utf-8")

    from kdcube_ai_app.apps.chat.sdk.runtime.workdir_discovery import resolve_output_dir
    from kdcube_ai_app.apps.chat.sdk.tools import rendering_tools

    async def render_pdf(*, path: str, content: str, **_kwargs):
        assert "<h1>Brief</h1>" in content
        output = resolve_output_dir() / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"%PDF-direct-runtime")
        return {"ok": True, "error": None}

    monkeypatch.setattr(rendering_tools.tools, "write_pdf", render_pdf)

    result = await runtime.write_pdf(
        source_path="files/research/brief.html",
        output_path="files/research/brief.pdf",
        title="Research brief",
    )

    assert result["ok"] is True
    assert result["source_path"] == "turn_demo/files/research/brief.html"
    assert result["output_path"] == "turn_demo/files/research/brief.pdf"
    assert runtime.workspace.current_file("research/brief.pdf").read_bytes().startswith(
        b"%PDF-"
    )


@pytest.mark.asyncio
async def test_renderer_enforces_source_format_per_document_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, monkeypatch)

    result = await runtime.write_docx(
        source_path="files/research/brief.html",
        output_path="files/research/brief.docx",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_render_path"
    assert "Markdown" not in result["error"]["message"]
    assert ".md" in result["error"]["message"]
