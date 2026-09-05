"""KDCube Web Search, isolated execution, and rendering tools for LangGraph."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.tool_runtime import (
    DirectToolRuntime,
)
from kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search import web_search_server


def _normalise_results(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    if isinstance(value, dict):
        value = value.get("results") or value.get("items") or []
    rows: list[dict[str, str]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title") or "Untitled"),
                "body": str(
                    item.get("body")
                    or item.get("snippet")
                    or item.get("text")
                    or ""
                ),
                "url": str(item.get("href") or item.get("url") or ""),
            }
        )
    return rows


def build_tools(
    runtime: DirectToolRuntime | None,
    *,
    enabled_ids: set[str] | None = None,
) -> list[BaseTool]:
    """Build tools for one turn; ``runtime`` binds filesystem state to that turn."""

    def require_runtime() -> DirectToolRuntime:
        if runtime is None:
            raise RuntimeError("this tool requires an active direct-agent turn")
        return runtime

    @tool
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search with KDCube Web Search. Returns title, excerpt, and URL rows."""
        limit = max(1, min(int(max_results or 5), 8))
        try:
            rows = _normalise_results(
                await web_search_server.web_search(
                    queries=query,
                    objective=query,
                    refinement="none",
                    n=limit,
                    fetch_content=False,
                    include_binary_base64=False,
                    use_llm=False,
                )
            )
            return json.dumps(
                {"ok": True, "query": query, "results": rows},
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {"ok": False, "query": query, "error": str(exc), "results": []}
            )

    @tool
    async def execute_python(
        code: str,
        artifacts: list[dict[str, Any]],
        program_name: str = "Agent-generated Python",
        timeout_s: int = 600,
    ) -> str:
        """Execute your Python in KDCube's isolated runtime and keep contracted files."""
        active = require_runtime()
        return active.tool_report(
            await active.execute_python(
                code=code,
                artifacts=artifacts,
                program_name=program_name,
                timeout_s=timeout_s,
            )
        )

    @tool
    async def write_pdf(source_path: str, output_path: str, title: str = "") -> str:
        """Render a current-turn HTML file to PDF with KDCube rendering tools."""
        active = require_runtime()
        return active.tool_report(
            await active.write_pdf(
                source_path=source_path,
                output_path=output_path,
                title=title,
            )
        )

    @tool
    async def write_docx(source_path: str, output_path: str, title: str = "") -> str:
        """Render a current-turn Markdown file to DOCX with KDCube rendering tools."""
        active = require_runtime()
        return active.tool_report(
            await active.write_docx(
                source_path=source_path,
                output_path=output_path,
                title=title,
            )
        )

    @tool
    async def write_pptx(source_path: str, output_path: str, title: str = "") -> str:
        """Render current-turn section-based HTML to PPTX with KDCube rendering tools."""
        active = require_runtime()
        return active.tool_report(
            await active.write_pptx(
                source_path=source_path,
                output_path=output_path,
                title=title,
            )
        )

    available = [web_search, execute_python, write_pdf, write_docx, write_pptx]
    if enabled_ids is None:
        return available
    return [item for item in available if item.name in enabled_ids]
