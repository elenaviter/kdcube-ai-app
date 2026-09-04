"""KDCube Web Search adapter and local deliverable tool for native ReAct."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from kdcube_ai_app.apps.chat.sdk.runtime.workdir_discovery import resolve_output_dir
from kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search import web_search_server


def _rows(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [{"title": "Research notes", "body": value, "url": ""}]
    if isinstance(value, dict):
        value = value.get("results") or value.get("items") or [value]
    rows: list[dict[str, str]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title") or "Untitled"),
                "body": str(item.get("body") or item.get("snippet") or item.get("text") or ""),
                "url": str(item.get("href") or item.get("url") or ""),
            }
        )
    return rows


async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search through KDCube Web Search and return compact sourced rows."""
    limit = max(1, min(int(max_results or 5), 8))
    try:
        hits = await web_search_server.web_search(
            queries=query,
            objective=query,
            refinement="none",
            n=limit,
            fetch_content=False,
            include_binary_base64=False,
            use_llm=False,
        )
    except Exception as exc:  # Network/provider failures are tool evidence.
        return {"ok": False, "query": query, "error": str(exc), "results": []}
    return {"ok": True, "query": query, "results": _rows(hits)}


def create_briefing(title: str, summary: str, findings: Any) -> dict[str, Any]:
    """Create research-brief.pdf and research-data.xlsx in the turn output directory."""
    output_dir = Path(resolve_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(findings)
    if not rows:
        rows = [{"title": "Summary", "body": summary, "url": ""}]

    pdf_path = output_dir / "research-brief.pdf"
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 5 * mm)]
    story.extend([Paragraph(escape(summary), styles["BodyText"]), Spacer(1, 5 * mm)])
    for row in rows:
        story.append(Paragraph(escape(row["title"]), styles["Heading2"]))
        story.append(
            Paragraph(escape(row["body"] or "No excerpt supplied."), styles["BodyText"])
        )
        if row["url"]:
            story.append(Paragraph(escape(row["url"]), styles["Code"]))
        story.append(Spacer(1, 3 * mm))
    SimpleDocTemplate(str(pdf_path), pagesize=A4).build(story)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Research"
    sheet.append(["Title", "Finding", "Source"])
    for row in rows:
        sheet.append([row["title"], row["body"], row["url"]])
    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 80
    sheet.column_dimensions["C"].width = 55
    xlsx_path = output_dir / "research-data.xlsx"
    workbook.save(xlsx_path)

    return {
        "ok": True,
        "files": [pdf_path.name, xlsx_path.name],
        "rows": len(rows),
        "output_dir": str(output_dir),
    }


class _Tools:
    web_search = staticmethod(web_search)
    create_briefing = staticmethod(create_briefing)


tools = _Tools()


def list_tools() -> dict[str, dict[str, Any]]:
    return {
        "web_search": {
            "callable": web_search,
            "description": web_search.__doc__,
            "returns": "A structured list of web results.",
        },
        "create_briefing": {
            "callable": create_briefing,
            "description": create_briefing.__doc__,
            "returns": "The generated PDF and XLSX filenames.",
        },
    }
