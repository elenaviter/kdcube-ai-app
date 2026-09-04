"""Web research and briefing tools for the direct LangGraph example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from ddgs import DDGS
from langchain_core.tools import BaseTool, tool
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


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
                "body": str(item.get("body") or item.get("snippet") or item.get("text") or ""),
                "url": str(item.get("href") or item.get("url") or ""),
            }
        )
    return rows


def build_tools(output_dir: Path) -> list[BaseTool]:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the public web. Returns JSON rows containing title, excerpt, and URL."""
        limit = max(1, min(int(max_results or 5), 8))
        try:
            rows = _normalise_results(list(DDGS().text(query, max_results=limit)))
            return json.dumps({"ok": True, "query": query, "results": rows}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"ok": False, "query": query, "error": str(exc), "results": []})

    @tool
    def create_briefing(title: str, summary: str, findings_json: str) -> str:
        """Create research-brief.pdf and research-data.xlsx from findings supplied as JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = _normalise_results(findings_json)
        if not rows:
            rows = [{"title": "Summary", "body": summary, "url": ""}]

        pdf_path = output_dir / "research-brief.pdf"
        styles = getSampleStyleSheet()
        story = [
            Paragraph(escape(title), styles["Title"]),
            Spacer(1, 5 * mm),
            Paragraph(escape(summary), styles["BodyText"]),
        ]
        for row in rows:
            story.extend(
                [
                    Spacer(1, 4 * mm),
                    Paragraph(escape(row["title"]), styles["Heading2"]),
                    Paragraph(
                        escape(row["body"] or "No excerpt supplied."),
                        styles["BodyText"],
                    ),
                ]
            )
            if row["url"]:
                story.append(Paragraph(escape(row["url"]), styles["Code"]))
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
        return json.dumps(
            {"ok": True, "files": [pdf_path.name, xlsx_path.name], "rows": len(rows)}
        )

    return [web_search, create_briefing]
