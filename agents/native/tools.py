"""KDCube Web Search adapter for the direct native ReAct example."""

from __future__ import annotations

from typing import Any
from kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search import web_search_server


def _rows(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
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


async def web_fetch(
    url: str,
    objective: str = "",
    max_content_length: int = 12000,
) -> dict[str, Any]:
    """Fetch one selected web result through KDCube's governed Web Fetch."""
    try:
        fetched = await web_search_server.web_fetch(
            urls=[url],
            objective=objective or None,
            refinement="none",
            max_content_length=max(1000, min(int(max_content_length or 12000), 50000)),
            include_binary_base64=False,
            use_archive_fallback=False,
            use_llm=False,
        )
    except Exception as exc:  # Network/provider failures are tool evidence.
        return {"ok": False, "url": url, "error": str(exc), "result": None}
    return {"ok": True, "url": url, "result": fetched.get(url)}


class _Tools:
    web_search = staticmethod(web_search)
    web_fetch = staticmethod(web_fetch)


tools = _Tools()


def list_tools() -> dict[str, dict[str, Any]]:
    return {
        "web_search": {
            "callable": web_search,
            "description": web_search.__doc__,
            "returns": "A structured list of web results.",
        },
        "web_fetch": {
            "callable": web_fetch,
            "description": web_fetch.__doc__,
            "returns": "The governed content result for one selected URL.",
        },
    }
