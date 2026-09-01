# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Web tool declarations for the productivity MCP surface.

Web search and web fetch as plain MCP tools. Unlike the connected-account
tools on this surface, they need no account and declare no claims: the
service side runs on the platform's search and fetch backends.

Two operator-owned properties, both read from the surface config
(``surfaces.as_provider.mcp.productivity.web`` in the descriptor):

  allowlist         list of domain entries; when set, search results from
                    hosts outside it are dropped before any content fetch,
                    and fetching a host outside it is denied with a result
                    naming the host and the allowlist. ``allowlist_file``
                    points at a file instead (one entry per line, re-read
                    on change). Unset = every host allowed.
  use_llm_default   default for the tools' ``use_llm`` parameter (False
                    when absent). With use_llm=false the pipeline runs
                    without LLM steps: no snippet relevance scoring, no
                    LLM content refinement; search and fetch still work.

Entry format: ``example.org`` (domain and subdomains), ``*.example.org``
(subdomains only). See ``tools/backends/web/allowlist.py``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable, Mapping, Optional

from pydantic import Field

import kdcube_ai_app.apps.chat.sdk.tools.backends.web.search_backends as search_backends
import kdcube_ai_app.apps.chat.sdk.tools.backends.web.fetch_backends as fetch_backends
from kdcube_ai_app.apps.chat.sdk.tools.backends.web.allowlist import (
    Allowlist,
    hostname_allowed,
)
from connection_hub.mcp_metadata import read_only_annotations

ConfigFactory = Callable[[], Mapping[str, Any]]

# Same declaration shape as the connected-account tools, with no
# connections: these tools require no account and raise no consent.
WEB_PRODUCTIVITY_TOOLS: dict[str, dict[str, Any]] = {
    "productivity_web_search": {
        "label": "Web search",
        "description": "Search the web; the operator may restrict results to a domain allowlist.",
        "connections": {},
    },
    "productivity_web_fetch": {
        "label": "Web fetch",
        "description": "Fetch pages from URLs already known; the operator may restrict hosts to a domain allowlist.",
        "connections": {},
    },
}

_SERVICE = None


async def _get_service():
    global _SERVICE
    if _SERVICE is None:
        from kdcube_ai_app.infra.service_hub.inventory import (
            _build_model_service_from_env,
        )

        _SERVICE = await _build_model_service_from_env()
    return _SERVICE


def _web_config(config_factory: ConfigFactory) -> Mapping[str, Any]:
    cfg = dict(config_factory() or {})
    web = cfg.get("web") or {}
    return web if isinstance(web, Mapping) else {}


def _allowlist_from_config(web_cfg: Mapping[str, Any]) -> Allowlist:
    file_path = str(web_cfg.get("allowlist_file") or "") or None
    entries = web_cfg.get("allowlist")
    env_value = None
    if isinstance(entries, (list, tuple)):
        env_value = ",".join(str(e) for e in entries)
    allowlist = Allowlist(file_path=file_path, env_value=env_value)
    allowlist.refresh()
    return allowlist


def _ok(ret: Any) -> dict[str, Any]:
    return {"ok": True, "error": None, "ret": ret}


def _error(*, code: str, message: str, where: str, ret: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {"code": code, "message": message, "where": where},
        "ret": ret,
    }


def _normalize_urls(urls: str | list[str]) -> list[str]:
    if isinstance(urls, str):
        raw = urls.strip()
        if raw.startswith("["):
            try:
                return [str(u).strip() for u in json.loads(raw) if str(u).strip()]
            except Exception:
                return [raw]
        return [raw] if raw else []
    return [str(u).strip() for u in urls if str(u).strip()]


def register_web_tools(
    *,
    mcp: Any,
    tool_annotations_type: Any,
    config_factory: ConfigFactory,
) -> None:
    """Register web search and fetch on the productivity MCP surface."""

    @mcp.tool(
        name="productivity_web_search",
        title="Web search",
        description=(
            "Web discovery tool (multi-query). Finds and deduplicates pages "
            "across query variants. With use_llm=true and an objective, snippet "
            "relevance to the objective is scored (0..1) and clearly irrelevant "
            "results may be dropped; with use_llm=false the pipeline runs "
            "without LLM steps and results come ranked by the search backend.\n"
            "Use when you need to FIND pages. For known URLs only, use "
            "productivity_web_fetch.\n"
            "When the operator configures a domain allowlist, results from "
            "hosts outside it are dropped server-side before any content "
            "fetch; the allowlist is the operator's config, a call cannot "
            "widen it.\n"
            "Returns an envelope {ok, error, ret}; ret is an array of results "
            "[{title, url, text, objective_relevance?, query_relevance?, "
            "content?, mime?, base64?, size_bytes?, ...dates}]. `text` is the "
            "search preview snippet; `content` is full fetched page text when "
            "fetch_content ran. Non-HTML supported files return mime/base64 "
            "instead of content."
        ),
        annotations=read_only_annotations(tool_annotations_type, title="Web search"),
        structured_output=False,
    )
    async def _web_search(
        queries: Annotated[
            str | list[str],
            Field(
                description=(
                    "Array of string queries (rephrases/synonyms) or a single "
                    "query string. Query results might be large; prefer max 2 "
                    "queries at a time."
                )
            ),
        ],
        objective: Annotated[
            Optional[str],
            Field(
                description=(
                    "Optional search objective (goal/question). With "
                    "use_llm=true it drives snippet relevance scoring and "
                    "content refinement."
                )
            ),
        ] = None,
        n: Annotated[
            int,
            Field(ge=1, le=20, description="Max unique results (1-20). Prefer max 5."),
        ] = 8,
        fetch_content: Annotated[
            bool,
            Field(
                description=(
                    "If true, fetch full page content for the results. If "
                    "false, return ranked snippets/URLs only (no content "
                    "attribute) — cheaper, and you can fetch selected URLs "
                    "yourself with productivity_web_fetch."
                )
            ),
        ] = True,
        freshness: Annotated[
            Optional[str],
            Field(description="Freshness window: 'day'|'week'|'month'|'year' or null."),
        ] = None,
        country: Annotated[
            Optional[str],
            Field(description="Country ISO2 for the search, e.g. 'DE', 'US'."),
        ] = None,
        safesearch: Annotated[
            str,
            Field(description="Safesearch: 'off'|'moderate'|'strict'."),
        ] = "moderate",
        use_llm: Annotated[
            Optional[bool],
            Field(
                description=(
                    "True adds LLM steps (snippet relevance scoring, content "
                    "refinement). Default comes from the operator's config."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        web_cfg = _web_config(config_factory)
        allowlist = _allowlist_from_config(web_cfg)
        effective_use_llm = (
            bool(web_cfg.get("use_llm_default", False)) if use_llm is None else bool(use_llm)
        )
        try:
            svc = await _get_service() if effective_use_llm else None
            rows = await search_backends.web_search(
                _SERVICE=svc,
                queries=queries,
                objective=objective,
                refinement="balanced" if effective_use_llm else "none",
                n=n,
                fetch_content=fetch_content,
                include_binary_base64=True,
                freshness=freshness,
                country=country,
                safesearch=safesearch,
                use_llm=effective_use_llm,
                allowed_domains=allowlist.entries if allowlist.configured else None,
            )
            cleaned = []
            for r in rows or []:
                if isinstance(r, dict):
                    r.pop("provider", None)
                    cleaned.append({k: v for k, v in r.items() if v is not None})
            return _ok(cleaned)
        except Exception as e:
            return _error(
                code=type(e).__name__,
                message=str(e).strip() or "web_search failed",
                where="productivity_web.web_search",
                ret=[],
            )

    @mcp.tool(
        name="productivity_web_fetch",
        title="Web fetch",
        description=(
            "Fetch-only URL dereferencer (no search). Returns main text plus "
            "status and date metadata for each URL.\n"
            "TOOL SELECTION RULES:\n"
            "- Use ONLY when you already have concrete HTTP/HTTPS URLs.\n"
            "- Never performs search or discovery; to FIND pages use "
            "productivity_web_search.\n"
            "- Skip URLs whose productivity_web_search row already has usable "
            "`content`; fetch only URLs whose content is missing or "
            "insufficient.\n"
            "When the operator configures a domain allowlist, a URL on a host "
            "outside it is denied: its result row carries status "
            "'denied_by_allowlist' and names the host and the allowlist. Other "
            "URLs in the same call still fetch; URLs are never silently "
            "dropped.\n"
            "Returns an envelope {ok, error, ret}; ret is an array of results "
            "[{url, title?, text?, content?, mime?, base64?, size_bytes?, "
            "status, error?, ...dates}] — same row shape as "
            "productivity_web_search."
        ),
        annotations=read_only_annotations(tool_annotations_type, title="Web fetch"),
        structured_output=False,
    )
    async def _web_fetch(
        urls: Annotated[
            str | list[str],
            Field(
                description=(
                    "Array of absolute HTTP/HTTPS URLs you already know, or a "
                    "single URL string."
                )
            ),
        ],
        max_content_length: Annotated[
            int,
            Field(
                description=(
                    "Max characters of cleaned content to keep per URL; longer "
                    "pages are truncated at a sentence boundary. -1 = no limit."
                )
            ),
        ] = -1,
    ) -> dict[str, Any]:
        web_cfg = _web_config(config_factory)
        allowlist = _allowlist_from_config(web_cfg)
        url_list = _normalize_urls(urls)
        if not url_list:
            return _error(
                code="ValueError",
                message="no valid URLs given",
                where="productivity_web.web_fetch",
                ret=[],
            )

        items: list[dict[str, Any]] = []
        allowed: list[str] = []
        source, entries = allowlist.describe()
        from urllib.parse import urlsplit

        for url in url_list:
            host = urlsplit(url).hostname
            if allowlist.check(host):
                allowed.append(url)
            else:
                items.append(
                    {
                        "url": url,
                        "status": "denied_by_allowlist",
                        "error": (
                            f"host '{host}' is outside the allowlist ({source}); "
                            "the operator owns this config"
                        ),
                    }
                )

        try:
            fetched: dict[str, Any] = {}
            if allowed:
                fetched = await fetch_backends.fetch_url_contents(
                    _SERVICE=None,
                    urls=json.dumps(allowed),
                    max_content_length=max_content_length,
                    # an archive mirror is a different host, so with an
                    # allowlist configured the fallback stays off
                    use_archive_fallback=not allowlist.configured,
                    include_binary_base64=True,
                    refinement="none",
                )
            for url, row in (fetched or {}).items():
                if not isinstance(row, dict):
                    continue
                item: dict[str, Any] = {"url": url}
                title = (row.get("title") or row.get("name") or "").strip()
                if title:
                    item["title"] = title
                text = (row.get("content") or row.get("text") or "").strip()
                if text:
                    item["text"] = text
                    item["content"] = text
                for key in (
                    "mime",
                    "base64",
                    "size_bytes",
                    "content_length",
                    "status",
                    "error",
                    "source_type",
                    "fetched_time_iso",
                    "published_time_iso",
                    "modified_time_iso",
                ):
                    if row.get(key) is not None:
                        item[key] = row.get(key)
                items.append(item)
            return _ok(items)
        except Exception as e:
            return _error(
                code=type(e).__name__,
                message=str(e).strip() or "web_fetch failed",
                where="productivity_web.web_fetch",
                ret=items,
            )
