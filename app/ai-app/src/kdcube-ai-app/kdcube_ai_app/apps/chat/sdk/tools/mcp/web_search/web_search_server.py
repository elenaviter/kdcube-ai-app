# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""
MCP server wrapper for web search and fetch.

Supports:
  - stdio (on-demand / local process)
  - http / sse (remote server mode)

Tool implementation delegates to search_backends.web_search and
fetch_backends.fetch_url_contents, with ModelService + cache built from
environment variables.

Two operator-facing properties, both enforced server-side:

  - LLM on/off: every tool takes ``use_llm``; with ``use_llm=false`` the
    pipeline runs without any LLM step (no reconciliation, no LLM content
    filtering) and needs no model API keys.
  - Domain allowlist: configured via WEB_ALLOWLIST_FILE (one entry per
    line, re-read on change) or WEB_ALLOWLIST (comma-separated). When
    configured, search results from hosts outside it are dropped before
    any content fetch, and fetch of a host outside it is denied with a
    result naming the host and the allowlist source. When not configured,
    every host is allowed. See backends.web.allowlist for the format.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase, _build_model_service_from_env
from kdcube_ai_app.infra.service_hub.cache import create_kv_cache_from_env
import kdcube_ai_app.apps.chat.sdk.tools.backends.web.search_backends as search_backends
import kdcube_ai_app.apps.chat.sdk.tools.backends.web.fetch_backends as fetch_backends
from kdcube_ai_app.apps.chat.sdk.tools.backends.web.allowlist import (
    ALLOWLIST_FILE_ENV,
    Allowlist,
)
from kdcube_ai_app.apps.chat.sdk.tools.mcp.mcp_app_transport import run_http, run_sse, run_stdio

_SERVICE: Optional[ModelServiceBase] = None
_CACHE = None
_ALLOWLIST: Optional[Allowlist] = None


async def _get_service() -> ModelServiceBase:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = await _build_model_service_from_env()
    return _SERVICE


def _get_cache() -> Any:
    global _CACHE
    if _CACHE is None:
        _CACHE = create_kv_cache_from_env(ttl_env_var="WEB_SEARCH_CACHE_TTL_SECONDS")
    return _CACHE


def _get_allowlist() -> Allowlist:
    global _ALLOWLIST
    if _ALLOWLIST is None:
        _ALLOWLIST = Allowlist.from_env()
    return _ALLOWLIST


async def web_search(
    queries: str | List[str],
    objective: Optional[str] = None,
    refinement: str = "balanced",
    n: int = 8,
    fetch_content: bool = True,
    include_binary_base64: bool = True,
    freshness: Optional[str] = None,
    country: Optional[str] = None,
    safesearch: str = "moderate",
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    svc = await _get_service() if use_llm else None
    cache = _get_cache()
    allowlist = _get_allowlist()
    return await search_backends.web_search(
        _SERVICE=svc,
        queries=queries,
        objective=objective,
        refinement=refinement,
        n=n,
        fetch_content=fetch_content,
        include_binary_base64=include_binary_base64,
        freshness=freshness,
        country=country,
        safesearch=safesearch,
        use_llm=use_llm,
        allowed_domains=allowlist.entries if allowlist.configured else None,
        namespaced_kv_cache=cache,
    )


async def web_fetch(
    urls: str | List[str],
    objective: Optional[str] = None,
    refinement: str = "none",
    max_content_length: int = -1,
    include_binary_base64: bool = True,
    use_archive_fallback: bool = False,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Fetch URLs; a host outside the configured allowlist is denied with the reason."""
    allowlist = _get_allowlist()
    if isinstance(urls, str):
        raw = urls.strip()
        url_list = [str(u).strip() for u in json.loads(raw)] if raw.startswith("[") else [raw]
    else:
        url_list = [str(u).strip() for u in urls]
    url_list = [u for u in url_list if u]

    out: Dict[str, Any] = {}
    allowed: List[str] = []
    source, entries = allowlist.describe()
    for url in url_list:
        host = urlsplit(url).hostname
        if allowlist.check(host):
            allowed.append(url)
        else:
            out[url] = {
                "status": "denied_by_allowlist",
                "error": (
                    f"host '{host}' is outside the allowlist ({source}); "
                    "the operator owns this config"
                ),
                "allowlist_entries": entries,
            }

    if allowed:
        svc = await _get_service() if use_llm else None
        # An archive mirror is a different host, so it would step around the
        # allowlist; with an allowlist configured the fallback stays off.
        fetched = await fetch_backends.fetch_url_contents(
            _SERVICE=svc,
            urls=json.dumps(allowed),
            max_content_length=max_content_length,
            use_archive_fallback=use_archive_fallback and not allowlist.configured,
            include_binary_base64=include_binary_base64,
            refinement=refinement if use_llm else "none",
            objective=objective,
            namespaced_kv_cache=_get_cache(),
        )
        if isinstance(fetched, dict):
            out.update(fetched)
    return out


async def allowlist_status() -> Dict[str, Any]:
    """The domain allowlist exactly as this server enforces it."""
    source, entries = _get_allowlist().describe()
    return {
        "allowlist_source": source,
        "allowlist_entries": entries,
        "entry_count": len(entries),
        "enforced": _get_allowlist().configured,
    }


def _build_mcp_app():
    try:
        from kdcube_ai_app.apps.chat.sdk.runtime.mcp.server import KDCubeMCPServer
    except Exception as e:  # pragma: no cover - runtime dependency
        raise ImportError("mcp server SDK is not installed") from e

    mcp = KDCubeMCPServer("web_search")

    @mcp.tool(
        name="web_search",
        description=(
            "Web discovery tool (multi-query). Finds and deduplicates pages across "
            "query variants; prefer max 2 queries at a time. With use_llm=true and an "
            "objective, snippet relevance to the objective is scored (0..1) and clearly "
            "irrelevant results may be dropped, and fetched content is refined per the "
            "refinement mode; with use_llm=false the pipeline runs without LLM steps "
            "(no model keys needed) and results come ranked by the search backend.\n"
            "Use when you need to FIND pages. For known URLs only, use web_fetch.\n"
            "Refinement modes (post-fetch, objective-guided, best-effort): 'none' full "
            "pages; 'balanced' target + context (50-70%); 'recall' content bodies, "
            "minimal chrome (80-95%); 'precision' directly relevant sections only "
            "(20-50%, needs objective).\n"
            "When the operator configures a domain allowlist, results from hosts "
            "outside it are dropped server-side before any content fetch; a call "
            "cannot widen the allowlist - see allowlist_status.\n"
            "Returns an array of results [{title, url, text, objective_relevance?, "
            "query_relevance?, content?, mime?, base64?, size_bytes?, ...dates}]. "
            "`text` is the search preview snippet; `content` is full fetched page "
            "text when fetch_content ran. Non-HTML supported files return mime/base64 "
            "instead of content."
        ),
    )
    async def _tool(
        queries: str | List[str],
        objective: Optional[str] = None,
        refinement: str = "balanced",
        n: int = 8,
        fetch_content: bool = True,
        include_binary_base64: bool = True,
        freshness: Optional[str] = None,
        country: Optional[str] = None,
        safesearch: str = "moderate",
        use_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        return await web_search(
            queries=queries,
            objective=objective,
            refinement=refinement,
            n=n,
            fetch_content=fetch_content,
            include_binary_base64=include_binary_base64,
            freshness=freshness,
            country=country,
            safesearch=safesearch,
            use_llm=use_llm,
        )

    @mcp.tool(
        name="web_fetch",
        description=(
            "Fetch-only URL dereferencer (no search). Returns main text plus status "
            "and date metadata for each URL.\n"
            "TOOL SELECTION RULES:\n"
            "- Use ONLY when you already have concrete HTTP/HTTPS URLs.\n"
            "- Never performs search or discovery; to FIND pages use web_search.\n"
            "- Skip URLs whose web_search row already has usable `content`; fetch "
            "only URLs whose content is missing or insufficient.\n"
            "Objective-aware refinement (use_llm=true) is optional and best-effort: "
            "URLs are never dropped; pages without reliable spans keep full content "
            "(recall-first). Modes: 'none' full pages (default); 'balanced' target + "
            "context (50-70%); 'recall' most body, minimal chrome (80-95%); "
            "'precision' direct answers (20-50%, requires objective). Without an "
            "objective, refinement is ignored and full content is returned.\n"
            "When the operator configures a domain allowlist, a URL on a host outside "
            "it is denied: its entry carries status 'denied_by_allowlist' and names "
            "the host and the allowlist source; other URLs in the call still fetch.\n"
            "Returns a JSON object mapping each input URL to a result "
            "{status, content?, content_length?, published_time_iso?, "
            "modified_time_iso?, error?}."
        ),
    )
    async def _fetch_tool(
        urls: str | List[str],
        objective: Optional[str] = None,
        refinement: str = "none",
        max_content_length: int = -1,
        include_binary_base64: bool = True,
        use_archive_fallback: bool = False,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        return await web_fetch(
            urls=urls,
            objective=objective,
            refinement=refinement,
            max_content_length=max_content_length,
            include_binary_base64=include_binary_base64,
            use_archive_fallback=use_archive_fallback,
            use_llm=use_llm,
        )

    @mcp.tool(
        name="allowlist_status",
        description="The domain allowlist this server enforces: source, entries, and whether it is active.",
    )
    async def _status_tool() -> Dict[str, Any]:
        return await allowlist_status()

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP web search server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "http"])
    parser.add_argument("--host", default=os.environ.get("MCP_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_SERVER_PORT", "8787")))
    parser.add_argument(
        "--allowlist",
        default=None,
        help=f"path to the domain allowlist file (same as {ALLOWLIST_FILE_ENV})",
    )
    args = parser.parse_args()

    if args.allowlist:
        os.environ[ALLOWLIST_FILE_ENV] = args.allowlist

    app = _build_mcp_app()
    if args.transport == "stdio":
        run_stdio(app)
    elif args.transport == "sse":
        run_sse(app, host=args.host, port=args.port)
    else:
        run_http(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
