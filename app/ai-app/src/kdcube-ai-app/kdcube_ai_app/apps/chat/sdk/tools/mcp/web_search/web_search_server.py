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
import logging
import os
import pathlib
from typing import Annotated, Any, Dict, List, Optional
from urllib.parse import urlsplit

from pydantic import Field

from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase, _build_model_service_from_env
from kdcube_ai_app.infra.service_hub.cache import create_kv_cache_from_env
import kdcube_ai_app.apps.chat.sdk.tools.backends.web.search_backends as search_backends
import kdcube_ai_app.apps.chat.sdk.tools.backends.web.fetch_backends as fetch_backends
from kdcube_ai_app.apps.chat.sdk.tools.backends.web.allowlist import (
    ALLOWLIST_ENV,
    ALLOWLIST_FILE_ENV,
    ALLOWLIST_YAML_ENV,
    BLOCKLIST_FILE_ENV,
    BLOCKLIST_YAML_ENV,
    EgressFilter,
)
from kdcube_ai_app.apps.chat.sdk.tools.mcp.mcp_app_transport import run_http, run_sse, run_stdio

_SERVICE: Optional[ModelServiceBase] = None
_CACHE = None
_FILTER: Optional[EgressFilter] = None

_logger = logging.getLogger(__name__)

# config.yaml keys -> the environment variables the stack reads. The YAML
# file is the friendly form of the same settings; values from it are
# applied onto the process environment, and a variable already set in the
# environment wins over the file.
# Scoped config keys -> the environment variables the stack reads. Every
# setting lives in a named section; see config.example.yaml.
SCOPED_ENV_MAP = {
    ("cache", "redis_url"): "REDIS_URL",
    ("cache", "ttl_seconds"): "WEB_SEARCH_CACHE_TTL_SECONDS",
    ("server", "host"): "MCP_SERVER_HOST",
    ("server", "port"): "MCP_SERVER_PORT",
    ("kdcube", "assembly_yaml"): "ASSEMBLY_YAML_DESCRIPTOR_PATH",
    ("kdcube", "global_secrets_yaml"): "GLOBAL_SECRETS_YAML",
    ("tls", "cert_file"): "SSL_CERT_FILE",
}

# ``services.secrets:`` - per-provider api_key blocks, the same shape a
# KDCube deployment's secrets.yaml nests under ``services:``, so the
# block can be carried over verbatim.
SERVICES_SECRETS_ENV_MAP = {
    "brave": "BRAVE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def apply_yaml_config(path: str | pathlib.Path) -> List[str]:
    """Apply a config.yaml onto the process environment.

    Every setting lives in a named section (see config.example.yaml):
    ``filter`` (allowlist inline - the config file itself becomes the
    live, re-read allowlist source - or allowlist_file),
    ``services.secrets`` (per-provider api_key blocks, the secrets.yaml
    shape), ``services.role_models`` (pinned provider+model per role,
    serialized into ROLE_MODELS_JSON, with ``default`` covering unpinned
    roles), ``cache``, ``server``, ``kdcube`` (a deployment's
    assembly/secrets YAMLs as the key source), ``tls``. Environment
    variables that are already set win over the file. Returns the names
    of the variables the file supplied; unknown keys are logged and
    skipped.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must hold a mapping")

    applied: List[str] = []

    def _set(env_name: str, value: Any) -> None:
        if value is None:
            return
        if os.environ.get(env_name):
            return  # explicit environment wins
        os.environ[env_name] = str(value)
        applied.append(env_name)

    # The config file is always armed as the live source for BOTH lists
    # (unless a *_file source is chosen below): an Allowlist watching a
    # YAML whose key is absent counts as not configured, so the first
    # add of filter.allowlist or filter.blocklist mid-session applies
    # live, exactly like any later edit.
    filter_cfg = data.get("filter") if isinstance(data.get("filter"), dict) else {}
    if filter_cfg.get("allowlist_file"):
        _set(ALLOWLIST_FILE_ENV, filter_cfg.get("allowlist_file"))
    else:
        _set(ALLOWLIST_YAML_ENV, str(path))
    if filter_cfg.get("blocklist_file"):
        _set(BLOCKLIST_FILE_ENV, filter_cfg.get("blocklist_file"))
    else:
        _set(BLOCKLIST_YAML_ENV, str(path))
    if "ssrf_guard" in filter_cfg:
        _set("WEB_SSRF_GUARD", filter_cfg.get("ssrf_guard"))
    if "expose_edit_tool" in filter_cfg:
        _set("WEB_FILTER_EDIT_TOOL", filter_cfg.get("expose_edit_tool"))

    for section, value in data.items():
        if section == "filter":
            continue
        if section == "services" and isinstance(value, dict):
            secrets = value.get("secrets")
            if isinstance(secrets, dict):
                for provider, block in secrets.items():
                    api_key = block.get("api_key") if isinstance(block, dict) else None
                    env_name = SERVICES_SECRETS_ENV_MAP.get(str(provider))
                    if env_name:
                        _set(env_name, api_key)
                    else:
                        _logger.warning(
                            "config %s: unknown service '%s' skipped", path, provider
                        )
            role_models = value.get("role_models")
            if isinstance(role_models, dict):
                # the ``default`` pseudo-role covers every role not pinned
                default_spec = role_models.get("default")
                if isinstance(default_spec, dict) and default_spec.get("model"):
                    _set("DEFAULT_LLM_MODEL_ID", default_spec.get("model"))
                pinned = {k: v for k, v in role_models.items() if k != "default"}
                if pinned:
                    _set("ROLE_MODELS_JSON", json.dumps(pinned))
            for key in value:
                if key not in ("secrets", "role_models"):
                    _logger.warning(
                        "config %s: unknown key 'services.%s' skipped", path, key
                    )
            continue
        if not isinstance(value, dict):
            _logger.warning("config %s: unknown key '%s' skipped", path, section)
            continue
        for key, item in value.items():
            env_name = SCOPED_ENV_MAP.get((section, key))
            if env_name:
                _set(env_name, item)
            else:
                _logger.warning(
                    "config %s: unknown key '%s.%s' skipped", path, section, key
                )
    return applied


def _discover_config(cli_path: Optional[str]) -> Optional[pathlib.Path]:
    """--config, then WEB_SEARCH_CONFIG, then config.yaml in the working
    directory (the operator's install dir - the intended home for the
    config, beside the clone rather than inside it), then config.yaml
    beside this file (the in-repo development case)."""
    for candidate in (
        cli_path,
        os.environ.get("WEB_SEARCH_CONFIG"),
        pathlib.Path.cwd() / "config.yaml",
        pathlib.Path(__file__).with_name("config.yaml"),
    ):
        if candidate and pathlib.Path(candidate).is_file():
            return pathlib.Path(candidate)
    return None


def load_config(path: Optional[str] = None) -> Optional[pathlib.Path]:
    """Discover and apply the YAML config; the entry point for direct calls.

    ``main()`` runs this automatically. Code that imports this module and
    calls the tool functions directly MUST call it first (or set the env
    vars itself): without it the allowlist is unconfigured, and
    unconfigured means every host is allowed. Returns the applied config
    path, or None when no config was found.
    """
    config_path = _discover_config(path)
    if config_path is not None:
        applied = apply_yaml_config(config_path)
        _logger.info(
            "config %s applied: %s", config_path, ", ".join(applied) or "nothing (env wins)"
        )
    return config_path


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


def _get_filter() -> EgressFilter:
    global _FILTER
    if _FILTER is None:
        _FILTER = EgressFilter.from_env()
    return _FILTER


def _edit_tool_enabled() -> bool:
    value = (os.environ.get("WEB_FILTER_EDIT_TOOL") or "").strip().lower()
    return value in ("true", "on", "1", "yes")


async def allowlist_edit(
    list_name: str,
    add: Optional[str | List[str]] = None,
    remove: Optional[str | List[str]] = None,
) -> Dict[str, Any]:
    """Operator-enabled list editing; changes are live on the next call."""
    from kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search import list_edit

    def _as_list(value: Optional[str | List[str]]) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            raw = value.strip()
            return [str(v).strip() for v in json.loads(raw)] if raw.startswith("[") else [raw]
        return [str(v).strip() for v in value]

    env_name = ALLOWLIST_YAML_ENV if list_name == "allowlist" else BLOCKLIST_YAML_ENV
    config_path = os.environ.get(env_name)
    if not config_path:
        return {
            "ok": False,
            "error": (
                f"the {list_name} is not sourced from a YAML config in this "
                "deployment; edit its configured source instead"
            ),
        }

    entries, error = list_edit.edit_lists(
        config_path,
        list_name=list_name,
        add=_as_list(add),
        remove=_as_list(remove),
    )
    if error:
        return {"ok": False, "error": error}
    status = await allowlist_status()
    return {"ok": True, "edited": list_name, "entries": entries, "status": status}


def _clamp_sites(sites: Optional[str | List[str]]) -> Optional[List[str]]:
    """Normalize the caller's site scoping and clamp it to the operator's
    filter: sites can narrow the search, never widen egress. Raises when
    every requested site is excluded, naming the reasons in the error."""
    if not sites:
        return None
    if isinstance(sites, str):
        raw = sites.strip()
        site_list = [str(s).strip() for s in json.loads(raw)] if raw.startswith("[") else [raw]
    else:
        site_list = [str(s).strip() for s in sites]
    site_list = [s for s in site_list if s]
    if not site_list:
        return None
    egress = _get_filter()
    kept = [s for s in site_list if egress.check(s)]
    if not kept:
        reasons = "; ".join(egress.deny_reason(s) for s in site_list)
        raise ValueError(
            f"every requested site is excluded by the operator's egress filter: {reasons}. "
            "The operator owns this config; ask them to change it, or search without 'sites'."
        )
    return kept


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
    sites: Optional[str | List[str]] = None,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    svc = await _get_service() if use_llm else None
    cache = _get_cache()
    egress = _get_filter()
    kept_sites = _clamp_sites(sites)
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
        sites=kept_sites,
        allowed_domains=(
            kept_sites if kept_sites
            else (egress.allowlist.entries if egress.allowlist.configured else None)
        ),
        blocked_domains=egress.blocklist.entries if egress.blocklist.configured else None,
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
    """Fetch URLs; a host the egress filter refuses is denied with the reason."""
    egress = _get_filter()
    if isinstance(urls, str):
        raw = urls.strip()
        url_list = [str(u).strip() for u in json.loads(raw)] if raw.startswith("[") else [raw]
    else:
        url_list = [str(u).strip() for u in urls]
    url_list = [u for u in url_list if u]

    out: Dict[str, Any] = {}
    allowed: List[str] = []
    for url in url_list:
        host = urlsplit(url).hostname
        if egress.check(host):
            allowed.append(url)
        else:
            blocked = egress.blocklist.matches(host)
            out[url] = {
                "status": "denied_by_blocklist" if blocked else "denied_by_allowlist",
                "error": f"{egress.deny_reason(host)}; the operator owns this config",
            }

    if allowed:
        svc = await _get_service() if use_llm else None
        # An archive mirror is a different host, so it would step around the
        # egress filter; with one configured the fallback stays off.
        fetched = await fetch_backends.fetch_url_contents(
            _SERVICE=svc,
            urls=json.dumps(allowed),
            max_content_length=max_content_length,
            use_archive_fallback=use_archive_fallback and not egress.configured,
            include_binary_base64=include_binary_base64,
            refinement=refinement if use_llm else "none",
            objective=objective,
            namespaced_kv_cache=_get_cache(),
        )
        if isinstance(fetched, dict):
            out.update(fetched)
    return out


async def allowlist_status() -> Dict[str, Any]:
    """The egress filter exactly as this server enforces it."""
    egress = _get_filter()
    allow_source, allow_entries = egress.allowlist.describe()
    block_source, block_entries = egress.blocklist.describe()
    from kdcube_ai_app.apps.chat.sdk.tools.backends.web import ssrf_guard

    return {
        "allowlist_source": allow_source,
        "allowlist_entries": allow_entries,
        "entry_count": len(allow_entries),
        "blocklist_source": block_source,
        "blocklist_entries": block_entries,
        "blocklist_count": len(block_entries),
        "ssrf_guard": ssrf_guard.enabled(),
        "enforced": egress.configured,
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
            "The 'sites' parameter scopes the search WITHIN the named domains "
            "(the provider query is rewritten with site: operators) - use it when "
            "you know where the answer lives; it can narrow but never widen the "
            "operator's egress filter.\n"
            "When the operator configures an allowlist and/or blocklist, results "
            "from refused hosts are dropped server-side before any content fetch; "
            "a call cannot widen the filter - see allowlist_status.\n"
            "Returns an array of results [{title, url, text, objective_relevance?, "
            "query_relevance?, content?, mime?, base64?, size_bytes?, ...dates}]. "
            "`text` is the search preview snippet; `content` is full fetched page "
            "text when fetch_content ran. Non-HTML supported files return mime/base64 "
            "instead of content."
        ),
    )
    async def _tool(
        queries: Annotated[str | List[str], Field(description=(
            "Array of string queries (rephrases/synonyms) or a single query "
            "string. Query results might be large; prefer max 2 queries at a time."
        ))],
        objective: Annotated[Optional[str], Field(description=(
            "The search objective - the goal or question behind the search. "
            "ALWAYS pass it when you have one (you almost always do): it drives "
            "snippet relevance scoring and content refinement, and without it "
            "results come unranked by relevance and pages stay untrimmed."
        ))] = None,
        refinement: Annotated[str, Field(description=(
            "Post-fetch content refinement (objective-guided, needs use_llm): "
            "'none' full pages; 'balanced' target + context (50-70%); 'recall' "
            "content bodies, minimal chrome (80-95%); 'precision' directly "
            "relevant sections only (20-50%, needs objective)."
        ))] = "balanced",
        n: Annotated[int, Field(ge=1, le=20, description=(
            "Max unique results (1-20). Prefer max 5."
        ))] = 8,
        fetch_content: Annotated[bool, Field(description=(
            "If true, fetch full page content for the results. If false, return "
            "ranked snippets/URLs only - cheaper; fetch selected URLs yourself "
            "with web_fetch."
        ))] = True,
        include_binary_base64: Annotated[bool, Field(description=(
            "If true, attach base64 for binary/image/PDF results when size "
            "limits allow."
        ))] = True,
        freshness: Annotated[Optional[str], Field(description=(
            "Freshness window: 'day'|'week'|'month'|'year' or null."
        ))] = None,
        country: Annotated[Optional[str], Field(description=(
            "Country ISO2 for the search, e.g. 'DE', 'US'."
        ))] = None,
        safesearch: Annotated[str, Field(description=(
            "Safesearch: 'off'|'moderate'|'strict'."
        ))] = "moderate",
        sites: Annotated[Optional[str | List[str]], Field(description=(
            "Scope the search WITHIN these domains (site: operators at the "
            "provider, up to 8). An array of domains or a single domain "
            "string. Use when you know where the answer lives; narrows inside "
            "the operator's egress filter, never widens it."
        ))] = None,
        use_llm: Annotated[bool, Field(description=(
            "True runs the neural pipeline (snippet relevance scoring against "
            "the objective, content refinement); false skips every LLM step - "
            "cheaper, needs no model key, provider ranking only."
        ))] = True,
    ) -> Annotated[List[Dict[str, Any]], Field(description=(
        "Array of result rows: [{title, url, text, objective_relevance?, "
        "query_relevance?, content?, mime?, base64?, size_bytes?, "
        "fetched_time_iso, published_time_iso?, ...}]. `text` is the search "
        "snippet; `content` is full (possibly refined) page text when "
        "fetch_content ran; non-HTML supported files carry mime/base64 "
        "instead of content. Relevance scores are meaningful only when the "
        "LLM reconciler ran - on backends with the reconciler off they "
        "default to 1.0 and carry no signal. Rows from hosts the operator's "
        "egress filter refuses are already gone."
    ))]:
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
            sites=sites,
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
            "When the operator configures an allowlist and/or blocklist, a URL on a "
            "refused host is denied: its entry carries status 'denied_by_allowlist' "
            "or 'denied_by_blocklist' and names the host and the list's source; "
            "other URLs in the call still fetch.\n"
            "Returns a JSON object mapping each input URL to a result "
            "{status, content?, content_length?, published_time_iso?, "
            "modified_time_iso?, error?}."
        ),
    )
    async def _fetch_tool(
        urls: Annotated[str | List[str], Field(description=(
            "Array of absolute HTTP/HTTPS URLs you already know, or a single "
            "URL string. This tool never searches; it only dereferences."
        ))],
        objective: Annotated[Optional[str], Field(description=(
            "The goal or question behind the fetch. Pass it when you want "
            "refinement (use_llm=true): it guides which spans of each page are "
            "kept. Without it content stays full."
        ))] = None,
        refinement: Annotated[str, Field(description=(
            "Post-fetch content refinement (needs use_llm=true and an "
            "objective): 'none' full pages (default); 'balanced' target + "
            "context (50-70%); 'recall' most body, minimal chrome (80-95%); "
            "'precision' direct answers (20-50%). URLs are never dropped: "
            "pages without reliable spans keep full content."
        ))] = "none",
        max_content_length: Annotated[int, Field(description=(
            "Max characters of cleaned content per URL, truncated at a "
            "sentence boundary. -1 = no limit."
        ))] = -1,
        include_binary_base64: Annotated[bool, Field(description=(
            "If true, attach base64 for binary/image/PDF fetches when size "
            "limits allow."
        ))] = True,
        use_archive_fallback: Annotated[bool, Field(description=(
            "Try an archive mirror for blocked or paywalled pages. Forced off "
            "while the operator's egress filter is configured: an archive host "
            "is a different host."
        ))] = False,
        use_llm: Annotated[bool, Field(description=(
            "True enables the objective-guided refinement path (spends model "
            "tokens); false returns cleaned full content with no model call."
        ))] = False,
    ) -> Annotated[Dict[str, Any], Field(description=(
        "JSON object mapping each input URL to its result: {status, "
        "content?, content_length?, published_time_iso?, modified_time_iso?, "
        "date_method?, date_confidence?, error?}. Statuses: success, timeout, "
        "paywall (any hard block, a bot-blocking 403 included), error, "
        "non_html, blocked_403, http_XXX, pdf_redirect, denied_by_allowlist, "
        "denied_by_blocklist, denied_by_ssrf_guard - the denied_* entries "
        "name the host and the reason, and other URLs in the same call still "
        "fetch."
    ))]:
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
        description=(
            "The egress filter this server enforces: allowlist and blocklist "
            "sources, entries, and whether filtering is active. Deny wins: a "
            "blocklisted host is refused even when the allowlist admits it."
        ),
    )
    async def _status_tool() -> Annotated[Dict[str, Any], Field(description=(
        "{allowlist_source, allowlist_entries, entry_count, blocklist_source, "
        "blocklist_entries, blocklist_count, ssrf_guard, enforced} - the "
        "egress filter exactly as this server enforces it."
    ))]:
        return await allowlist_status()

    if _edit_tool_enabled():
        @mcp.tool(
            name="allowlist_edit",
            description=(
                "Edit the operator's egress lists in the live config - enabled "
                "by the operator (filter.expose_edit_tool). Adds/removes domain "
                "entries in the allowlist or blocklist; changes apply on the "
                "next call, no restart. Entries must look like domains "
                "(example.org covers subdomains, *.example.org subdomains "
                "only); anything else is refused. The SSRF guard is not "
                "editable through any tool: private, loopback, link-local, and "
                "metadata addresses stay unreachable regardless of the lists."
            ),
        )
        async def _edit_tool(
            list_name: Annotated[str, Field(description=(
                "'allowlist' or 'blocklist'."
            ))],
            add: Annotated[Optional[str | List[str]], Field(description=(
                "Domain entries to add - an array or a single domain string."
            ))] = None,
            remove: Annotated[Optional[str | List[str]], Field(description=(
                "Domain entries to remove - an array or a single domain string."
            ))] = None,
        ) -> Annotated[Dict[str, Any], Field(description=(
            "{ok, edited, entries, status} on success - entries is the list "
            "after the edit and status the full filter state; {ok: false, "
            "error} names the refusal reason (invalid entry, file-sourced "
            "list, unsupported config shape)."
        ))]:
            return await allowlist_edit(list_name=list_name, add=add, remove=remove)

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
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "path to a config.yaml (see config.example.yaml); also found via "
            "WEB_SEARCH_CONFIG or a config.yaml beside this file. Environment "
            "variables win over file values."
        ),
    )
    args = parser.parse_args()

    if args.allowlist:
        os.environ[ALLOWLIST_FILE_ENV] = args.allowlist

    load_config(args.config)

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
