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
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase, _build_model_service_from_env
from kdcube_ai_app.infra.service_hub.cache import create_kv_cache_from_env
import kdcube_ai_app.apps.chat.sdk.tools.backends.web.search_backends as search_backends
import kdcube_ai_app.apps.chat.sdk.tools.backends.web.fetch_backends as fetch_backends
from kdcube_ai_app.apps.chat.sdk.tools.backends.web.allowlist import (
    ALLOWLIST_ENV,
    ALLOWLIST_FILE_ENV,
    ALLOWLIST_YAML_ENV,
    Allowlist,
)
from kdcube_ai_app.apps.chat.sdk.tools.mcp.mcp_app_transport import run_http, run_sse, run_stdio

_SERVICE: Optional[ModelServiceBase] = None
_CACHE = None
_ALLOWLIST: Optional[Allowlist] = None

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

    for section, value in data.items():
        if section == "filter" and isinstance(value, dict):
            if isinstance(value.get("allowlist"), (list, tuple)):
                # the config file itself becomes the live allowlist source:
                # edits to its filter.allowlist apply on the next call
                _set(ALLOWLIST_YAML_ENV, str(path))
            elif value.get("allowlist_file"):
                _set(ALLOWLIST_FILE_ENV, value.get("allowlist_file"))
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
    """--config beats WEB_SEARCH_CONFIG beats a config.yaml beside this file."""
    for candidate in (
        cli_path,
        os.environ.get("WEB_SEARCH_CONFIG"),
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
