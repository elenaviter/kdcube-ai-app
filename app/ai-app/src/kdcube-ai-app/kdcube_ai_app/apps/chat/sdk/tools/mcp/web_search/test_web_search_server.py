# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import asyncio
import json
from types import SimpleNamespace

import pytest

import kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server as srv
from kdcube_ai_app.infra.service_hub import cache as cache_mod


@pytest.fixture(autouse=True)
def _reset_server_state(monkeypatch):
    monkeypatch.delenv("WEB_ALLOWLIST_FILE", raising=False)
    monkeypatch.delenv("WEB_ALLOWLIST", raising=False)
    srv._ALLOWLIST = None
    srv._SERVICE = None
    srv._CACHE = None
    yield
    srv._ALLOWLIST = None
    srv._SERVICE = None
    srv._CACHE = None


def test_web_search_server_uses_backend(monkeypatch):
    called = {}

    async def _fake_search(*, _SERVICE, queries, objective, refinement, n, fetch_content,
                           include_binary_base64, freshness, country, safesearch,
                           use_llm, allowed_domains, namespaced_kv_cache):
        called["svc"] = _SERVICE
        called["cache"] = namespaced_kv_cache
        called["use_llm"] = use_llm
        called["allowed_domains"] = allowed_domains
        return [{"ok": True}]

    monkeypatch.setenv("DEFAULT_LLM_MODEL_ID", "o3-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setattr(srv.search_backends, "web_search", _fake_search)

    async def _run():
        out = await srv.web_search(
            queries="test",
            objective=None,
            refinement="balanced",
            n=3,
            fetch_content=False,
            include_binary_base64=False,
            freshness=None,
            country=None,
            safesearch="moderate",
        )
        assert out == [{"ok": True}]
        assert called.get("svc") is not None
        assert called.get("use_llm") is True
        assert called.get("allowed_domains") is None  # no allowlist configured

    asyncio.run(_run())


def test_web_search_without_llm_skips_model_service(monkeypatch):
    called = {}

    async def _fake_search(**kwargs):
        called.update(kwargs)
        return []

    async def _no_service():  # pragma: no cover - must not be reached
        raise AssertionError("model service built although use_llm is False")

    monkeypatch.setattr(srv.search_backends, "web_search", _fake_search)
    monkeypatch.setattr(srv, "_build_model_service_from_env", _no_service)
    monkeypatch.setenv("WEB_ALLOWLIST", "usgs.gov, noaa.gov")

    asyncio.run(srv.web_search(queries="quakes", use_llm=False))
    assert called["_SERVICE"] is None
    assert called["use_llm"] is False
    assert called["allowed_domains"] == ["usgs.gov", "noaa.gov"]


def test_web_fetch_denies_hosts_outside_allowlist(monkeypatch):
    monkeypatch.setenv("WEB_ALLOWLIST", "usgs.gov")
    seen = {}

    async def _fake_fetch(*, _SERVICE, urls, use_archive_fallback, **kwargs):
        seen["urls"] = json.loads(urls)
        seen["archive"] = use_archive_fallback
        seen["service"] = _SERVICE
        return {u: {"status": "success", "content": "ok"} for u in json.loads(urls)}

    monkeypatch.setattr(srv.fetch_backends, "fetch_url_contents", _fake_fetch)

    out = asyncio.run(
        srv.web_fetch(
            urls=json.dumps(["https://usgs.gov/quakes", "https://evil.com/x"]),
            use_archive_fallback=True,
        )
    )
    assert out["https://usgs.gov/quakes"]["status"] == "success"
    denied = out["https://evil.com/x"]
    assert denied["status"] == "denied_by_allowlist"
    assert "evil.com" in denied["error"]
    assert denied["allowlist_entries"] == ["usgs.gov"]
    assert seen["urls"] == ["https://usgs.gov/quakes"]
    # an archive mirror is another host: fallback forced off while the allowlist is on
    assert seen["archive"] is False


def test_web_fetch_without_allowlist_keeps_archive_fallback(monkeypatch):
    seen = {}

    async def _fake_fetch(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(srv.fetch_backends, "fetch_url_contents", _fake_fetch)
    asyncio.run(srv.web_fetch(urls="https://example.org/a", use_archive_fallback=True))
    assert seen["use_archive_fallback"] is True


def test_allowlist_status(monkeypatch):
    monkeypatch.setenv("WEB_ALLOWLIST", "usgs.gov,noaa.gov")
    out = asyncio.run(srv.allowlist_status())
    assert out["enforced"] is True
    assert out["allowlist_entries"] == ["usgs.gov", "noaa.gov"]
    assert out["entry_count"] == 2

    srv._ALLOWLIST = None
    monkeypatch.delenv("WEB_ALLOWLIST", raising=False)
    out = asyncio.run(srv.allowlist_status())
    assert out["enforced"] is False


def test_web_search_cache_settings(monkeypatch):
    monkeypatch.setattr(
        cache_mod,
        "get_settings",
        lambda: SimpleNamespace(REDIS_URL=""),
    )
    cache = cache_mod.create_kv_cache_from_env(ttl_env_var="WEB_SEARCH_CACHE_TTL_SECONDS")
    assert cache is None

    monkeypatch.setattr(
        cache_mod,
        "get_settings",
        lambda: SimpleNamespace(REDIS_URL="redis://localhost:6379/0"),
    )
    monkeypatch.setenv("WEB_SEARCH_CACHE_TTL_SECONDS", "120")
    cache = cache_mod.create_kv_cache_from_env(ttl_env_var="WEB_SEARCH_CACHE_TTL_SECONDS")
    # In CI without redis, this still builds the cache object; connection is lazy.
    assert cache is not None


class _StubMCPServer:
    """Registration-shape stand-in for KDCubeMCPServer (SDK v2 is not in test venvs)."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.tool_names = []

    def tool(self, name=None, description=None):
        def _decorator(fn):
            self.tool_names.append(name or fn.__name__)
            return fn

        return _decorator


def test_registered_tools(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "kdcube_ai_app.apps.chat.sdk.runtime.mcp.server",
        SimpleNamespace(KDCubeMCPServer=_StubMCPServer),
    )
    app = srv._build_mcp_app()
    assert set(app.tool_names) == {"web_search", "web_fetch", "allowlist_status"}
