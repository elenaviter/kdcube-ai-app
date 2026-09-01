# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

import kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server as srv
from kdcube_ai_app.infra.service_hub import cache as cache_mod


@pytest.fixture(autouse=True)
def _reset_server_state(monkeypatch):
    monkeypatch.delenv("WEB_ALLOWLIST_YAML", raising=False)
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


def test_yaml_config_applies_to_env(tmp_path, monkeypatch):
    for var in ("WEB_ALLOWLIST_YAML", "ROLE_MODELS_JSON", "BRAVE_API_KEY", "DEFAULT_LLM_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "filter:\n  allowlist:\n    - example.org\n    - usgs.gov\n"
        "services:\n"
        "  secrets:\n    brave:\n      api_key: brave-key\n"
        "  role_models:\n"
        "    default: {provider: anthropic, model: claude-haiku-4-5-20251001}\n"
        "    tool.source.reconciler: {provider: anthropic, model: claude-haiku-4-5-20251001}\n"
        "unknown_knob: 1\n"
    )
    applied = srv.apply_yaml_config(cfg)
    # inline allowlist makes the config file itself the live source
    assert os.environ["WEB_ALLOWLIST_YAML"] == str(cfg)
    assert os.environ["BRAVE_API_KEY"] == "brave-key"
    assert os.environ["DEFAULT_LLM_MODEL_ID"] == "claude-haiku-4-5-20251001"
    assert json.loads(os.environ["ROLE_MODELS_JSON"]) == {
        "tool.source.reconciler": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
    }
    assert "WEB_ALLOWLIST_YAML" in applied and "ROLE_MODELS_JSON" in applied

    srv._ALLOWLIST = None
    out = asyncio.run(srv.allowlist_status())
    assert out["enforced"] is True
    assert out["allowlist_entries"] == ["example.org", "usgs.gov"]

    # editing the allowlist in the yaml applies on the next call
    cfg.write_text(cfg.read_text().replace("    - usgs.gov\n", "    - usgs.gov\n    - noaa.gov\n"))
    os.utime(cfg, (os.path.getmtime(cfg) + 10, os.path.getmtime(cfg) + 10))
    out = asyncio.run(srv.allowlist_status())
    assert "noaa.gov" in out["allowlist_entries"]

    for var in ("WEB_ALLOWLIST_YAML", "ROLE_MODELS_JSON", "BRAVE_API_KEY", "DEFAULT_LLM_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)


def test_yaml_config_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "from-env")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("services:\n  secrets:\n    brave:\n      api_key: from-file\n")
    applied = srv.apply_yaml_config(cfg)
    assert os.environ["BRAVE_API_KEY"] == "from-env"
    assert "BRAVE_API_KEY" not in applied


def test_load_config_discovers_and_applies(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("services:\n  secrets:\n    brave:\n      api_key: k1\n")
    monkeypatch.setenv("WEB_SEARCH_CONFIG", str(cfg))
    assert srv.load_config() == cfg
    assert os.environ["BRAVE_API_KEY"] == "k1"
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)


def test_config_discovery_precedence(tmp_path, monkeypatch):
    import pathlib

    cli = tmp_path / "cli.yaml"
    env_cfg = tmp_path / "env.yaml"
    cli.write_text("{}")
    env_cfg.write_text("{}")
    monkeypatch.setenv("WEB_SEARCH_CONFIG", str(env_cfg))
    assert srv._discover_config(str(cli)) == cli
    assert srv._discover_config(None) == env_cfg
    monkeypatch.delenv("WEB_SEARCH_CONFIG")

    # the operator's working directory owns the config, beside the clone
    workdir = tmp_path / "install"
    workdir.mkdir()
    (workdir / "config.yaml").write_text("{}")
    monkeypatch.chdir(workdir)
    assert srv._discover_config(None) == workdir / "config.yaml"

    # last resort: a config.yaml beside the server file (in-repo dev case)
    monkeypatch.chdir(tmp_path)
    found = srv._discover_config(None)
    module_cfg = pathlib.Path(srv.__file__).with_name("config.yaml")
    assert found == (module_cfg if module_cfg.is_file() else None)


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
