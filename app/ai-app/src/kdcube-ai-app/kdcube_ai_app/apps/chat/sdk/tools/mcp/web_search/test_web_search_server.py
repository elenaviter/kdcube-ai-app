# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

import kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server as srv
from kdcube_ai_app.infra.service_hub import cache as cache_mod


_FILTER_ENV_VARS = (
    "WEB_ALLOWLIST_YAML", "WEB_ALLOWLIST_FILE", "WEB_ALLOWLIST",
    "WEB_BLOCKLIST_YAML", "WEB_BLOCKLIST_FILE", "WEB_BLOCKLIST",
    "WEB_FILTER_YAML_SECTION", "WEB_FILTER_YAML_TOOL_ID",
)


@pytest.fixture(autouse=True)
def _reset_server_state(monkeypatch):
    for var in _FILTER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    srv._FILTER = None
    srv._SERVICE = None
    srv._CACHE = None
    yield
    srv._FILTER = None
    srv._SERVICE = None
    srv._CACHE = None


def test_web_search_server_uses_backend(monkeypatch):
    called = {}

    async def _fake_search(*, _SERVICE, use_llm, allowed_domains, blocked_domains,
                           sites, namespaced_kv_cache, **kwargs):
        called["svc"] = _SERVICE
        called["cache"] = namespaced_kv_cache
        called["use_llm"] = use_llm
        called["allowed_domains"] = allowed_domains
        called["blocked_domains"] = blocked_domains
        called["sites"] = sites
        return [{
            "title": "t", "url": "https://example.org/a", "text": "snippet",
            "content": "body", "content_length": 4, "fetch_status": "success",
            # internal plumbing that must NOT leave the server:
            "sid": 7, "seg_spans": [{"s": "a", "e": "b"}],
            "seg_end_boundary": "exclusive", "provider_rank": 2,
            "weighted_rank": 0.87, "authority": "web",
            "content_original_length": 100, "content_pruned_length": 4,
            "published_time_raw": None, "archive_snapshot_date": None,
        }]

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
        # rows leave the server in the outward shape only: keep-listed
        # fields, no segmenter/ranking/sid internals, no None values
        assert out == [{
            "title": "t", "url": "https://example.org/a", "text": "snippet",
            "content": "body", "content_length": 4, "fetch_status": "success",
        }]
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


def test_sites_accepts_bare_string(monkeypatch):
    """Regression: a real Claude Desktop call sent sites as a bare string
    and the declared List-only type rejected it at schema validation,
    before our string-tolerant handling could run (2026-09-02)."""
    called = {}

    async def _fake_search(**kwargs):
        called.update(kwargs)
        return []

    monkeypatch.setattr(srv.search_backends, "web_search", _fake_search)
    asyncio.run(srv.web_search(queries="q", use_llm=False, sites="en.wikipedia.org"))
    assert called["sites"] == ["en.wikipedia.org"]


def test_model_facing_params_tolerate_model_shapes(monkeypatch):
    """The MCP schema is generated from the tool signatures; every
    parameter a model may plausibly send as a bare string instead of a
    list must be typed as a union, or pydantic rejects the call before
    our code runs."""
    import sys
    import typing
    from types import SimpleNamespace

    class _CapturingStub:
        def __init__(self, name, **kwargs):
            self.fns = {}

        def tool(self, name=None, description=None, **kwargs):
            def _decorator(fn):
                self.fns[name or fn.__name__] = fn
                return fn

            return _decorator

    monkeypatch.setitem(
        sys.modules,
        "kdcube_ai_app.apps.chat.sdk.runtime.mcp.server",
        SimpleNamespace(KDCubeMCPServer=_CapturingStub),
    )
    app = srv._build_mcp_app()

    def _accepts_str(fn, param):
        hints = typing.get_type_hints(fn, include_extras=False)
        hint = hints[param]
        args = typing.get_args(hint)
        flat = set()
        for a in args or (hint,):
            flat.add(a)
            flat.update(typing.get_args(a))
        return str in flat

    assert _accepts_str(app.fns["web_search"], "queries")
    assert _accepts_str(app.fns["web_search"], "sites")
    assert _accepts_str(app.fns["web_fetch"], "urls")


def test_sites_narrow_within_filter(monkeypatch):
    called = {}

    async def _fake_search(**kwargs):
        called.update(kwargs)
        return []

    monkeypatch.setattr(srv.search_backends, "web_search", _fake_search)
    monkeypatch.setenv("WEB_ALLOWLIST", "usgs.gov, noaa.gov")
    monkeypatch.setenv("WEB_BLOCKLIST", "noaa.gov")

    # earthquake.usgs.gov narrows inside the allowlist; noaa.gov is
    # blocklisted (deny wins) and evil.com is outside — both clamped away
    asyncio.run(srv.web_search(
        queries="q", use_llm=False,
        sites=["earthquake.usgs.gov", "noaa.gov", "evil.com"],
    ))
    assert called["sites"] == ["earthquake.usgs.gov"]
    # the post-filter guarantee follows the narrowed sites
    assert called["allowed_domains"] == ["earthquake.usgs.gov"]
    assert called["blocked_domains"] == ["noaa.gov"]


def test_sites_all_excluded_raises_with_reasons(monkeypatch):
    async def _fake_search(**kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("search ran although every site was excluded")

    monkeypatch.setattr(srv.search_backends, "web_search", _fake_search)
    monkeypatch.setenv("WEB_ALLOWLIST", "usgs.gov")
    with pytest.raises(ValueError) as err:
        asyncio.run(srv.web_search(queries="q", use_llm=False, sites=["evil.com"]))
    assert "evil.com" in str(err.value) and "allowlist" in str(err.value)


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
    assert "evil.com" in denied["error"] and "allowlist" in denied["error"]
    assert seen["urls"] == ["https://usgs.gov/quakes"]
    # an archive mirror is another host: fallback forced off while the allowlist is on
    assert seen["archive"] is False


def test_web_fetch_denies_blocklisted_hosts(monkeypatch):
    monkeypatch.setenv("WEB_BLOCKLIST", "tracker.example")
    seen = {}

    async def _fake_fetch(*, _SERVICE, urls, use_archive_fallback, **kwargs):
        seen["urls"] = json.loads(urls)
        return {u: {"status": "success"} for u in json.loads(urls)}

    monkeypatch.setattr(srv.fetch_backends, "fetch_url_contents", _fake_fetch)
    out = asyncio.run(
        srv.web_fetch(urls=json.dumps([
            "https://ok.example/a", "https://cdn.tracker.example/x",
        ]))
    )
    # no allowlist: everything except the blocklist passes
    assert out["https://ok.example/a"]["status"] == "success"
    denied = out["https://cdn.tracker.example/x"]
    assert denied["status"] == "denied_by_blocklist"
    assert "blocklist" in denied["error"]
    assert seen["urls"] == ["https://ok.example/a"]


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
    monkeypatch.setenv("WEB_BLOCKLIST", "tracker.example")
    out = asyncio.run(srv.allowlist_status())
    assert out["enforced"] is True
    assert out["allowlist_entries"] == ["usgs.gov", "noaa.gov"]
    assert out["entry_count"] == 2
    assert out["blocklist_entries"] == ["tracker.example"]
    assert out["blocklist_count"] == 1

    srv._FILTER = None
    monkeypatch.delenv("WEB_ALLOWLIST", raising=False)
    # blocklist alone still counts as an enforced filter
    out = asyncio.run(srv.allowlist_status())
    assert out["enforced"] is True

    srv._FILTER = None
    monkeypatch.delenv("WEB_BLOCKLIST", raising=False)
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
        "filter:\n"
        "  allowlist:\n    - example.org\n    - usgs.gov\n"
        "  blocklist:\n    - tracker.example\n"
        "services:\n"
        "  secrets:\n    brave:\n      api_key: brave-key\n"
        "  role_models:\n"
        "    default: {provider: anthropic, model: claude-haiku-4-5-20251001}\n"
        "    tool.source.reconciler: {provider: anthropic, model: claude-haiku-4-5-20251001}\n"
        "unknown_knob: 1\n"
    )
    applied = srv.apply_yaml_config(cfg)
    # inline lists make the config file itself the live source
    assert os.environ["WEB_ALLOWLIST_YAML"] == str(cfg)
    assert os.environ["WEB_BLOCKLIST_YAML"] == str(cfg)
    assert os.environ["BRAVE_API_KEY"] == "brave-key"
    assert os.environ["DEFAULT_LLM_MODEL_ID"] == "claude-haiku-4-5-20251001"
    assert json.loads(os.environ["ROLE_MODELS_JSON"]) == {
        "tool.source.reconciler": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
    }
    assert "WEB_ALLOWLIST_YAML" in applied and "ROLE_MODELS_JSON" in applied

    srv._FILTER = None
    out = asyncio.run(srv.allowlist_status())
    assert out["enforced"] is True
    assert out["allowlist_entries"] == ["example.org", "usgs.gov"]
    assert out["blocklist_entries"] == ["tracker.example"]

    # editing the allowlist in the yaml applies on the next call
    cfg.write_text(cfg.read_text().replace("    - usgs.gov\n", "    - usgs.gov\n    - noaa.gov\n"))
    os.utime(cfg, (os.path.getmtime(cfg) + 10, os.path.getmtime(cfg) + 10))
    out = asyncio.run(srv.allowlist_status())
    assert "noaa.gov" in out["allowlist_entries"]

    for var in ("WEB_ALLOWLIST_YAML", "ROLE_MODELS_JSON", "BRAVE_API_KEY", "DEFAULT_LLM_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)


def test_yaml_config_can_load_named_embedding_section(tmp_path, monkeypatch):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        "agent:\n  tools: []\n"
        "web_search:\n"
        "  filter:\n"
        "    allowlist:\n      - python.org\n"
        "    blocklist: []\n"
        "    ssrf_guard: true\n"
        "  server:\n    log_level: WARNING\n"
    )

    applied = srv.apply_yaml_config(cfg, section="web_search")

    assert os.environ["WEB_ALLOWLIST_YAML"] == str(cfg)
    assert os.environ["WEB_FILTER_YAML_SECTION"] == "web_search"
    assert "WEB_FILTER_YAML_SECTION" in applied
    srv._FILTER = None
    status = asyncio.run(srv.allowlist_status())
    assert status["allowlist_entries"] == ["python.org"]
    assert status["blocklist_entries"] == []
    assert status["ssrf_guard"] is True

    cfg.write_text(cfg.read_text().replace("      - python.org\n", "      - python.org\n      - docs.python.org\n"))
    os.utime(cfg, (os.path.getmtime(cfg) + 10, os.path.getmtime(cfg) + 10))
    status = asyncio.run(srv.allowlist_status())
    assert status["allowlist_entries"] == ["python.org", "docs.python.org"]


def test_yaml_config_rejects_missing_embedding_section(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text("agent:\n  tools: []\n")

    with pytest.raises(ValueError, match="no mapping section 'web_search'"):
        srv.apply_yaml_config(cfg, section="web_search")


def test_yaml_config_can_load_exact_agent_tool_settings(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        "agent:\n"
        "  tools:\n"
        "    - id: other.search\n"
        "      settings:\n"
        "        filter:\n"
        "          allowlist: [wrong.example]\n"
        "    - id: demo.web_search\n"
        "      settings:\n"
        "        filter:\n"
        "          allowlist:\n"
        "            - python.org\n"
        "          blocklist: []\n"
        "          ssrf_guard: true\n"
        "        server:\n"
        "          log_level: WARNING\n"
    )

    applied = srv.apply_yaml_config(cfg, tool_id="demo.web_search")

    assert os.environ["WEB_ALLOWLIST_YAML"] == str(cfg)
    assert os.environ["WEB_FILTER_YAML_TOOL_ID"] == "demo.web_search"
    assert "WEB_FILTER_YAML_TOOL_ID" in applied
    assert "WEB_FILTER_YAML_SECTION" not in os.environ
    srv._FILTER = None
    status = asyncio.run(srv.allowlist_status())
    assert status["allowlist_entries"] == ["python.org"]
    assert status["blocklist_entries"] == []
    assert status["ssrf_guard"] is True


def test_yaml_config_tool_selector_rejects_missing_duplicate_and_mixed_selector(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        "agent:\n"
        "  tools:\n"
        "    - id: demo.web_search\n"
        "      settings: {}\n"
        "    - id: demo.web_search\n"
        "      settings: {}\n"
    )

    with pytest.raises(ValueError, match="duplicate agent tool 'demo.web_search'"):
        srv.apply_yaml_config(cfg, tool_id="demo.web_search")
    with pytest.raises(ValueError, match="no agent tool 'missing.search'"):
        srv.apply_yaml_config(cfg, tool_id="missing.search")
    with pytest.raises(ValueError, match="mutually exclusive"):
        srv.apply_yaml_config(cfg, section="agent", tool_id="demo.web_search")


def test_first_list_add_and_full_removal_are_live(tmp_path, monkeypatch):
    """Regression for the e2e finding: a blocklist added to a config that
    started without one must apply live, and removing the allowlist key
    entirely must return to allow-all - both without a restart."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("filter:\n  allowlist:\n    - example.org\n")
    monkeypatch.setenv("WEB_SEARCH_CONFIG", str(cfg))
    srv.load_config()

    out = asyncio.run(srv.allowlist_status())
    assert out["enforced"] is True
    assert out["blocklist_entries"] == []

    # first-ever blocklist add, same process
    cfg.write_text(
        "filter:\n  allowlist:\n    - example.org\n  blocklist:\n    - example.org\n"
    )
    os.utime(cfg, (os.path.getmtime(cfg) + 10, os.path.getmtime(cfg) + 10))
    out = asyncio.run(srv.allowlist_status())
    assert out["blocklist_entries"] == ["example.org"]
    assert srv._get_filter().check("example.org") is False  # deny wins

    # removing the allowlist key entirely returns to allow-all
    cfg.write_text("filter:\n  blocklist:\n    - example.org\n")
    os.utime(cfg, (os.path.getmtime(cfg) + 20, os.path.getmtime(cfg) + 20))
    egress = srv._get_filter()
    assert egress.allowlist.configured is False
    assert egress.check("anything.example") is True
    assert egress.check("example.org") is False  # still blocklisted

    monkeypatch.delenv("WEB_SEARCH_CONFIG", raising=False)


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
    monkeypatch.delenv("WEB_FILTER_EDIT_TOOL", raising=False)
    app = srv._build_mcp_app()
    assert set(app.tool_names) == {"web_search", "web_fetch", "allowlist_status"}

    # operator opt-in exposes the edit tool
    monkeypatch.setenv("WEB_FILTER_EDIT_TOOL", "true")
    app = srv._build_mcp_app()
    assert "site_filter_edit" in app.tool_names


def test_site_filter_edit_applies_live(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("filter:\n  expose_edit_tool: true\n  allowlist:\n    - example.org\n")
    monkeypatch.setenv("WEB_SEARCH_CONFIG", str(cfg))
    srv.load_config()
    assert srv._edit_tool_enabled()

    out = asyncio.run(srv.site_filter_edit("allowlist", add="noaa.gov"))
    assert out["ok"] is True
    assert out["entries"] == ["example.org", "noaa.gov"]
    assert out["status"]["allowlist_entries"] == ["example.org", "noaa.gov"]

    out = asyncio.run(srv.site_filter_edit("blocklist", add=["example.org"]))
    assert out["ok"] is True
    assert srv._get_filter().check("example.org") is False  # deny wins, live

    out = asyncio.run(srv.site_filter_edit("allowlist", add="not a domain"))
    assert out["ok"] is False and "does not look like a domain" in out["error"]

    monkeypatch.delenv("WEB_SEARCH_CONFIG", raising=False)
    monkeypatch.delenv("WEB_FILTER_EDIT_TOOL", raising=False)


def test_site_filter_edit_applies_live_in_embedding_section(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        "agent:\n  topic: keep-this\n"
        "web_search:\n"
        "  filter:\n"
        "    expose_edit_tool: true\n"
        "    allowlist:\n      - python.org\n"
    )
    srv.load_config(cfg, section="web_search")

    out = asyncio.run(srv.site_filter_edit("allowlist", add="docs.python.org"))

    assert out["ok"] is True
    assert out["status"]["allowlist_entries"] == ["python.org", "docs.python.org"]
    assert "  topic: keep-this" in cfg.read_text()


def test_site_filter_edit_applies_live_in_exact_agent_tool_settings(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        "agent:\n"
        "  topic: keep-this\n"
        "  tools:\n"
        "    - id: demo.web_search\n"
        "      settings:\n"
        "        filter:\n"
        "          expose_edit_tool: true\n"
        "          allowlist:\n"
        "            - python.org\n"
    )
    srv.load_config(cfg, tool_id="demo.web_search")

    out = asyncio.run(srv.site_filter_edit("allowlist", add="docs.python.org"))

    assert out["ok"] is True
    assert out["status"]["allowlist_entries"] == ["python.org", "docs.python.org"]
    assert "  topic: keep-this" in cfg.read_text()
