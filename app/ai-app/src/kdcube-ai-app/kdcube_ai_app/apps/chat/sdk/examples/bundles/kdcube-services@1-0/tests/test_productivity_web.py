# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Productivity web tools: contract tests for the no-account web door.

The regression surface: the operator's allowlist reaches the search backend
and denies fetch hosts server-side, use_llm defaults come from config and a
False default builds no model service, the archive fallback stays off while
an allowlist is configured, and both tools answer in the {ok, error, ret}
envelope with rows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.runtime.dynamic_module_loader import (
    load_dynamic_module_for_path,
)

BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def _web_module():
    _name, module = load_dynamic_module_for_path(
        BUNDLE_ROOT / "surfaces" / "mcp" / "productivity_web.py"
    )
    return module


class _StubAnnotations:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StubMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None, description=None, **kwargs):
        def _decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn

        return _decorator


def _register(mod, config):
    mcp = _StubMCP()
    mod.register_web_tools(
        mcp=mcp,
        tool_annotations_type=_StubAnnotations,
        config_factory=lambda: config,
    )
    return mcp


def test_search_passes_allowlist_and_llm_off(monkeypatch):
    mod = _web_module()
    called = {}

    async def _fake_search(**kwargs):
        called.update(kwargs)
        return [{"title": "t", "url": "https://example.org/a", "provider": "brave", "x": None}]

    monkeypatch.setattr(mod.search_backends, "web_search", _fake_search)

    async def _no_service():  # pragma: no cover - must not be reached
        raise AssertionError("model service built although use_llm is off")

    monkeypatch.setattr(mod, "_get_service", _no_service)

    mcp = _register(
        mod,
        {"web": {"allowlist": ["example.org"], "use_llm_default": False}},
    )
    out = asyncio.run(mcp.tools["productivity_web_search"](queries="quakes"))

    assert out["ok"] is True
    assert called["_SERVICE"] is None
    assert called["use_llm"] is False
    assert called["allowed_domains"] == ["example.org"]
    # provider stripped, None fields dropped, envelope rows returned
    assert out["ret"] == [{"title": "t", "url": "https://example.org/a"}]


def test_search_without_allowlist_passes_none(monkeypatch):
    mod = _web_module()
    called = {}

    async def _fake_search(**kwargs):
        called.update(kwargs)
        return []

    monkeypatch.setattr(mod.search_backends, "web_search", _fake_search)
    mcp = _register(mod, {"web": {"use_llm_default": False}})
    asyncio.run(mcp.tools["productivity_web_search"](queries="q"))
    assert called["allowed_domains"] is None


def test_fetch_denies_and_converts_rows(monkeypatch):
    mod = _web_module()
    seen = {}

    async def _fake_fetch(**kwargs):
        seen.update(kwargs)
        urls = json.loads(kwargs["urls"])
        return {
            u: {
                "status": "success",
                "content": "body text",
                "content_length": 9,
                "published_time_iso": "2026-01-01T00:00:00+00:00",
            }
            for u in urls
        }

    monkeypatch.setattr(mod.fetch_backends, "fetch_url_contents", _fake_fetch)

    mcp = _register(mod, {"web": {"allowlist": ["usgs.gov"]}})
    out = asyncio.run(
        mcp.tools["productivity_web_fetch"](
            urls=json.dumps(["https://usgs.gov/quakes", "https://evil.com/x"])
        )
    )

    assert out["ok"] is True
    by_url = {row["url"]: row for row in out["ret"]}
    denied = by_url["https://evil.com/x"]
    assert denied["status"] == "denied_by_allowlist"
    assert "evil.com" in denied["error"]
    fetched = by_url["https://usgs.gov/quakes"]
    assert fetched["content"] == "body text"
    assert fetched["published_time_iso"] == "2026-01-01T00:00:00+00:00"
    # only the allowed URL reached the fetcher; archive fallback off with allowlist on
    assert json.loads(seen["urls"]) == ["https://usgs.gov/quakes"]
    assert seen["use_archive_fallback"] is False


def test_fetch_without_allowlist_keeps_archive_fallback(monkeypatch):
    mod = _web_module()
    seen = {}

    async def _fake_fetch(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(mod.fetch_backends, "fetch_url_contents", _fake_fetch)
    mcp = _register(mod, {})
    asyncio.run(mcp.tools["productivity_web_fetch"](urls="https://example.org/a"))
    assert seen["use_archive_fallback"] is True


def test_fetch_no_urls_is_managed_error():
    mod = _web_module()
    mcp = _register(mod, {})
    out = asyncio.run(mcp.tools["productivity_web_fetch"](urls="  "))
    assert out["ok"] is False
    assert out["error"]["code"] == "ValueError"


def test_tool_declarations_have_no_account_requirements():
    mod = _web_module()
    for name, cfg in mod.WEB_PRODUCTIVITY_TOOLS.items():
        assert cfg["connections"] == {}, name
