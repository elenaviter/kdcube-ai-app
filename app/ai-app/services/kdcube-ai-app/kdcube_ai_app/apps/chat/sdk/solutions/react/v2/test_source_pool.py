# SPDX-License-Identifier: MIT

from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.external import (
    _format_sources_pool_path,
    _remap_tool_sources,
)


class _CtxBrowserStub:
    def __init__(self, sources_pool):
        self._sources_pool = list(sources_pool or [])

    @property
    def sources_pool(self):
        return list(self._sources_pool)

    def set_sources_pool(self, *, sources_pool):
        self._sources_pool = list(sources_pool or [])


def _mk_source(sid, url, title="T"):
    return {"sid": sid, "url": url, "title": title, "text": ""}


def test_inline_merge_remap_assigns_new_sids():
    prior = [_mk_source(i, f"https://example.com/p{i}") for i in range(1, 10)]
    ctx = _CtxBrowserStub(prior)
    rows = [
        _mk_source(1, "https://vanta.com/a"),
        _mk_source(2, "https://vanta.com/b"),
        _mk_source(3, "https://vanta.com/c"),
        _mk_source(4, "https://vanta.com/d"),
    ]

    remapped_rows, used_sids = _remap_tool_sources(ctx_browser=ctx, rows=rows)

    assert [r["sid"] for r in remapped_rows] == [10, 11, 12, 13]
    assert used_sids == [10, 11, 12, 13]
    assert len(ctx.sources_pool) == 13
    assert _format_sources_pool_path(used_sids) == "so:sources_pool[10-13]"
    # prior SIDs should remain stable
    prior_sids = [r["sid"] for r in ctx.sources_pool if r["url"].startswith("https://example.com/p")]
    assert prior_sids == list(range(1, 10))


def test_inline_merge_remap_reuses_existing_sid_for_duplicates():
    prior = [
        _mk_source(1, "https://example.com/p1"),
        _mk_source(2, "https://example.com/p2"),
        _mk_source(3, "https://vanta.com/dup"),
        _mk_source(4, "https://example.com/p4"),
    ]
    ctx = _CtxBrowserStub(prior)
    rows = [
        _mk_source(1, "https://vanta.com/dup"),
        _mk_source(2, "https://vanta.com/new1"),
        _mk_source(3, "https://vanta.com/new2"),
    ]

    remapped_rows, used_sids = _remap_tool_sources(ctx_browser=ctx, rows=rows)

    assert [r["sid"] for r in remapped_rows] == [3, 5, 6]
    assert used_sids == [3, 5, 6]
    assert _format_sources_pool_path(used_sids) == "so:sources_pool[3, 5-6]"
