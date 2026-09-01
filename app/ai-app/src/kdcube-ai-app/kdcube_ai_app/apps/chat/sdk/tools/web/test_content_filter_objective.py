# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Regression: the content filter must tolerate a missing objective.

Standalone MCP callers may search with no objective at all; the filter
stage used to crash on `objective[:100]` (surfaced by a Claude Desktop
call with default parameters, 2026-09-02)."""

import asyncio

import kdcube_ai_app.apps.chat.sdk.tools.web.with_llm as with_llm


def test_filter_tolerates_missing_objective(monkeypatch):
    async def _fake_segment(**kwargs):
        return {}

    monkeypatch.setattr(with_llm, "sources_filter_and_segment", _fake_segment)
    rows = [
        {"sid": 1, "url": "https://example.org/a", "content": "text",
         "fetch_status": "success"},
        {"sid": 2, "url": "https://example.org/b", "content": "more text",
         "fetch_status": "success"},
    ]
    out = asyncio.run(
        with_llm.filter_search_results_by_content(
            _SERVICE=None,
            objective=None,
            queries=["q"],
            search_results=rows,
            do_segment=True,
            mode="balanced",
        )
    )
    assert out == []  # empty spans keep nothing - and, above all, no crash
