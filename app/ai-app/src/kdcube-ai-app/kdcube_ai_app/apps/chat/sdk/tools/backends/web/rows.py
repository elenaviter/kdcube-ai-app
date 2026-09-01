# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The outward-facing shape of a web result row.

Backend rows carry internal plumbing — segmenter spans and boundaries,
provider/weighted ranks, sids, raw date fields — that costs the calling
model context and serves it nothing. MCP surfaces pass rows through
``clean_rows``: a keep-list, so future internals cannot leak either.
"""

from __future__ import annotations

from typing import Any, Dict, List

ROW_KEEP = frozenset({
    "title", "url", "text", "content", "content_length",
    "mime", "base64", "size_bytes", "fetch_status",
    "objective_relevance", "query_relevance",
    "published_time_iso", "modified_time_iso", "fetched_time_iso",
    "archive_snapshot_url",
})


def clean_rows(rows: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, dict):
            out.append({k: v for k, v in row.items() if k in ROW_KEEP and v is not None})
    return out
