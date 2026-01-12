# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/tools/backends/web/ranking.py
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional

from kdcube_ai_app.apps.chat.sdk.tools.backends.web.inventory import PROVIDERS_AUTHORITY_RANK
from kdcube_ai_app.apps.chat.sdk.util import ts_key
from kdcube_ai_app.infra.service_hub.multimodality import estimate_tokens, MODALITY_IMAGE_MIME, \
    estimate_image_tokens_from_base64, MODALITY_DOC_MIME, estimate_pdf_tokens_from_base64

DEFAULT_WEIGHTED_RANK = 1.0
WEIGHTED_RANK_WEIGHTS = {"relevance": 0.60, "provider": 0.25, "freshness": 0.15}
FRESHNESS_HALF_LIFE_DAYS = 30.0

_PROVIDER_MAX_RANK = max(PROVIDERS_AUTHORITY_RANK.values() or [0])


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def provider_rank(provider: Optional[str]) -> int:
    if not provider:
        return 0
    return int(PROVIDERS_AUTHORITY_RANK.get(str(provider).lower(), 0))


def max_relevance_score(row: Dict[str, Any]) -> float:
    score = 0.0
    for key in ("objective_relevance", "query_relevance", "relevance_score", "rerank_score"):
        try:
            if row.get(key) is not None:
                score = max(score, float(row.get(key) or 0.0))
        except Exception:
            continue
    try:
        q_relevance = row.get("q_relevance") or []
        for qr in q_relevance if isinstance(q_relevance, list) else []:
            if not isinstance(qr, dict):
                continue
            score = max(score, float(qr.get("score") or 0.0))
    except Exception:
        pass
    return score


def compute_weighted_rank(row: Dict[str, Any]) -> tuple[int, float]:
    rank = provider_rank(row.get("provider"))
    prov_score = (rank / _PROVIDER_MAX_RANK) if _PROVIDER_MAX_RANK > 0 else 0.0

    rel_score = max_relevance_score(row)

    ts = ts_key(row.get("modified_time_iso") or row.get("published_time_iso"))
    if ts and ts != float("-inf"):
        age_days = max(0.0, (time.time() - ts) / 86400.0)
        fresh_score = math.exp(-math.log(2) * (age_days / FRESHNESS_HALF_LIFE_DAYS))
    else:
        fresh_score = 0.0

    weighted = (
        WEIGHTED_RANK_WEIGHTS["relevance"] * rel_score
        + WEIGHTED_RANK_WEIGHTS["provider"] * prov_score
        + WEIGHTED_RANK_WEIGHTS["freshness"] * fresh_score
    )
    return rank, _clamp01(weighted)


def ensure_weighted_rank(row: Dict[str, Any]) -> tuple[int, float]:
    try:
        existing = float(row.get("weighted_rank"))
        if 0.0 <= existing <= 1.0:
            rank = row.get("provider_rank")
            if rank is None:
                rank = provider_rank(row.get("provider"))
            return int(rank), existing
    except Exception:
        pass

    has_signals = any(
        row.get(k) not in (None, "")
        for k in (
            "provider",
            "objective_relevance",
            "query_relevance",
            "relevance_score",
            "rerank_score",
            "published_time_iso",
            "modified_time_iso",
        )
    )
    if not has_signals:
        rank = row.get("provider_rank")
        if rank is None:
            rank = provider_rank(row.get("provider"))
        return int(rank), DEFAULT_WEIGHTED_RANK

    return compute_weighted_rank(row)


def source_sort_key(row: Dict[str, Any]) -> tuple[float, float, float]:
    _, weighted = ensure_weighted_rank(row)
    relevance = max_relevance_score(row)
    fresh_ts = ts_key(row.get("modified_time_iso") or row.get("published_time_iso"))
    return (weighted, relevance, fresh_ts)


def apply_weighted_rank(rows: Iterable[Dict[str, Any]], *, force: bool = False) -> None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not force and row.get("weighted_rank") is not None:
            continue
        rank, weighted = compute_weighted_rank(row)
        row["provider_rank"] = rank
        row["weighted_rank"] = weighted


def estimate_source_tokens(
        row: Dict[str, Any],
        *,
        text_divisor: int = 4,
        overhead_tokens: int = 140,
        include_blocks: bool = False,
) -> int:
    """
    Estimate total token cost for a source, including multimodal content.

    Handles:
    - Text fields (title, url, content, text)
    - Images (via mime + base64): 150-1600 tokens
    - PDFs (via mime + base64): pages × 10k tokens
    - XML overhead: ~140 tokens per source

    Args:
        row: Source dict with:
            - text fields: title, url, content, text
            - mime: MIME type (e.g., "image/jpeg", "application/pdf")
            - base64: Base64-encoded binary data (optional)
        text_divisor: Characters per token for text estimation (default 4)
        overhead_tokens: Fixed overhead per source (XML tags, metadata)

    Returns:
        Estimated total tokens for this source

    Examples:
        # Text-only source
        row = {"title": "Article", "content": "...", "url": "https://..."}
        tokens = estimate_source_tokens(row)  # ~overhead + text

        # Image source
        row = {"title": "Photo", "mime": "image/jpeg", "base64": "..."}
        tokens = estimate_source_tokens(row)  # ~overhead + 150-1600

        # PDF source
        row = {"title": "Report", "mime": "application/pdf", "base64": "..."}
        tokens = estimate_source_tokens(row)  # ~overhead + pages*10k
    """
    tokens = overhead_tokens

    def _estimate_blocks(blocks: List[Dict[str, Any]]) -> int:
        est = 0
        for b in blocks or []:
            if not isinstance(b, dict):
                continue
            b_type = (b.get("type") or "").strip().lower()
            if b_type == "text":
                txt = b.get("text") or ""
                if txt:
                    est += estimate_tokens(txt, divisor=text_divisor)
                continue
            if b_type == "image":
                mime = (b.get("mime") or "").lower()
                data_b64 = b.get("base64") or ""
                alt = b.get("alt") or ""
                caption = b.get("caption") or ""
                if alt:
                    est += estimate_tokens(alt, divisor=text_divisor)
                if caption:
                    est += estimate_tokens(caption, divisor=text_divisor)
                if data_b64:
                    if mime in MODALITY_IMAGE_MIME:
                        est += estimate_image_tokens_from_base64(data_b64)
                    elif mime in MODALITY_DOC_MIME:
                        est += estimate_pdf_tokens_from_base64(data_b64)
                elif b.get("url"):
                    est += estimate_tokens(str(b.get("url")), divisor=text_divisor)
        return est

    # Text content (title, url, text/content)
    for field in ("title", "url"):
        text = row.get(field) or ""
        if text:
            tokens += estimate_tokens(text, divisor=text_divisor)

    content_blocks = row.get("content_blocks")
    if include_blocks and isinstance(content_blocks, list):
        tokens += _estimate_blocks(content_blocks)
    else:
        for field in ("content", "text"):
            text = row.get(field) or ""
            if text:
                tokens += estimate_tokens(text, divisor=text_divisor)

    # Multimodal content
    mime = (row.get("mime") or "").lower()
    base64_data = row.get("base64", "")

    if base64_data:
        if mime in MODALITY_IMAGE_MIME:
            tokens += estimate_image_tokens_from_base64(base64_data)
        elif mime in MODALITY_DOC_MIME:
            tokens += estimate_pdf_tokens_from_base64(base64_data)
        # else: unknown mime, treat as text overhead (already counted)

    return tokens

def cap_sources_for_context(
        sources: List[Dict[str, Any]],
        *,
        context_texts: List[str],
    max_tokens: int = 200000,
    reserved_tokens: int = 12000,
        overhead_tokens: int = 140,
        token_estimate_divisor: int = 4,
        include_blocks: bool = False,
) -> List[Dict[str, Any]]:
    """
    Cap sources to fit within token budget, prioritizing by quality.
    Now supports multimodal content (images, PDFs).

    Strategy:
    1. Sort sources by weighted rank (quality)
    2. Greedily select highest-quality sources that fit
    3. Account for text + images + PDFs in token calculation

    Args:
        sources: List of source dicts (may include 'mime' and 'base64')
        context_texts: Additional context (system prompt, user query, etc.)
        max_tokens: Model context limit (e.g., 200k for Claude)
        reserved_tokens: Reserve for system prompt + output
        overhead_tokens: Per-source XML overhead
        token_estimate_divisor: Chars per token for text

    Returns:
        Filtered list of sources that fit in budget, sorted by quality

    Example:
        sources = [
            {"title": "Article 1", "content": "...", "weighted_rank": 0.9},
            {"title": "Image", "mime": "image/jpeg", "base64": "...", "weighted_rank": 0.7},
            {"title": "PDF", "mime": "application/pdf", "base64": "...", "weighted_rank": 0.5},
        ]

        kept = cap_sources_for_context(
            sources,
            context_texts=["System prompt", "User query"],
            max_tokens=180_000,
            reserved_tokens=4_000,
        )
        # Returns highest-quality sources that fit (~176k tokens)
    """
    if not sources:
        return []

    # Sort by quality (weighted rank, relevance, freshness)
    sorted_sources = sorted(sources, key=source_sort_key, reverse=True)

    # Calculate budget
    ctx_tokens = sum(
        estimate_tokens(t or "", divisor=token_estimate_divisor)
        for t in context_texts
    )
    budget = max_tokens - reserved_tokens - ctx_tokens

    if budget <= 0:
        return []

    # Greedily select sources that fit
    kept: List[Dict[str, Any]] = []
    used = 0

    for row in sorted_sources:
        est = estimate_source_tokens(
            row,
            text_divisor=token_estimate_divisor,
            overhead_tokens=overhead_tokens,
            include_blocks=include_blocks,
        )

        if used + est > budget:
            continue

        kept.append(row)
        used += est

    return kept


def cap_sources_for_context_evenly(
        buckets: List[List[Dict[str, Any]]],
        *,
        context_texts: List[str],
    max_tokens: int = 200000,
    reserved_tokens: int = 12000,
        overhead_tokens: int = 140,
        token_estimate_divisor: int = 4,
        include_blocks: bool = False,
) -> List[Dict[str, Any]]:
    """
    Cap sources evenly across buckets (e.g., per-query results).
    Now supports multimodal content.

    Strategy:
    1. Sort each bucket by quality
    2. Round-robin: take best from bucket 1, then bucket 2, etc.
    3. Ensures fair representation from all buckets
    4. Account for text + images + PDFs in token calculation

    Args:
        buckets: List of source lists (e.g., results per query)
        context_texts: Additional context
        max_tokens: Model context limit
        reserved_tokens: Reserve for system + output
        overhead_tokens: Per-source overhead
        token_estimate_divisor: Chars per token

    Returns:
        Filtered list with fair representation from each bucket

    Example:
        buckets = [
            [  # Query 1 results
                {"title": "Q1 Result 1", "content": "...", "weighted_rank": 0.9},
                {"title": "Q1 Image", "mime": "image/png", "base64": "...", "weighted_rank": 0.7},
            ],
            [  # Query 2 results
                {"title": "Q2 Result 1", "content": "...", "weighted_rank": 0.8},
                {"title": "Q2 PDF", "mime": "application/pdf", "base64": "...", "weighted_rank": 0.6},
            ],
        ]

        kept = cap_sources_for_context_evenly(
            buckets,
            context_texts=["System", "Queries"],
            max_tokens=180_000,
        )
        # Returns: Q1-R1, Q2-R1, Q1-Image, Q2-PDF (round-robin, as budget allows)
    """
    if not buckets:
        return []

    # Sort each bucket by quality
    prepared: List[List[Dict[str, Any]]] = []
    for bucket in buckets:
        if not bucket:
            continue
        prepared.append(sorted(bucket, key=source_sort_key, reverse=True))

    if not prepared:
        return []

    # Calculate budget
    ctx_tokens = sum(
        estimate_tokens(t or "", divisor=token_estimate_divisor)
        for t in context_texts
    )
    budget = max_tokens - reserved_tokens - ctx_tokens

    if budget <= 0:
        return []

    # Round-robin across buckets
    indices = [0] * len(prepared)
    kept: List[Dict[str, Any]] = []
    used = 0

    while True:
        progressed = False

        for i, bucket in enumerate(prepared):
            while indices[i] < len(bucket):
                row = bucket[indices[i]]
                indices[i] += 1

                est = estimate_source_tokens(
                    row,
                    text_divisor=token_estimate_divisor,
                    overhead_tokens=overhead_tokens,
                    include_blocks=include_blocks,
                )

                if used + est > budget:
                    # Can't fit this source, try next bucket
                    continue

                kept.append(row)
                used += est
                progressed = True
                break  # Move to next bucket

        if not progressed:
            # No bucket made progress, we're done
            break

        # All buckets exhausted?
        if all(indices[i] >= len(prepared[i]) for i in range(len(prepared))):
            break

    return kept


def cap_sources_for_llm(
    sources: List[Dict[str, Any]],
    *,
    instruction: Optional[str],
    input_context: Optional[str],
    max_tokens: int = 200000,
    reserved_tokens: int = 12000,
    overhead_tokens: int = 140,
    token_estimate_divisor: int = 4,
    include_blocks: bool = False,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper for cap_sources_for_context.
    Now supports multimodal content.
    """
    return cap_sources_for_context(
        sources,
        context_texts=[instruction or "", input_context or ""],
        max_tokens=max_tokens,
        reserved_tokens=reserved_tokens,
        overhead_tokens=overhead_tokens,
        token_estimate_divisor=token_estimate_divisor,
        include_blocks=include_blocks,
    )


def cap_sources_for_llm_evenly(
    buckets: List[List[Dict[str, Any]]],
    *,
    instruction: Optional[str],
    input_context: Optional[str],
    max_tokens: int = 180000,
    reserved_tokens: int = 4000,
    overhead_tokens: int = 140,
    token_estimate_divisor: int = 4,
    include_blocks: bool = False,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper for cap_sources_for_context_evenly.
    Now supports multimodal content.
    """
    return cap_sources_for_context_evenly(
        buckets,
        context_texts=[instruction or "", input_context or ""],
        max_tokens=max_tokens,
        reserved_tokens=reserved_tokens,
        overhead_tokens=overhead_tokens,
        token_estimate_divisor=token_estimate_divisor,
        include_blocks=include_blocks,
    )
