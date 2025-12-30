# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional

from kdcube_ai_app.apps.chat.sdk.tools.backends.web.inventory import PROVIDERS_AUTHORITY_RANK
from kdcube_ai_app.apps.chat.sdk.util import ts_key

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


def estimate_tokens(text: str, *, divisor: int = 4) -> int:
    if not text:
        return 0
    return max(1, len(text) // max(1, divisor))


def cap_sources_for_context(
        sources: List[Dict[str, Any]],
        *,
        context_texts: List[str],
    max_tokens: int = 200000,
    reserved_tokens: int = 12000,
        overhead_tokens: int = 140,
        token_estimate_divisor: int = 4,
) -> List[Dict[str, Any]]:
    if not sources:
        return []

    sorted_sources = sorted(sources, key=source_sort_key, reverse=True)
    ctx_tokens = sum(estimate_tokens(t or "", divisor=token_estimate_divisor) for t in context_texts)
    budget = max_tokens - reserved_tokens - ctx_tokens
    if budget <= 0:
        return []

    kept: List[Dict[str, Any]] = []
    used = 0
    for row in sorted_sources:
        content = row.get("content") or row.get("text") or ""
        title = row.get("title") or ""
        url = row.get("url") or ""
        est = (
            estimate_tokens(content, divisor=token_estimate_divisor)
            + estimate_tokens(title, divisor=token_estimate_divisor)
            + estimate_tokens(url, divisor=token_estimate_divisor)
            + overhead_tokens
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
) -> List[Dict[str, Any]]:
    if not buckets:
        return []

    prepared: List[List[Dict[str, Any]]] = []
    for bucket in buckets:
        if not bucket:
            continue
        prepared.append(sorted(bucket, key=source_sort_key, reverse=True))
    if not prepared:
        return []

    ctx_tokens = sum(estimate_tokens(t or "", divisor=token_estimate_divisor) for t in context_texts)
    budget = max_tokens - reserved_tokens - ctx_tokens
    if budget <= 0:
        return []

    indices = [0] * len(prepared)
    kept: List[Dict[str, Any]] = []
    used = 0

    while True:
        progressed = False
        for i, bucket in enumerate(prepared):
            while indices[i] < len(bucket):
                row = bucket[indices[i]]
                indices[i] += 1
                content = row.get("content") or row.get("text") or ""
                title = row.get("title") or ""
                url = row.get("url") or ""
                est = (
                    estimate_tokens(content, divisor=token_estimate_divisor)
                    + estimate_tokens(title, divisor=token_estimate_divisor)
                    + estimate_tokens(url, divisor=token_estimate_divisor)
                    + overhead_tokens
                )
                if used + est > budget:
                    continue
                kept.append(row)
                used += est
                progressed = True
                break
        if not progressed:
            break
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
) -> List[Dict[str, Any]]:
    return cap_sources_for_context(
        sources,
        context_texts=[instruction or "", input_context or ""],
        max_tokens=max_tokens,
        reserved_tokens=reserved_tokens,
        overhead_tokens=overhead_tokens,
        token_estimate_divisor=token_estimate_divisor,
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
) -> List[Dict[str, Any]]:
    return cap_sources_for_context_evenly(
        buckets,
        context_texts=[instruction or "", input_context or ""],
        max_tokens=max_tokens,
        reserved_tokens=reserved_tokens,
        overhead_tokens=overhead_tokens,
        token_estimate_divisor=token_estimate_divisor,
    )
