# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import time
from typing import Dict, Optional, List

from kdcube_ai_app.infra.namespaces import ns_key


DEFAULT_RETENTION_SEC = 3600


def metric_key(base: str, *, tenant: str, project: str, component: str) -> str:
    return ns_key(f"{base}:{component}", tenant=tenant, project=project)


async def record_metric(
    redis,
    *,
    base: str,
    tenant: str,
    project: str,
    component: str,
    value: float,
    retention_sec: int = DEFAULT_RETENTION_SEC,
) -> None:
    if redis is None:
        return
    ts = time.time()
    key = metric_key(base, tenant=tenant, project=project, component=component)
    member = f"{ts}:{value}"
    try:
        await redis.zadd(key, {member: ts})
        await redis.zremrangebyscore(key, 0, ts - retention_sec)
        await redis.expire(key, retention_sec + 60)
    except Exception:
        return


async def read_metric_windows(
    redis,
    *,
    base: str,
    tenant: str,
    project: str,
    component: str,
    retention_sec: int = DEFAULT_RETENTION_SEC,
) -> Dict[str, Optional[float]]:
    if redis is None:
        return {"1m": None, "15m": None, "1h": None, "max": None}
    key = metric_key(base, tenant=tenant, project=project, component=component)
    now = time.time()

    async def _avg_since(seconds: int) -> Optional[float]:
        try:
            members = await redis.zrangebyscore(key, now - seconds, now)
            if not members:
                return None
            vals = _decode_values(members)
            if not vals:
                return None
            return round(sum(vals) / len(vals), 2)
        except Exception:
            return None

    async def _max_since(seconds: int) -> Optional[float]:
        try:
            members = await redis.zrangebyscore(key, now - seconds, now)
            if not members:
                return None
            vals = _decode_values(members)
            if not vals:
                return None
            return round(max(vals), 2)
        except Exception:
            return None

    return {
        "1m": await _avg_since(60),
        "15m": await _avg_since(900),
        "1h": await _avg_since(retention_sec),
        "max": await _max_since(retention_sec),
    }


async def read_metric_percentiles(
    redis,
    *,
    base: str,
    tenant: str,
    project: str,
    component: str,
    retention_sec: int = DEFAULT_RETENTION_SEC,
    windows: Optional[List[int]] = None,
    percentiles: Optional[List[int]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    if redis is None:
        return {}
    if windows is None:
        windows = [60, 900, retention_sec]
    if percentiles is None:
        percentiles = [50, 95, 99]

    key = metric_key(base, tenant=tenant, project=project, component=component)
    now = time.time()
    results: Dict[str, Dict[str, Optional[float]]] = {}

    for seconds in windows:
        label = "1m" if seconds == 60 else "15m" if seconds == 900 else "1h"
        try:
            members = await redis.zrangebyscore(key, now - seconds, now)
        except Exception:
            members = []
        values = _decode_values(members)
        values.sort()
        bucket: Dict[str, Optional[float]] = {"count": len(values)}
        for p in percentiles:
            bucket[f"p{p}"] = _percentile(values, p)
        bucket["max"] = max(values) if values else None
        results[label] = bucket

    return results


def _decode_values(members: List) -> List[float]:
    vals: List[float] = []
    for m in members:
        raw = m.decode() if isinstance(m, (bytes, bytearray)) else str(m)
        parts = raw.split(":", 1)
        if len(parts) == 2:
            try:
                vals.append(float(parts[1]))
            except Exception:
                continue
    return vals


def _percentile(values: List[float], percentile: int) -> Optional[float]:
    if not values:
        return None
    if percentile <= 0:
        return values[0]
    if percentile >= 100:
        return values[-1]
    k = int((percentile / 100) * len(values))
    if k <= 0:
        return values[0]
    if k >= len(values):
        return values[-1]
    return values[k]
