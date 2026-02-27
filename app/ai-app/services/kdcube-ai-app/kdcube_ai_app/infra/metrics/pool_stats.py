# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from typing import Any, Dict, Optional


def pg_pool_stats(pool) -> Optional[Dict[str, Any]]:
    if not pool:
        return None
    stats: Dict[str, Any] = {}
    for name, key in (("min_size", "_minsize"), ("max_size", "_maxsize")):
        try:
            stats[name] = getattr(pool, key)
        except Exception:
            stats[name] = None
    size = None
    idle = None
    try:
        if hasattr(pool, "get_size"):
            size = pool.get_size()
        else:
            size = getattr(pool, "_size")
    except Exception:
        size = None
    try:
        if hasattr(pool, "get_idle_size"):
            idle = pool.get_idle_size()
        else:
            q = getattr(pool, "_queue", None)
            idle = q.qsize() if q is not None else None
    except Exception:
        idle = None
    stats["size"] = size
    stats["available"] = idle
    if size is not None and idle is not None:
        try:
            stats["in_use"] = int(size) - int(idle)
        except Exception:
            stats["in_use"] = None
    return stats


def redis_pool_stats(client) -> Optional[Dict[str, Any]]:
    if not client:
        return None
    pool = getattr(client, "connection_pool", None)
    if not pool:
        return None
    try:
        in_use = len(getattr(pool, "_in_use_connections", []) or [])
        available = len(getattr(pool, "_available_connections", []) or [])
    except Exception:
        in_use = None
        available = None
    return {
        "max_connections": getattr(pool, "max_connections", None),
        "in_use": in_use,
        "available": available,
        "total": (in_use + available) if (in_use is not None and available is not None) else None,
    }


def build_pool_metadata(pg_pool=None, redis_clients: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "pools": {
            "postgres": pg_pool_stats(pg_pool),
            "redis": {
                "async": redis_pool_stats((redis_clients or {}).get("async")),
                "async_decode": redis_pool_stats((redis_clients or {}).get("async_decode")),
                "sync": redis_pool_stats((redis_clients or {}).get("sync")),
            },
        }
    }
