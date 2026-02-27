# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""
Print effective gateway pool limits for the current process.

Usage:
  python -m kdcube_ai_app.infra.tools.gateway_config_dump
  python -m kdcube_ai_app.infra.tools.gateway_config_dump --json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from kdcube_ai_app.infra.gateway.config import get_gateway_config


def _effective_pg_pool(cfg) -> Dict[str, Any]:
    max_size = cfg.pools.pg_pool_max_size
    if max_size is None:
        max_size = cfg.service_capacity.concurrent_requests_per_process
    min_size = cfg.pools.pg_pool_min_size if cfg.pools.pg_pool_min_size is not None else 0
    return {
        "min_size": min_size,
        "max_size": max_size,
    }


def _effective_redis_pool(cfg) -> Dict[str, Any]:
    max_conn = cfg.pools.redis_max_connections
    approx_total = None
    if max_conn is not None:
        # async + async_decode + sync
        approx_total = 3 * max_conn
    return {
        "max_connections": max_conn,
        "approx_total_connections_per_process": approx_total,
    }


def _effective_limits(cfg) -> Dict[str, Any]:
    return {
        "max_sse_connections_per_instance": cfg.limits.max_sse_connections_per_instance,
        "max_integrations_ops_concurrency": cfg.limits.max_integrations_ops_concurrency,
    }


def _effective_redis_cfg(cfg) -> Dict[str, Any]:
    return {
        "sse_stats_ttl_seconds": cfg.redis.sse_stats_ttl_seconds,
        "sse_stats_max_age_seconds": cfg.redis.sse_stats_max_age_seconds,
    }


def _payload(cfg) -> Dict[str, Any]:
    component = (os.getenv("GATEWAY_COMPONENT") or "ingress").strip().lower()
    return {
        "component": component,
        "tenant_id": cfg.tenant_id,
        "project_id": cfg.project_id,
        "instance_id": cfg.instance_id,
        "service_capacity": {
            "concurrent_requests_per_process": cfg.service_capacity.concurrent_requests_per_process,
            "processes_per_instance": cfg.service_capacity.processes_per_instance,
            "avg_processing_time_seconds": cfg.service_capacity.avg_processing_time_seconds,
        },
        "pools": {
            "pg_pool": _effective_pg_pool(cfg),
            "redis_pool": _effective_redis_pool(cfg),
            "pg_max_connections": cfg.pools.pg_max_connections,
        },
        "limits": _effective_limits(cfg),
        "redis": _effective_redis_cfg(cfg),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    cfg = get_gateway_config()
    data = _payload(cfg)

    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    print("Gateway config (effective per-process limits)")
    print(f"  component: {data['component']}")
    print(f"  tenant/project: {data['tenant_id']} / {data['project_id']}")
    print(f"  instance: {data['instance_id']}")
    print("")
    print("Service capacity")
    sc = data["service_capacity"]
    print(f"  concurrent_requests_per_process: {sc['concurrent_requests_per_process']}")
    print(f"  processes_per_instance: {sc['processes_per_instance']}")
    print(f"  avg_processing_time_seconds: {sc['avg_processing_time_seconds']}")
    print("")
    print("Pools")
    pg = data["pools"]["pg_pool"]
    print(f"  pg_pool_min_size: {pg['min_size']}")
    print(f"  pg_pool_max_size: {pg['max_size']}")
    print(f"  pg_max_connections (reference): {data['pools']['pg_max_connections']}")
    rd = data["pools"]["redis_pool"]
    print(f"  redis_max_connections: {rd['max_connections']}")
    print(f"  approx_redis_conns_per_process: {rd['approx_total_connections_per_process']}")
    print("")
    print("Limits")
    limits = data["limits"]
    print(f"  max_sse_connections_per_instance: {limits['max_sse_connections_per_instance']}")
    print(f"  max_integrations_ops_concurrency: {limits['max_integrations_ops_concurrency']}")
    print("")
    print("Redis settings")
    rcfg = data["redis"]
    print(f"  sse_stats_ttl_seconds: {rcfg['sse_stats_ttl_seconds']}")
    print(f"  sse_stats_max_age_seconds: {rcfg['sse_stats_max_age_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
