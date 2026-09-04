"""Independent infrastructure helpers for the direct agent examples."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


def _section(config: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping):
            raise ValueError(f"configuration section {'.'.join(path)!r} must be a mapping")
        value = value.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {'.'.join(path)!r} must be a mapping")
    return value


def _secret(section: Mapping[str, Any], *, field: str, check_only: bool) -> str:
    ref = str(section.get(f"{field}_ref") or "").strip()
    if not ref:
        raise ValueError(f"{field}_ref must name a secret environment variable")
    value = os.environ.get(ref, "")
    if not value and not check_only:
        raise RuntimeError(f"secret {ref!r} is not set")
    return value or "check-only"


def redis_url(config: Mapping[str, Any], *, check_only: bool = False) -> str:
    redis = _section(config, "infra", "redis")
    password = quote(_secret(redis, field="password", check_only=check_only), safe="")
    host = str(redis.get("host") or "127.0.0.1")
    port = int(redis.get("port") or 56379)
    database = int(redis.get("database") or 0)
    return f"redis://:{password}@{host}:{port}/{database}"


def redis_ttl_seconds(config: Mapping[str, Any]) -> int:
    redis = _section(config, "infra", "redis")
    return int(redis.get("turn_cache_ttl_seconds") or 3600)


def postgres_url(config: Mapping[str, Any], *, check_only: bool = False) -> str:
    postgres = _section(config, "infra", "postgres")
    password = quote(_secret(postgres, field="password", check_only=check_only), safe="")
    user = quote(str(postgres.get("user") or "kdcube_agents"), safe="")
    host = str(postgres.get("host") or "127.0.0.1")
    port = int(postgres.get("port") or 55432)
    database = quote(str(postgres.get("database") or "kdcube_agents"), safe="")
    sslmode = quote(str(postgres.get("sslmode") or "disable"), safe="")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"


def postgres_label(config: Mapping[str, Any]) -> str:
    postgres = _section(config, "infra", "postgres")
    return (
        f"{postgres.get('host') or '127.0.0.1'}:"
        f"{int(postgres.get('port') or 55432)}/"
        f"{postgres.get('database') or 'kdcube_agents'}"
    )


def storage_uri(config: Mapping[str, Any], *, config_path: Path) -> str:
    storage = _section(config, "infra", "storage")
    raw = str(storage.get("uri") or "").strip()
    if not raw:
        raise ValueError("infra.storage.uri is required")
    parsed = urlparse(raw)
    if parsed.scheme:
        return raw
    return (config_path.parent / raw).expanduser().resolve().as_uri()


def direct_harness_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    project: str,
    bundle_id: str,
    agent_id: str,
    check_only: bool = False,
):
    """Project the example descriptor into the SDK's direct-host contract."""
    from kdcube_ai_app.apps.chat.sdk.runtime.direct_harness import (
        DirectAgentHarnessConfig,
    )

    return DirectAgentHarnessConfig(
        tenant="standalone",
        project=project,
        user_id="demo-user",
        user_type="regular",
        session_id="local-session",
        bundle_id=bundle_id,
        agent_id=agent_id,
        postgres_url=postgres_url(config, check_only=check_only),
        redis_url=redis_url(config, check_only=check_only),
        storage_uri=storage_uri(config, config_path=config_path),
        turn_cache_ttl_seconds=redis_ttl_seconds(config),
    )
