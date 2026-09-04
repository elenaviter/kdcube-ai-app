# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube host bindings for Connection Hub delegated MCP resolution."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional


_core = import_module("connection_hub.delegated_mcp")


def kdcube_runtime_local_base() -> str:
    try:
        from kdcube_ai_app.apps.chat.sdk.config import get_settings

        port = int(getattr(get_settings(), "CHAT_PROCESSOR_PORT", None) or 8020)
    except Exception:
        port = 8020
    return f"http://127.0.0.1:{port}"


# Kept for callers outside this checkout that imported the earlier private
# helper before the public host adapter was added.
_kdcube_runtime_local_base = kdcube_runtime_local_base


async def _kdcube_default_minter(
    sub: str,
    scopes: List[str],
    *,
    client_id: str,
    ttl_seconds: Optional[int],
) -> Mapping[str, Any]:
    from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.oauth.grants import (
        mint_delegated_client_access_token,
    )

    kwargs: Dict[str, Any] = {"client_id": client_id}
    if ttl_seconds:
        kwargs["ttl_seconds"] = int(ttl_seconds)
    return await mint_delegated_client_access_token(sub, scopes, **kwargs)


def self_hosted_url(
    conn: Mapping[str, Any],
    url: str,
    *,
    runtime_local_base: str = "",
) -> str:
    return _core.self_hosted_url(
        conn,
        url,
        runtime_local_base=runtime_local_base or kdcube_runtime_local_base(),
    )


async def resolve_mcp_server_map(
    connections: List[Dict[str, Any]],
    *,
    user_sub: Optional[str] = None,
    minter: Any = None,
    client_id: str = "kdcube-agent",
    ttl_seconds: Optional[int] = None,
    consent_gate: Any = None,
    bearer_provider: Any = None,
    drop_sink: Optional[Dict[str, str]] = None,
    runtime_local_base: str = "",
) -> Dict[str, Dict[str, Any]]:
    return await _core.resolve_mcp_server_map(
        connections,
        user_sub=user_sub,
        minter=minter or _kdcube_default_minter,
        client_id=client_id,
        ttl_seconds=ttl_seconds,
        consent_gate=consent_gate,
        bearer_provider=bearer_provider,
        drop_sink=drop_sink,
        runtime_local_base=runtime_local_base or kdcube_runtime_local_base(),
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = sorted(
    {name for name in dir(_core) if not name.startswith("_")}
    | {"kdcube_runtime_local_base"}
)
