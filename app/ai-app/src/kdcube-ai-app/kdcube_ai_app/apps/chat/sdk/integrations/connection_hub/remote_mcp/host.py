# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube storage, secret, lock, and MCP-client bindings for remote MCP."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Mapping

from connection_hub.remote_mcp import (
    BundleStorageRemoteMCPConnectorStore,
    RemoteMCPConnectorConflict,
    RemoteMCPConnectorService,
    RemoteMCPDiscovery,
    RemoteMCPEndpointPolicy,
)
from kdcube_ai_app.apps.chat.sdk.config import (
    delete_user_secret,
    get_secret,
    set_user_secret,
)
from kdcube_ai_app.apps.chat.sdk.runtime.mcp.client import (
    mcp_tool_schema,
    normalize_mcp_tool_result,
    open_mcp_client,
)
from kdcube_ai_app.storage.observed_file_locks import (
    ObservedFileLockTimeout,
    observed_file_lock_async,
)

DEFAULT_MAX_RESULT_BYTES = 2 * 1024 * 1024
DEFAULT_READ_TIMEOUT_SECONDS = 60.0


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _remote_mcp_config(connections: Mapping[str, Any] | None) -> Mapping[str, Any]:
    root = connections if isinstance(connections, Mapping) else {}
    value = root.get("remote_mcp")
    return value if isinstance(value, Mapping) else {}


def remote_mcp_endpoint_policy(
    connections: Mapping[str, Any] | None,
) -> RemoteMCPEndpointPolicy:
    config = _remote_mcp_config(connections)
    outbound = config.get("outbound")
    outbound = outbound if isinstance(outbound, Mapping) else {}
    raw_hosts = outbound.get("allowed_hosts")
    hosts = (
        frozenset(
            str(item or "").strip().lower().rstrip(".")
            for item in raw_hosts
            if str(item or "").strip()
        )
        if isinstance(raw_hosts, (list, tuple, set))
        else frozenset()
    )
    return RemoteMCPEndpointPolicy(
        allow_http=_bool(outbound.get("allow_http"), False),
        allow_private_networks=_bool(
            outbound.get("allow_private_networks"), False
        ),
        allowed_hosts=hosts,
    )


class KDCubeRemoteMCPSecretStore:
    def __init__(self, *, bundle_id: str) -> None:
        self._bundle_id = str(bundle_id or "").strip()

    async def set(
        self, *, owner_subject: str, secret_ref: str, value: str
    ) -> None:
        await set_user_secret(
            secret_ref,
            value,
            user_id=owner_subject,
            bundle_id=self._bundle_id,
        )

    async def get(self, *, owner_subject: str, secret_ref: str) -> str | None:
        return await get_secret(
            f"u:{secret_ref}",
            user_id=owner_subject,
            bundle_id=self._bundle_id,
        )

    async def delete(self, *, owner_subject: str, secret_ref: str) -> None:
        await delete_user_secret(
            secret_ref,
            user_id=owner_subject,
            bundle_id=self._bundle_id,
        )


class GuardedRemoteMCPNetworkBackend:
    """Resolve, validate, and pin the address used by each outbound socket."""

    def __init__(self, *, inner: Any, endpoint_policy: RemoteMCPEndpointPolicy) -> None:
        self._inner = inner
        self._endpoint_policy = endpoint_policy

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        addresses = await self._endpoint_policy.connect_addresses(host, port)
        last_error: Exception | None = None
        for address in addresses:
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("remote_mcp_endpoint_has_no_connectable_address")

    async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("remote_mcp_unix_socket_forbidden")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class KDCubeRemoteMCPTransport:
    def __init__(
        self,
        *,
        endpoint_policy: RemoteMCPEndpointPolicy,
        read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    ) -> None:
        self._endpoint_policy = endpoint_policy
        self._read_timeout_seconds = float(read_timeout_seconds)
        self._max_result_bytes = int(max_result_bytes)

    def _http_transport(self) -> Any:
        import httpx2

        transport = httpx2.AsyncHTTPTransport(trust_env=False, retries=0)
        pool = getattr(transport, "_pool", None)
        inner = getattr(pool, "_network_backend", None)
        if pool is None or inner is None:
            raise RuntimeError("remote_mcp_guarded_transport_unavailable")
        pool._network_backend = GuardedRemoteMCPNetworkBackend(
            inner=inner,
            endpoint_policy=self._endpoint_policy,
        )
        return transport

    async def discover(
        self,
        *,
        connector_id: str,
        endpoint: str,
        transport: str,
        headers: Mapping[str, str],
    ) -> RemoteMCPDiscovery:
        async with open_mcp_client(
            transport=transport,
            endpoint=endpoint,
            headers=headers,
            read_timeout_seconds=self._read_timeout_seconds,
            follow_redirects=False,
            http_transport=self._http_transport(),
            trust_env=False,
        ) as client:
            response = await client.list_tools()
            tools = [mcp_tool_schema(tool) for tool in (response.tools or ())]
            server_info = getattr(client, "server_info", None)
            return RemoteMCPDiscovery.build(
                connector_id=connector_id,
                tools=tools,
                server_name=getattr(server_info, "name", "") or "",
                server_version=getattr(server_info, "version", "") or "",
                protocol_version=getattr(client, "protocol_version", "") or "",
            )

    async def call_tool(
        self,
        *,
        connector_id: str,
        endpoint: str,
        transport: str,
        headers: Mapping[str, str],
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        del connector_id
        async with open_mcp_client(
            transport=transport,
            endpoint=endpoint,
            headers=headers,
            read_timeout_seconds=self._read_timeout_seconds,
            follow_redirects=False,
            http_transport=self._http_transport(),
            trust_env=False,
        ) as client:
            result = normalize_mcp_tool_result(
                await client.call_tool(tool_name, dict(arguments or {}))
            )
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        if len(encoded) > self._max_result_bytes:
            raise ValueError("remote_mcp_result_too_large")
        return result


@asynccontextmanager
async def _kdcube_remote_mcp_mutation_lock(**kwargs: Any):
    try:
        async with observed_file_lock_async(**kwargs) as metadata:
            yield metadata
    except ObservedFileLockTimeout as exc:
        raise RemoteMCPConnectorConflict("connector_mutation_lock_timeout") from exc


def build_remote_mcp_connector_service(
    *,
    storage_root: Any,
    bundle_id: str,
    connections: Mapping[str, Any] | None,
) -> RemoteMCPConnectorService:
    config = _remote_mcp_config(connections)
    endpoint_policy = remote_mcp_endpoint_policy(connections)
    return RemoteMCPConnectorService(
        store=BundleStorageRemoteMCPConnectorStore(storage_root),
        secret_store=KDCubeRemoteMCPSecretStore(bundle_id=bundle_id),
        transport=KDCubeRemoteMCPTransport(
            endpoint_policy=endpoint_policy,
            read_timeout_seconds=_positive_float(
                config.get("read_timeout_seconds"), DEFAULT_READ_TIMEOUT_SECONDS
            ),
            max_result_bytes=_positive_int(
                config.get("max_result_bytes"), DEFAULT_MAX_RESULT_BYTES
            ),
        ),
        endpoint_policy=endpoint_policy,
        mutation_lock=_kdcube_remote_mcp_mutation_lock,
    )


__all__ = [
    "DEFAULT_MAX_RESULT_BYTES",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "GuardedRemoteMCPNetworkBackend",
    "KDCubeRemoteMCPSecretStore",
    "KDCubeRemoteMCPTransport",
    "build_remote_mcp_connector_service",
    "remote_mcp_endpoint_policy",
]
