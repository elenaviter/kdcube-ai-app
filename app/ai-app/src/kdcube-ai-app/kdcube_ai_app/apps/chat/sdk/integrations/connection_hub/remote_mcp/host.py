# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube storage, secret, lock, and MCP-client bindings for remote MCP."""

from __future__ import annotations

import base64
import json
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, Mapping
from urllib.parse import quote_plus

from connection_hub.remote_mcp import (
    AUTH_OAUTH,
    BundleStorageRemoteMCPConnectorStore,
    RemoteMCPConnectorConflict,
    RemoteMCPConnectorService,
    RemoteMCPDiscovery,
    RemoteMCPEndpointPolicy,
    RemoteMCPOAuthCredential,
)
from connection_hub.remote_mcp.store import owner_hash_for
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
DEFAULT_OAUTH_EXPIRY_LEEWAY_SECONDS = 60
OAUTH_REFRESH_LOCK_FILENAME = ".oauth-refresh.lock"


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
        secret_store: KDCubeRemoteMCPSecretStore | None = None,
        connector_store: BundleStorageRemoteMCPConnectorStore | None = None,
        read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
        oauth_expiry_leeway_seconds: int = DEFAULT_OAUTH_EXPIRY_LEEWAY_SECONDS,
    ) -> None:
        self._endpoint_policy = endpoint_policy
        self._secret_store = secret_store
        self._connector_store = connector_store
        self._read_timeout_seconds = float(read_timeout_seconds)
        self._max_result_bytes = int(max_result_bytes)
        self._oauth_expiry_leeway_seconds = int(oauth_expiry_leeway_seconds)

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

    async def prepare_headers(
        self,
        *,
        connector_id: str,
        credential: Any,
        owner_subject: str,
        credential_ref: str,
    ) -> dict[str, str]:
        if str(getattr(credential, "mode", "") or "").strip().lower() != AUTH_OAUTH:
            return dict(credential.request_headers())
        oauth = RemoteMCPOAuthCredential.from_json(
            str(getattr(credential, "value", "") or "")
        )
        if not oauth.is_expiring(
            now=int(time.time()),
            leeway_seconds=self._oauth_expiry_leeway_seconds,
        ):
            return oauth.request_headers()
        if (
            not owner_subject
            or not credential_ref
            or self._secret_store is None
            or self._connector_store is None
        ):
            raise RemoteMCPConnectorConflict(
                "connector_oauth_reauthorization_required"
            )
        lock_path = (
            self._connector_store.connector_path(
                owner_hash=owner_hash_for(owner_subject),
                connector_id=connector_id,
            )
            / OAUTH_REFRESH_LOCK_FILENAME
        )
        try:
            async with observed_file_lock_async(
                lock_path=lock_path,
                resource_id=f"remote-mcp-oauth:{connector_id}",
                operation="remote-mcp-oauth-refresh",
                wait_seconds=30.0,
            ):
                latest_raw = await self._secret_store.get(
                    owner_subject=owner_subject,
                    secret_ref=credential_ref,
                )
                if not latest_raw:
                    raise RemoteMCPConnectorConflict(
                        "connector_credential_unavailable"
                    )
                latest = RemoteMCPOAuthCredential.from_json(latest_raw)
                if latest.is_expiring(
                    now=int(time.time()),
                    leeway_seconds=self._oauth_expiry_leeway_seconds,
                ):
                    latest = await self._refresh_oauth(latest)
                    await self._secret_store.set(
                        owner_subject=owner_subject,
                        secret_ref=credential_ref,
                        value=latest.to_json(),
                    )
                return latest.request_headers()
        except ObservedFileLockTimeout as exc:
            raise RemoteMCPConnectorConflict(
                "connector_oauth_refresh_lock_timeout"
            ) from exc

    async def revoke_credential(
        self,
        *,
        connector_id: str,
        endpoint: str,
        transport: str,
        credential: Any,
        owner_subject: str,
        credential_ref: str,
    ) -> bool:
        del connector_id, endpoint, transport
        if str(getattr(credential, "mode", "") or "").strip().lower() != AUTH_OAUTH:
            return False
        if self._secret_store is None or not owner_subject or not credential_ref:
            return False
        latest_raw = await self._secret_store.get(
            owner_subject=owner_subject,
            secret_ref=credential_ref,
        )
        oauth = RemoteMCPOAuthCredential.from_json(
            latest_raw or str(getattr(credential, "value", "") or "")
        )
        if not oauth.revocation_endpoint:
            return False
        import httpx2

        revocation_endpoint = await self._endpoint_policy.validate(
            oauth.revocation_endpoint
        )
        tokens = [
            (oauth.refresh_token, "refresh_token"),
            (oauth.access_token, "access_token"),
        ]
        attempted = False
        async with httpx2.AsyncClient(
            timeout=httpx2.Timeout(30.0, read=60.0),
            follow_redirects=False,
            transport=self._http_transport(),
            trust_env=False,
        ) as client:
            for token, hint in tokens:
                if not token:
                    continue
                data = {
                    "token": token,
                    "token_type_hint": hint,
                    "client_id": oauth.client_id,
                }
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                if oauth.token_endpoint_auth_method == "client_secret_post":
                    data["client_secret"] = oauth.client_secret
                elif oauth.token_endpoint_auth_method == "client_secret_basic":
                    encoded = base64.b64encode(
                        (
                            f"{quote_plus(oauth.client_id, safe='')}:"
                            f"{quote_plus(oauth.client_secret, safe='')}"
                        ).encode("utf-8")
                    ).decode("ascii")
                    headers["Authorization"] = f"Basic {encoded}"
                response = await client.post(
                    revocation_endpoint,
                    data=data,
                    headers=headers,
                )
                await response.aread()
                attempted = True
                if response.status_code not in {200, 201, 204}:
                    raise RemoteMCPConnectorConflict(
                        "connector_oauth_upstream_revocation_failed"
                    )
        return attempted

    async def _refresh_oauth(
        self, credential: RemoteMCPOAuthCredential
    ) -> RemoteMCPOAuthCredential:
        import httpx2

        if not credential.refresh_token:
            raise RemoteMCPConnectorConflict(
                "connector_oauth_reauthorization_required"
            )
        token_endpoint = await self._endpoint_policy.validate(
            credential.token_endpoint
        )
        data = {
            "grant_type": "refresh_token",
            "refresh_token": credential.refresh_token,
            "client_id": credential.client_id,
        }
        if credential.resource:
            data["resource"] = credential.resource
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        method = credential.token_endpoint_auth_method
        if method == "client_secret_post":
            data["client_secret"] = credential.client_secret
        elif method == "client_secret_basic":
            encoded = base64.b64encode(
                (
                    f"{quote_plus(credential.client_id, safe='')}:"
                    f"{quote_plus(credential.client_secret, safe='')}"
                ).encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        elif method != "none":
            raise RemoteMCPConnectorConflict(
                "connector_oauth_reauthorization_required"
            )
        async with httpx2.AsyncClient(
            timeout=httpx2.Timeout(30.0, read=60.0),
            follow_redirects=False,
            transport=self._http_transport(),
            trust_env=False,
        ) as client:
            response = await client.post(
                token_endpoint,
                data=data,
                headers=headers,
            )
            raw = await response.aread()
        if response.status_code not in {200, 201}:
            raise RemoteMCPConnectorConflict(
                "connector_oauth_reauthorization_required"
            )
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RemoteMCPConnectorConflict(
                "connector_oauth_reauthorization_required"
            ) from exc
        if not isinstance(payload, Mapping) or not str(
            payload.get("access_token") or ""
        ).strip():
            raise RemoteMCPConnectorConflict(
                "connector_oauth_reauthorization_required"
            )
        try:
            expires_in = int(payload.get("expires_in") or 0)
        except (TypeError, ValueError):
            expires_in = 0
        moment = int(time.time())
        refreshed = replace(
            credential,
            access_token=str(payload.get("access_token") or "").strip(),
            token_type=str(payload.get("token_type") or "Bearer").strip(),
            refresh_token=(
                str(payload.get("refresh_token") or "").strip()
                or credential.refresh_token
            ),
            expires_at=moment + expires_in if expires_in > 0 else 0,
            scope=(
                str(payload.get("scope") or "").strip() or credential.scope
            ),
            issued_at=moment,
        )
        refreshed.verify()
        return refreshed


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
    connector_store = BundleStorageRemoteMCPConnectorStore(storage_root)
    secret_store = KDCubeRemoteMCPSecretStore(bundle_id=bundle_id)
    return RemoteMCPConnectorService(
        store=connector_store,
        secret_store=secret_store,
        transport=KDCubeRemoteMCPTransport(
            endpoint_policy=endpoint_policy,
            secret_store=secret_store,
            connector_store=connector_store,
            read_timeout_seconds=_positive_float(
                config.get("read_timeout_seconds"), DEFAULT_READ_TIMEOUT_SECONDS
            ),
            max_result_bytes=_positive_int(
                config.get("max_result_bytes"), DEFAULT_MAX_RESULT_BYTES
            ),
            oauth_expiry_leeway_seconds=_positive_int(
                _mapping(config.get("oauth")).get("expiry_leeway_seconds"),
                DEFAULT_OAUTH_EXPIRY_LEEWAY_SECONDS,
            ),
        ),
        endpoint_policy=endpoint_policy,
        mutation_lock=_kdcube_remote_mcp_mutation_lock,
    )


__all__ = [
    "DEFAULT_MAX_RESULT_BYTES",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "DEFAULT_OAUTH_EXPIRY_LEEWAY_SECONDS",
    "GuardedRemoteMCPNetworkBackend",
    "KDCubeRemoteMCPSecretStore",
    "KDCubeRemoteMCPTransport",
    "OAUTH_REFRESH_LOCK_FILENAME",
    "build_remote_mcp_connector_service",
    "remote_mcp_endpoint_policy",
]
