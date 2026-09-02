# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Upstream OAuth for owner-configured remote MCP connectors."""

from __future__ import annotations

import base64
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote_plus, urlencode, urljoin, urlsplit, urlunsplit

from connection_hub.remote_mcp import (
    AUTH_OAUTH,
    BundleStorageRemoteMCPOAuthStateStore,
    RemoteMCPConnectorService,
    RemoteMCPEndpointPolicy,
    RemoteMCPOAuthCredential,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.remote_mcp.host import (
    GuardedRemoteMCPNetworkBackend,
    KDCubeRemoteMCPSecretStore,
)

DEFAULT_OAUTH_STATE_TTL_SECONDS = 900
DEFAULT_OAUTH_CLIENT_NAME = "KDCube Connection Hub"


class RemoteMCPOAuthFlowError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RemoteMCPOAuthDiscovery:
    authorization_server: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str
    revocation_endpoint: str
    resource: str
    scope: str
    authorization_response_iss_parameter_supported: bool


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _append_query(url: str, params: Mapping[str, str]) -> str:
    parts = urlsplit(url)
    existing = parts.query
    added = urlencode({key: value for key, value in params.items() if value})
    query = f"{existing}&{added}" if existing and added else existing or added
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def _validated_callback_url(value: Any) -> str:
    raw = _clean(value)
    if not raw or len(raw) > 2048:
        raise RemoteMCPOAuthFlowError("oauth_callback_url_invalid")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise RemoteMCPOAuthFlowError("oauth_callback_url_invalid") from exc
    if parsed.scheme.lower() not in {"https", "http"}:
        raise RemoteMCPOAuthFlowError("oauth_callback_url_invalid")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise RemoteMCPOAuthFlowError("oauth_callback_url_invalid")
    if parsed.fragment:
        raise RemoteMCPOAuthFlowError("oauth_callback_url_invalid")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


def _oauth_config(connections: Mapping[str, Any] | None) -> dict[str, Any]:
    root = _mapping(connections)
    remote = _mapping(root.get("remote_mcp"))
    return _mapping(remote.get("oauth"))


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class KDCubeRemoteMCPOAuthService:
    def __init__(
        self,
        *,
        connector_service: RemoteMCPConnectorService,
        state_store: BundleStorageRemoteMCPOAuthStateStore,
        endpoint_policy: RemoteMCPEndpointPolicy,
        connections: Mapping[str, Any] | None,
    ) -> None:
        config = _oauth_config(connections)
        self._connector_service = connector_service
        self._state_store = state_store
        self._endpoint_policy = endpoint_policy
        self._state_ttl_seconds = _positive_int(
            config.get("state_ttl_seconds"), DEFAULT_OAUTH_STATE_TTL_SECONDS
        )
        self._client_name = (
            _clean(config.get("client_name")) or DEFAULT_OAUTH_CLIENT_NAME
        )
        self._enabled = config.get("enabled") is not False

    def _http_transport(self) -> Any:
        import httpx2

        transport = httpx2.AsyncHTTPTransport(trust_env=False, retries=0)
        pool = getattr(transport, "_pool", None)
        inner = getattr(pool, "_network_backend", None)
        if pool is None or inner is None:
            raise RemoteMCPOAuthFlowError("oauth_guarded_transport_unavailable")
        pool._network_backend = GuardedRemoteMCPNetworkBackend(
            inner=inner,
            endpoint_policy=self._endpoint_policy,
        )
        return transport

    def _http_client(self) -> Any:
        import httpx2

        return httpx2.AsyncClient(
            timeout=httpx2.Timeout(30.0, read=60.0),
            follow_redirects=False,
            transport=self._http_transport(),
            trust_env=False,
        )

    async def start(
        self,
        *,
        owner_subject: str,
        label: str,
        endpoint: str,
        callback_url: str,
        client_metadata_url: str = "",
        return_hint: str = "",
        connector_id: str = "",
        expected_revision: int = 0,
    ) -> dict[str, Any]:
        if not self._enabled:
            raise RemoteMCPOAuthFlowError("remote_mcp_oauth_disabled")
        owner = _clean(owner_subject)
        connector_label = _clean(label)
        if not owner:
            raise RemoteMCPOAuthFlowError("oauth_owner_missing")
        existing_connector_id = _clean(connector_id)
        existing_revision = 0
        if existing_connector_id:
            current = await self._connector_service.get(
                owner_subject=owner,
                connector_id=existing_connector_id,
            )
            existing_revision = int(expected_revision or 0)
            if existing_revision != current.revision:
                raise RemoteMCPOAuthFlowError("connector_revision_moved")
            connector_label = current.label
            endpoint = current.endpoint
        if not connector_label or len(connector_label) > 160:
            raise RemoteMCPOAuthFlowError("connector_label_invalid")
        canonical_endpoint = await self._endpoint_policy.validate(endpoint)
        canonical_callback = _validated_callback_url(callback_url)
        await self._state_store.purge_expired()

        discovery, client_info = await self._discover_and_register(
            endpoint=canonical_endpoint,
            callback_url=canonical_callback,
            client_metadata_url=_clean(client_metadata_url),
        )
        from mcp.client.auth import PKCEParameters

        pkce = PKCEParameters.generate()
        transaction = {
            "owner_subject": owner,
            "label": connector_label,
            "endpoint": canonical_endpoint,
            "callback_url": canonical_callback,
            "return_hint": _clean(return_hint),
            "connector_id": existing_connector_id,
            "expected_revision": existing_revision,
            "authorization_server": discovery.authorization_server,
            "authorization_endpoint": discovery.authorization_endpoint,
            "token_endpoint": discovery.token_endpoint,
            "revocation_endpoint": discovery.revocation_endpoint,
            "resource": discovery.resource,
            "scope": discovery.scope,
            "authorization_response_iss_parameter_supported": (
                discovery.authorization_response_iss_parameter_supported
            ),
            "client_id": _clean(client_info.get("client_id")),
            "client_secret": _clean(client_info.get("client_secret")),
            "token_endpoint_auth_method": (
                _clean(client_info.get("token_endpoint_auth_method")) or "none"
            ),
            "code_verifier": pkce.code_verifier,
        }
        handle = await self._state_store.create(
            owner_subject=owner,
            transaction=transaction,
            ttl_seconds=self._state_ttl_seconds,
        )
        params = {
            "response_type": "code",
            "client_id": transaction["client_id"],
            "redirect_uri": canonical_callback,
            "state": handle.state,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
            "resource": discovery.resource,
            "scope": discovery.scope,
        }
        if "offline_access" in discovery.scope.split():
            params["prompt"] = "consent"
        return {
            "authorize_url": _append_query(
                discovery.authorization_endpoint, params
            ),
            "state_id": handle.state_digest,
            "expires_at": handle.expires_at,
            "endpoint": canonical_endpoint,
            "authorization_server": discovery.authorization_server,
        }

    def client_metadata(self, *, callback_url: str) -> dict[str, Any]:
        from mcp.shared.auth import OAuthClientMetadata

        metadata = OAuthClientMetadata(
            client_name=self._client_name,
            redirect_uris=[_validated_callback_url(callback_url)],
            token_endpoint_auth_method="none",
            grant_types=["authorization_code", "refresh_token"],
            application_type="web",
        )
        return metadata.model_dump(by_alias=True, mode="json", exclude_none=True)

    async def complete(
        self,
        *,
        state: str,
        code: str,
        callback_url: str,
        issuer: str = "",
        provider_error: str = "",
    ) -> dict[str, Any]:
        transaction = await self._state_store.consume(state=state)
        expected_callback = _clean(transaction.get("callback_url"))
        actual_callback = _validated_callback_url(callback_url)
        if not secrets.compare_digest(expected_callback, actual_callback):
            raise RemoteMCPOAuthFlowError("oauth_callback_url_mismatch")
        if provider_error:
            raise RemoteMCPOAuthFlowError("oauth_provider_denied")
        authorization_code = _clean(code)
        if not authorization_code:
            raise RemoteMCPOAuthFlowError("oauth_authorization_code_missing")
        self._validate_callback_issuer(transaction, issuer)
        token = await self._exchange_code(
            transaction=transaction,
            code=authorization_code,
        )
        connector_id = _clean(transaction.get("connector_id"))
        if connector_id:
            connector = await self._connector_service.replace_credential(
                owner_subject=_clean(transaction.get("owner_subject")),
                connector_id=connector_id,
                expected_revision=int(transaction.get("expected_revision") or 0),
                credential_mode=AUTH_OAUTH,
                credential_value=token.to_json(),
            )
        else:
            connector = await self._connector_service.create(
                owner_subject=_clean(transaction.get("owner_subject")),
                label=_clean(transaction.get("label")),
                endpoint=_clean(transaction.get("endpoint")),
                credential_mode=AUTH_OAUTH,
                credential_value=token.to_json(),
            )
        return {
            "connector": connector,
            "return_hint": _clean(transaction.get("return_hint")),
        }

    async def _discover_and_register(
        self,
        *,
        endpoint: str,
        callback_url: str,
        client_metadata_url: str,
    ) -> tuple[RemoteMCPOAuthDiscovery, dict[str, Any]]:
        from mcp.client.auth.oauth2 import (
            build_oauth_authorization_server_metadata_discovery_urls,
            build_protected_resource_metadata_discovery_urls,
            check_registration_usable,
            check_resource_allowed,
            extract_resource_metadata_from_www_auth,
            extract_scope_from_www_auth,
            get_client_metadata_scopes,
            is_valid_client_metadata_url,
            resource_url_from_server_url,
            should_use_client_metadata_url,
            validate_metadata_issuer,
            create_client_info_from_metadata_url,
        )
        from mcp.shared.auth import (
            OAuthClientInformationFull,
            OAuthClientMetadata,
            OAuthMetadata,
            ProtectedResourceMetadata,
        )

        async with self._http_client() as client:
            challenge = await self._probe(client=client, endpoint=endpoint)
            prm = None
            for candidate in build_protected_resource_metadata_discovery_urls(
                extract_resource_metadata_from_www_auth(challenge), endpoint
            ):
                response = await client.get(
                    await self._endpoint_policy.validate(candidate),
                    headers={"MCP-Protocol-Version": "2026-07-28"},
                )
                if response.status_code == 200:
                    try:
                        prm = ProtectedResourceMetadata.model_validate_json(
                            await response.aread()
                        )
                    except ValueError:
                        prm = None
                if prm is not None:
                    break
            auth_server = (
                _clean(prm.authorization_servers[0])
                if prm is not None and prm.authorization_servers
                else ""
            )
            if prm is not None and prm.resource:
                requested = resource_url_from_server_url(endpoint)
                if not check_resource_allowed(
                    requested_resource=requested,
                    configured_resource=str(prm.resource),
                ):
                    raise RemoteMCPOAuthFlowError(
                        "oauth_protected_resource_mismatch"
                    )
            oauth_metadata = None
            for candidate in build_oauth_authorization_server_metadata_discovery_urls(
                auth_server or None, endpoint
            ):
                response = await client.get(
                    await self._endpoint_policy.validate(candidate),
                    headers={"MCP-Protocol-Version": "2026-07-28"},
                )
                if response.status_code == 200:
                    try:
                        oauth_metadata = OAuthMetadata.model_validate_json(
                            await response.aread()
                        )
                    except ValueError:
                        oauth_metadata = None
                if oauth_metadata is not None:
                    break
            if oauth_metadata is None:
                raise RemoteMCPOAuthFlowError(
                    "oauth_authorization_server_metadata_unavailable"
                )
            if auth_server:
                try:
                    validate_metadata_issuer(oauth_metadata, auth_server)
                except Exception as exc:
                    raise RemoteMCPOAuthFlowError(
                        "oauth_authorization_server_issuer_mismatch"
                    ) from exc
            else:
                auth_server = str(oauth_metadata.issuer)
            await self._endpoint_policy.validate(auth_server)
            authorization_endpoint = await self._endpoint_policy.validate(
                str(oauth_metadata.authorization_endpoint)
            )
            token_endpoint = await self._endpoint_policy.validate(
                str(oauth_metadata.token_endpoint)
            )
            revocation_endpoint = ""
            if oauth_metadata.revocation_endpoint:
                revocation_endpoint = await self._endpoint_policy.validate(
                    str(oauth_metadata.revocation_endpoint)
                )
            scope = get_client_metadata_scopes(
                extract_scope_from_www_auth(challenge),
                prm,
                oauth_metadata,
                ["authorization_code", "refresh_token"],
            ) or ""
            client_metadata = OAuthClientMetadata(
                client_name=self._client_name,
                redirect_uris=[callback_url],
                token_endpoint_auth_method="none",
                grant_types=["authorization_code", "refresh_token"],
                application_type="web",
                scope=scope or None,
            )
            metadata_url = (
                client_metadata_url
                if is_valid_client_metadata_url(client_metadata_url)
                else ""
            )
            registration_url = ""
            if should_use_client_metadata_url(oauth_metadata, metadata_url or None):
                client_info = create_client_info_from_metadata_url(
                    metadata_url,
                    redirect_uris=client_metadata.redirect_uris,
                )
            else:
                registration_url = (
                    str(oauth_metadata.registration_endpoint)
                    if oauth_metadata.registration_endpoint
                    else urljoin(auth_server.rstrip("/") + "/", "register")
                )
                registration_url = await self._endpoint_policy.validate(
                    registration_url
                )
                registration = await client.post(
                    registration_url,
                    json=client_metadata.model_dump(
                        by_alias=True, mode="json", exclude_none=True
                    ),
                    headers={"Content-Type": "application/json"},
                )
                if registration.status_code not in {200, 201}:
                    await registration.aread()
                    raise RemoteMCPOAuthFlowError(
                        "oauth_dynamic_client_registration_failed"
                    )
                try:
                    client_info = OAuthClientInformationFull.model_validate_json(
                        await registration.aread()
                    )
                    check_registration_usable(client_info)
                except Exception as exc:
                    raise RemoteMCPOAuthFlowError(
                        "oauth_dynamic_client_registration_invalid"
                    ) from exc
            registered_redirects = {
                str(item) for item in (client_info.redirect_uris or ())
            }
            if registered_redirects and callback_url not in registered_redirects:
                raise RemoteMCPOAuthFlowError(
                    "oauth_registered_redirect_uri_mismatch"
                )
            resource = (
                str(prm.resource)
                if prm is not None and prm.resource
                else resource_url_from_server_url(endpoint)
            )
            discovery = RemoteMCPOAuthDiscovery(
                authorization_server=auth_server,
                authorization_endpoint=authorization_endpoint,
                token_endpoint=token_endpoint,
                registration_endpoint=registration_url,
                revocation_endpoint=revocation_endpoint,
                resource=resource,
                scope=scope,
                authorization_response_iss_parameter_supported=bool(
                    oauth_metadata.authorization_response_iss_parameter_supported
                ),
            )
            return discovery, client_info.model_dump(
                by_alias=True, mode="json", exclude_none=True
            )

    async def _probe(self, *, client: Any, endpoint: str) -> Any:
        from mcp.types import LATEST_PROTOCOL_VERSION

        response = await client.post(
            endpoint,
            json={
                "jsonrpc": "2.0",
                "id": "connection-hub-oauth-discovery",
                "method": "initialize",
                "params": {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "kdcube-connection-hub-oauth-discovery",
                        "version": "1",
                    },
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION,
            },
        )
        if response.status_code != 401:
            await response.aread()
            raise RemoteMCPOAuthFlowError("oauth_challenge_not_advertised")
        return response

    @staticmethod
    def _validate_callback_issuer(
        transaction: Mapping[str, Any], issuer: str
    ) -> None:
        expected = _clean(transaction.get("authorization_server"))
        received = _clean(issuer)
        advertised = bool(
            transaction.get("authorization_response_iss_parameter_supported")
        )
        if received and not secrets.compare_digest(received, expected):
            raise RemoteMCPOAuthFlowError("oauth_callback_issuer_mismatch")
        if advertised and not received:
            raise RemoteMCPOAuthFlowError("oauth_callback_issuer_missing")

    async def _exchange_code(
        self,
        *,
        transaction: Mapping[str, Any],
        code: str,
    ) -> RemoteMCPOAuthCredential:
        token_endpoint = await self._endpoint_policy.validate(
            transaction.get("token_endpoint")
        )
        client_id = _clean(transaction.get("client_id"))
        client_secret = _clean(transaction.get("client_secret"))
        method = _clean(transaction.get("token_endpoint_auth_method")) or "none"
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _clean(transaction.get("callback_url")),
            "client_id": client_id,
            "code_verifier": _clean(transaction.get("code_verifier")),
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if transaction.get("resource"):
            data["resource"] = _clean(transaction.get("resource"))
        if method == "client_secret_post":
            data["client_secret"] = client_secret
        elif method == "client_secret_basic":
            encoded_id = quote_plus(client_id, safe="")
            encoded_secret = quote_plus(client_secret, safe="")
            basic = base64.b64encode(
                f"{encoded_id}:{encoded_secret}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {basic}"
        elif method != "none":
            raise RemoteMCPOAuthFlowError("oauth_token_auth_method_unsupported")
        async with self._http_client() as client:
            response = await client.post(
                token_endpoint,
                data=data,
                headers=headers,
            )
        if response.status_code not in {200, 201}:
            await response.aread()
            raise RemoteMCPOAuthFlowError("oauth_token_exchange_failed")
        try:
            payload = json.loads(await response.aread())
        except (TypeError, ValueError) as exc:
            raise RemoteMCPOAuthFlowError("oauth_token_response_invalid") from exc
        if not isinstance(payload, Mapping):
            raise RemoteMCPOAuthFlowError("oauth_token_response_invalid")
        moment = int(time.time())
        try:
            expires_in = int(payload.get("expires_in") or 0)
        except (TypeError, ValueError):
            expires_in = 0
        token = RemoteMCPOAuthCredential(
            access_token=_clean(payload.get("access_token")),
            refresh_token=_clean(payload.get("refresh_token")),
            expires_at=moment + expires_in if expires_in > 0 else 0,
            scope=_clean(payload.get("scope")) or _clean(transaction.get("scope")),
            resource=_clean(transaction.get("resource")),
            authorization_server=_clean(
                transaction.get("authorization_server")
            ),
            token_endpoint=token_endpoint,
            revocation_endpoint=_clean(transaction.get("revocation_endpoint")),
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=method,
            issued_at=moment,
        )
        token.verify()
        return token


def build_remote_mcp_oauth_service(
    *,
    storage_root: Any,
    bundle_id: str,
    connections: Mapping[str, Any] | None,
    connector_service: RemoteMCPConnectorService,
    endpoint_policy: RemoteMCPEndpointPolicy,
) -> KDCubeRemoteMCPOAuthService:
    secret_store = KDCubeRemoteMCPSecretStore(bundle_id=bundle_id)
    return KDCubeRemoteMCPOAuthService(
        connector_service=connector_service,
        state_store=BundleStorageRemoteMCPOAuthStateStore(
            storage_root, secret_store=secret_store
        ),
        endpoint_policy=endpoint_policy,
        connections=connections,
    )


__all__ = [
    "DEFAULT_OAUTH_CLIENT_NAME",
    "DEFAULT_OAUTH_STATE_TTL_SECONDS",
    "KDCubeRemoteMCPOAuthService",
    "RemoteMCPOAuthDiscovery",
    "RemoteMCPOAuthFlowError",
    "build_remote_mcp_oauth_service",
]
