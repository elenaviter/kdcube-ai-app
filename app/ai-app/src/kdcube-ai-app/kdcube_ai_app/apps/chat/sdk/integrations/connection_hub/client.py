# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Consumer SDK for the `connections` named service.

``ConnectionsClient`` wraps a ``NamedServiceClient`` and exposes typed
convenience methods. It works identically over the local (in-process) and API
(HTTP) transports — the transport is a property of the wrapped client.
"""

from __future__ import annotations

from typing import Any

from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceClient,
    NamedServiceRegistry,
    NamedServiceRequest,
    NamedServiceResponse,
    TRANSPORT_LOCAL,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.authority_registry_client import AuthorityRegistryClient
from connection_hub.client import (
    ConnectionsClient as PortableConnectionsClient,
    ConnectionsError,
)
from connection_hub.contract import NAMESPACE, Connection


class _NamedServiceTransport:
    """Adapt KDCube named-service calls to Connection Hub's operation transport."""

    def __init__(self, client: NamedServiceClient) -> None:
        self._client = client

    async def call(self, operation: str, payload: dict[str, Any]) -> NamedServiceResponse:
        return await self._client.call(
            NamedServiceRequest(
                operation=operation,
                namespace=NAMESPACE,
                payload=payload,
            )
        )


class ConnectionsClient(PortableConnectionsClient):
    """Typed client for the `connections` named service.

    Construct either from a registry (+ optional transport / auth) or from an
    already-built ``NamedServiceClient``::

        ConnectionsClient(registry, transport=TRANSPORT_API)
        ConnectionsClient(client=named_service_client)
    """

    def __init__(
        self,
        registry: NamedServiceRegistry | None = None,
        *,
        client: NamedServiceClient | None = None,
        transport: str = TRANSPORT_LOCAL,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        elif registry is not None:
            self._client = NamedServiceClient(registry, transport=transport, **client_kwargs)
        else:
            raise ValueError("ConnectionsClient requires a registry or a NamedServiceClient")
        super().__init__(_NamedServiceTransport(self._client))


class ConnectionHubClient:
    """Facade for SDK-owned Connection Hub runtime capabilities.

    This is the boundary general platform code should use when it needs a
    Connection Hub answer. The implementation may use descriptor-backed bundle
    props, Redis, or an explicit in-memory registry, but callers should not
    inspect Connection Hub descriptors directly and should not call the
    Connection Hub bundle just to resolve SDK-owned registry metadata.
    """

    def __init__(
        self,
        entrypoint: Any = None,
        *,
        connection_hub_bundle_id: str | None = None,
        tenant: str | None = None,
        project: str | None = None,
        redis: Any = None,
        registry: dict[str, Any] | None = None,
        bundle_props: dict[str, Any] | None = None,
    ) -> None:
        self.authority_registry = AuthorityRegistryClient(
            entrypoint,
            connection_hub_bundle_id=connection_hub_bundle_id,
            tenant=tenant,
            project=project,
            redis=redis,
            registry=registry,
            bundle_props=bundle_props,
        )

    async def resolve_authority_provider(
        self,
        *,
        authority_id: str = "",
        provider_id: str = "",
        provider_type: str = "",
        host_bundle_id: str = "",
        host_route: str = "",
        host_operation: str = "",
    ) -> dict[str, Any]:
        return await self.authority_registry.resolve_provider(
            authority_id=authority_id,
            provider_id=provider_id,
            provider_type=provider_type,
            host_bundle_id=host_bundle_id,
            host_route=host_route,
            host_operation=host_operation,
        )

    async def resolve_authority_provider_entrypoint(
        self,
        *,
        authority_id: str = "",
        provider_id: str = "",
        provider_type: str = "",
        entrypoint: str = "login",
        request: Any = None,
        public_origin: str = "",
    ) -> dict[str, Any]:
        return await self.authority_registry.resolve_provider_entrypoint(
            authority_id=authority_id,
            provider_id=provider_id,
            provider_type=provider_type,
            entrypoint=entrypoint,
            request=request,
            public_origin=public_origin,
        )


__all__ = ["ConnectionsClient", "ConnectionsError", "Connection", "ConnectionHubClient"]
