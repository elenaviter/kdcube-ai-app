# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Live delegated Gateway ports for one resident KDCube agent Card."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub._resident_gateway_contract import (
    ConnectionHubResidentCardResolver,
    ResidentCardCredential,
    ResidentGatewayEndpoints,
    ResidentGatewayHostError,
    clean,
    resident_gateway_endpoints,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub._resident_gateway_transport import (
    gateway_tools_from_observation,
    read_resident_gateway_access,
    read_resident_gateway_tools,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connection_edges import (
    connection_hub_bundle_id_from_entrypoint,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.card_adapter import (
    delegated_card_snapshot_from_view,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.gateway_observation import (
    ResidentGatewayObservation,
    compose_gateway_resource_facts,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AvailabilityReason,
    ResidentAgentCeiling,
    ResourceBinding,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.service import (
    CurrentResidentResourceFacts,
)


AccessReader = Callable[[str, str], Awaitable[Mapping[str, Any]]]
ToolReader = Callable[[str, str], Awaitable[Sequence[Any]]]


def _binding(endpoint: str, access_id: str) -> ResourceBinding:
    suffix = "".join(
        character if character.isalnum() else "_" for character in access_id
    )[-24:]
    server_id = f"connection_hub_gateway_{suffix}"
    return ResourceBinding(
        mode="gateway",
        server_id=server_id,
        alias=server_id,
        transport="streamable-http",
        endpoint=endpoint,
    )


class ConnectionHubResidentGatewayFactsLoader:
    """Observe one exact resident Card and the existing Gateway per turn."""

    def __init__(
        self,
        *,
        card_resolver: ConnectionHubResidentCardResolver,
        endpoints: ResidentGatewayEndpoints,
        access_reader: AccessReader = read_resident_gateway_access,
        tool_reader: ToolReader = read_resident_gateway_tools,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._card = card_resolver
        self._endpoints = endpoints
        self._access_reader = access_reader
        self._tool_reader = tool_reader
        self._clock = clock

    async def load_current(
        self,
        *,
        ceiling: ResidentAgentCeiling,
        grantor_subject: str,
    ) -> CurrentResidentResourceFacts:
        observed_at = int(self._clock())
        try:
            credential = await self._card.resolve(
                grantor_subject=grantor_subject,
                application=ceiling.application,
                agent_id=ceiling.agent_id,
            )
            if credential is None:
                return compose_gateway_resource_facts(
                    observed_at_epoch=observed_at,
                    card_unavailable_reasons={
                        "grantor": AvailabilityReason.CARD_MISSING
                    },
                )
            card = delegated_card_snapshot_from_view(
                credential.card,
                tenant=ceiling.tenant,
                project=ceiling.project,
            )
            if (
                card.application != ceiling.application
                or card.agent_id != ceiling.agent_id
            ):
                raise ResidentGatewayHostError(
                    "resident_gateway_card_profile_mismatch"
                )
            access = await self._access_reader(
                self._endpoints.access,
                credential.access_token,
            )
            raw_tools = await self._tool_reader(
                self._endpoints.mcp,
                credential.access_token,
            )
            observation = ResidentGatewayObservation(
                card=card,
                tools=gateway_tools_from_observation(
                    raw_tools,
                    card=card,
                    access=access,
                ),
                access=access,
                binding=_binding(self._endpoints.mcp, card.access_id),
            )
            return compose_gateway_resource_facts(
                observation=observation,
                observed_at_epoch=observed_at,
            )
        except Exception:
            return compose_gateway_resource_facts(
                observed_at_epoch=observed_at,
                card_unavailable_reasons={
                    "grantor": AvailabilityReason.CARD_AUTHORITY_UNAVAILABLE
                },
            )


class ConnectionHubResidentGatewayBearerProvider:
    """Resolve the exact Card again immediately before an MCP request."""

    def __init__(
        self,
        *,
        card_resolver: ConnectionHubResidentCardResolver,
        application: str,
        agent_id: str,
    ) -> None:
        self._card = card_resolver
        self._application = clean(application)
        self._agent_id = clean(agent_id)

    async def __call__(
        self,
        connection: Mapping[str, Any],
        user_subject: str,
    ) -> str | None:
        access_id = clean(connection.get("access_id"))
        try:
            expected_revision = int(connection.get("card_revision") or 0)
        except (TypeError, ValueError):
            return None
        if not access_id or expected_revision < 1:
            return None
        try:
            credential = await self._card.resolve(
                grantor_subject=clean(user_subject),
                application=self._application,
                agent_id=self._agent_id,
                expected_access_id=access_id,
            )
        except Exception:
            return None
        if credential is None or credential.card.card_revision != expected_revision:
            return None
        selected_resources = {
            clean(value) for value in connection.get("resources", ()) if clean(value)
        }
        card_resources = {item.resource for item in credential.card.resources}
        if not selected_resources or not selected_resources.issubset(card_resources):
            return None
        return credential.access_token


@dataclass(frozen=True)
class ResidentGatewayRuntimePorts:
    facts_loader: ConnectionHubResidentGatewayFactsLoader
    bearer_provider: ConnectionHubResidentGatewayBearerProvider


def build_resident_gateway_runtime_ports(
    host: Any,
    *,
    tenant: str,
    project: str,
    application: str,
    agent_id: str,
    runtime_origin: str = "",
) -> ResidentGatewayRuntimePorts:
    bundle_id = connection_hub_bundle_id_from_entrypoint(host)
    card_resolver = ConnectionHubResidentCardResolver(
        connection_hub_bundle_id=bundle_id,
    )
    endpoints = resident_gateway_endpoints(
        tenant=tenant,
        project=project,
        connection_hub_bundle_id=bundle_id,
        runtime_origin=runtime_origin,
    )
    return ResidentGatewayRuntimePorts(
        facts_loader=ConnectionHubResidentGatewayFactsLoader(
            card_resolver=card_resolver,
            endpoints=endpoints,
        ),
        bearer_provider=ConnectionHubResidentGatewayBearerProvider(
            card_resolver=card_resolver,
            application=application,
            agent_id=agent_id,
        ),
    )


__all__ = [
    "ConnectionHubResidentCardResolver",
    "ConnectionHubResidentGatewayBearerProvider",
    "ConnectionHubResidentGatewayFactsLoader",
    "ResidentCardCredential",
    "ResidentGatewayEndpoints",
    "ResidentGatewayHostError",
    "ResidentGatewayRuntimePorts",
    "build_resident_gateway_runtime_ports",
    "read_resident_gateway_access",
    "read_resident_gateway_tools",
    "resident_gateway_endpoints",
]
