# SPDX-License-Identifier: MIT

"""Resident KDCube agents observe and use one exact Connection Hub Card."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from connection_hub.delegated_credentials.cards.identity import ResidentCallerProfile
from connection_hub.delegated_credentials.cards.read_model import (
    CALLER_KIND_RESIDENT,
    CardOperationView,
    CardResourceView,
    DelegatedCardView,
)
from connection_hub.delegated_gateway import (
    ACCESS_DESCRIBE_TOOL,
    AcceptedDescriptor,
    GatewayToolRoute,
    qualified_tool_name,
)
from connection_hub.delegated_gateway.models import canonical_digest

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.resident_gateway import (
    ConnectionHubResidentCardResolver,
    ConnectionHubResidentGatewayBearerProvider,
    ConnectionHubResidentGatewayFactsLoader,
    ResidentGatewayEndpoints,
    resident_gateway_endpoints,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub._resident_gateway_transport import (
    _gateway_access_payload,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources import (
    AuthoritySource,
    ResidentAgentCeiling,
    ResourceFamilyCeiling,
    resolve_current_resident_runtime_projection,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceResponse,
)


TENANT = "tenant-a"
PROJECT = "project-a"
APPLICATION = "workspace@1-0"
AGENT = "main"
USER = "user-a"
CLIENT = "kdcube-agent:workspace@1-0:main"
RESOURCE = "urn:connection-hub:remote-mcp:fixture"
MANAGED_RESOURCE = (
    "*/api/integrations/bundles/*/*/knowledge@1-0/public/mcp/"
    "knowledge_managed*"
)
NOW = 1_800_000_000


def _digest(label: str) -> str:
    return canonical_digest({"label": label})


def _profile() -> ResidentCallerProfile:
    return ResidentCallerProfile(
        grantor_subject=USER,
        application=APPLICATION,
        agent_id=AGENT,
    )


def _card() -> DelegatedCardView:
    profile = _profile()
    operation_digest = _digest("search-v1")
    resource_digest = _digest("fixture-v1")
    return DelegatedCardView(
        access_id=profile.access_id,
        client_id=profile.client_id,
        caller_kind=CALLER_KIND_RESIDENT,
        profile=profile,
        grantor_subject=USER,
        delegate_subject="integration:resident:user-a",
        source="agent",
        label="main",
        card_revision=4,
        catalog_version="catalog-1",
        state="active",
        created_at=NOW - 100,
        expires_at=NOW + 3600,
        identity_scope="grantor",
        resources=(
            CardResourceView(
                resource=RESOURCE,
                kind="remote_mcp",
                provider="remote_mcp",
                label="Fixture",
                state="current",
                identity_scope="grantor",
                grants=("external_mcp:use",),
                operations=(
                    CardOperationView(
                        name="search",
                        state="current",
                        accepted_digest=operation_digest,
                        current_digest=operation_digest,
                        policy={
                            "authority": {
                                "access_id": profile.access_id,
                                "resource": RESOURCE,
                                "surface": "outer",
                                "operation": "search",
                            },
                            "mode": "always",
                            "state": "available",
                            "revision": 2,
                            "remaining": None,
                        },
                    ),
                ),
                accepted_revision="1",
                current_revision="1",
                accepted_digest=resource_digest,
                current_digest=resource_digest,
            ),
        ),
    )


def _access() -> dict:
    card = _card()
    resource = card.resources[0]
    accepted = AcceptedDescriptor(
        revision=resource.accepted_revision,
        digest=resource.accepted_digest,
        operation_digests={
            operation.name: operation.accepted_digest
            for operation in resource.operations
        },
    )
    return {
        "schema": "connection_hub.delegated_gateway.access.v1",
        "caller": {
            "type": "resident",
            "profile_id": CLIENT,
            "access_id": card.access_id,
        },
        "card": {
            "revision": card.card_revision,
            "status": "active",
            "expires_at": card.expires_at,
            "expired": False,
            "source": "agent",
            "identity_scope": "grantor",
        },
        "resources": [
            {
                "resource_id": RESOURCE,
                "kind": "remote_mcp",
                "provider_id": "remote_mcp",
                "display_label": "Fixture",
                "endpoint_relation": "connection_hub_remote_mcp_proxy",
                "identity_scope": "grantor",
                "state": "active",
                "grants": ["external_mcp:use"],
                "operations": ["search"],
                "accepted_descriptor": accepted.to_public_dict(),
                "current_descriptor": {
                    "revision": "1",
                    "digest": resource.current_digest,
                    "state": "current",
                },
                "invocation_policies": {
                    "search": {
                        "mode": "always",
                        "state": "available",
                        "revision": 2,
                        "remaining": None,
                    }
                },
                "unavailable_reason": "",
                "recovery": [],
            }
        ],
        "requestable_resources": [],
        "requestable_discovery": "permitted",
    }


def _raw_tools() -> tuple[dict, ...]:
    card = _card()
    resource = card.resources[0]
    accepted = AcceptedDescriptor(
        revision=resource.accepted_revision,
        digest=resource.accepted_digest,
        operation_digests={"search": resource.operations[0].accepted_digest},
    )
    route = GatewayToolRoute(
        resource_id=RESOURCE,
        resource_kind="remote_mcp",
        operation="search",
        accepted_descriptor_identity=accepted.operation_identity("search"),
        provider_id="remote_mcp",
    )
    return (
        {
            "name": ACCESS_DESCRIBE_TOOL,
            "title": "Describe access",
            "description": "Describe the current card.",
            "inputSchema": {"type": "object", "additionalProperties": False},
        },
        {
            "name": qualified_tool_name(route),
            "title": "Search fixture",
            "description": "Search fixture records.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "_meta": {
                "connection_hub": {
                    "resource_id": RESOURCE,
                    "resource_kind": "remote_mcp",
                    "operation": "search",
                }
            },
        },
    )


def _ceiling() -> ResidentAgentCeiling:
    return ResidentAgentCeiling(
        tenant=TENANT,
        project=PROJECT,
        application=APPLICATION,
        agent_id=AGENT,
        resource_families=(
            ResourceFamilyCeiling(
                family_id="external_mcp",
                resource_kinds=("remote_mcp",),
                authority_sources=(AuthoritySource.DELEGATED_CARD,),
                transports=("streamable-http",),
                resource_patterns=("urn:connection-hub:remote-mcp:*",),
            ),
        ),
    )


def _card_resolver(card: DelegatedCardView | None = None):
    card = card or _card()
    token = "resident-secret-token"

    async def named_service_caller(**kwargs):
        payload = kwargs["request"]["payload"]
        assert payload == {
            "client_id": CLIENT,
            "access_id": card.access_id,
        }
        response = NamedServiceResponse.ok_response(
            provider={"id": "connections"},
            namespace="connections",
            object={
                "access_token": token,
                "expires_at": str(card.expires_at),
                "access_id": card.access_id,
                "client_id": CLIENT,
                "identity_scope": "grantor",
                "card_revision": card.card_revision,
                "card": card.to_dict(),
            },
            attrs={"has_token": True},
        )
        return SimpleNamespace(value=response)

    return (
        ConnectionHubResidentCardResolver(
            connection_hub_bundle_id="connection-hub@1-0",
            named_service_caller=named_service_caller,
        ),
        token,
    )


def _managed_resource(card: DelegatedCardView) -> CardResourceView:
    operation_digest = _digest("knowledge-read-v1")
    return CardResourceView(
        resource=MANAGED_RESOURCE,
        kind="catalog",
        provider="",
        label="Knowledge",
        state="current",
        identity_scope="grantor",
        grants=("knowledge:read",),
        operations=(
            CardOperationView(
                name="read_refs",
                state="current",
                accepted_digest=operation_digest,
                current_digest=operation_digest,
                policy={
                    "authority": {
                        "access_id": card.access_id,
                        "resource": MANAGED_RESOURCE,
                        "surface": "outer",
                        "operation": "read_refs",
                    },
                    "mode": "always",
                    "state": "available",
                    "revision": 3,
                    "remaining": None,
                },
            ),
        ),
        accepted_revision="knowledge-1",
        current_revision="knowledge-1",
        accepted_digest=_digest("knowledge-v1"),
        current_digest=_digest("knowledge-v1"),
    )


def test_processor_local_gateway_endpoints_are_deployment_scoped():
    endpoints = resident_gateway_endpoints(
        tenant=TENANT,
        project=PROJECT,
        connection_hub_bundle_id="connection-hub@1-0",
        runtime_origin="http://127.0.0.1:8020",
    )
    assert endpoints.mcp == (
        "http://127.0.0.1:8020/api/integrations/bundles/tenant-a/project-a/"
        "connection-hub%401-0/public/mcp/delegated_mcp_gateway"
    )
    assert endpoints.access.endswith(
        "/public/delegated_mcp_gateway_access?include_requestable=true"
    )


def test_gateway_access_payload_accepts_hosted_operation_envelope():
    access = _access()
    assert _gateway_access_payload(
        {"delegated_mcp_gateway_access": {"ok": True, "access": access}}
    ) == access
    assert _gateway_access_payload({"ok": True, "access": access}) == access


@pytest.mark.asyncio
async def test_loader_projects_exact_card_without_retaining_credential():
    card_resolver, token = _card_resolver()
    observed_tokens: list[str] = []

    async def access_reader(endpoint, access_token):
        assert endpoint == "http://127.0.0.1:8020/access"
        observed_tokens.append(access_token)
        return _access()

    async def tool_reader(endpoint, access_token):
        assert endpoint == "http://127.0.0.1:8020/mcp"
        observed_tokens.append(access_token)
        return _raw_tools()

    loader = ConnectionHubResidentGatewayFactsLoader(
        card_resolver=card_resolver,
        endpoints=ResidentGatewayEndpoints(
            mcp="http://127.0.0.1:8020/mcp",
            access="http://127.0.0.1:8020/access",
        ),
        access_reader=access_reader,
        tool_reader=tool_reader,
        clock=lambda: NOW,
    )
    projection = await resolve_current_resident_runtime_projection(
        loader=loader,
        ceiling=_ceiling(),
        grantor_subject=USER,
    )

    assert observed_tokens == [token, token]
    assert len(projection.inventory.resources) == 1
    assert projection.inventory.resources[0].resource_id == RESOURCE
    connection = projection.gateway_plan.connections[0]
    assert connection.access_id == _card().access_id
    assert connection.card_revision == 4
    assert "resident-secret-token" not in repr(projection)
    assert "resident-secret-token" not in str(connection.to_connection_dict())


@pytest.mark.asyncio
async def test_one_resident_card_projects_managed_and_external_resources_to_one_gateway():
    external_card = _card()
    card = replace(
        external_card,
        resources=(external_card.resources[0], _managed_resource(external_card)),
    )
    card_resolver, token = _card_resolver(card)
    access = _access()
    managed = card.resources[1]
    managed_operation = managed.operations[0]
    managed_descriptor = AcceptedDescriptor(
        revision=managed.accepted_revision,
        digest=managed.accepted_digest,
        operation_digests={
            managed_operation.name: managed_operation.accepted_digest,
        },
    )
    access["resources"].append(
        {
            "resource_id": MANAGED_RESOURCE,
            "kind": "catalog",
            "provider_id": "managed_kdcube_mcp",
            "display_label": "Knowledge",
            "endpoint_relation": (
                "same_kdcube:knowledge@1-0:public:mcp:knowledge_managed"
            ),
            "identity_scope": "grantor",
            "state": "active",
            "grants": ["knowledge:read"],
            "operations": ["read_refs"],
            "accepted_descriptor": managed_descriptor.to_public_dict(),
            "current_descriptor": {
                "revision": managed.current_revision,
                "digest": managed.current_digest,
                "state": "current",
            },
            "invocation_policies": {
                "read_refs": {
                    "mode": "always",
                    "state": "available",
                    "revision": 3,
                    "remaining": None,
                }
            },
            "unavailable_reason": "",
            "recovery": [],
        }
    )
    managed_route = GatewayToolRoute(
        resource_id=MANAGED_RESOURCE,
        resource_kind="catalog",
        operation="read_refs",
        accepted_descriptor_identity=managed_descriptor.operation_identity(
            "read_refs"
        ),
        provider_id="managed_kdcube_mcp",
    )
    raw_tools = list(_raw_tools())
    raw_tools.append(
        {
            "name": qualified_tool_name(managed_route),
            "title": "Read knowledge references",
            "description": "Read exact knowledge references.",
            "inputSchema": {
                "type": "object",
                "properties": {"refs": {"type": "array"}},
                "required": ["refs"],
            },
            "_meta": {
                "connection_hub": {
                    "resource_id": MANAGED_RESOURCE,
                    "resource_kind": "catalog",
                    "operation": "read_refs",
                }
            },
        }
    )

    async def access_reader(_endpoint, access_token):
        assert access_token == token
        return access

    async def tool_reader(_endpoint, access_token):
        assert access_token == token
        return tuple(raw_tools)

    loader = ConnectionHubResidentGatewayFactsLoader(
        card_resolver=card_resolver,
        endpoints=ResidentGatewayEndpoints(
            mcp="http://127.0.0.1:8020/mcp",
            access="http://127.0.0.1:8020/access",
        ),
        access_reader=access_reader,
        tool_reader=tool_reader,
        clock=lambda: NOW,
    )
    ceiling = replace(_ceiling(), declared_resource_ids=(MANAGED_RESOURCE,))

    projection = await resolve_current_resident_runtime_projection(
        loader=loader,
        ceiling=ceiling,
        grantor_subject=USER,
    )

    assert {resource.resource_id for resource in projection.inventory.resources} == {
        RESOURCE,
        MANAGED_RESOURCE,
    }
    assert {
        resource.access_id for resource in projection.inventory.resources
    } == {card.access_id}
    assert len(projection.gateway_plan.connections) == 1
    connection = projection.gateway_plan.connections[0]
    assert connection.access_id == card.access_id
    assert set(connection.resource_ids) == {RESOURCE, MANAGED_RESOURCE}


@pytest.mark.asyncio
async def test_bearer_provider_rechecks_exact_card_and_revision():
    card_resolver, token = _card_resolver()
    provider = ConnectionHubResidentGatewayBearerProvider(
        card_resolver=card_resolver,
        application=APPLICATION,
        agent_id=AGENT,
    )
    connection = {
        "access_id": _card().access_id,
        "card_revision": 4,
        "identity_scope": "grantor",
        "resources": [RESOURCE],
    }
    assert await provider(connection, USER) == token
    assert await provider({**connection, "card_revision": 3}, USER) is None
    assert await provider({**connection, "resources": ["urn:other"]}, USER) is None
