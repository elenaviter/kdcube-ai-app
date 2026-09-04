# SPDX-License-Identifier: MIT

"""Projection joins Gateway provider facts to Card authority without widening."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest

from connection_hub.delegated_gateway import (
    ACCESS_DESCRIBE_TOOL,
    AcceptedDescriptor,
    GatewayTool,
    GatewayToolRoute,
    qualified_tool_name,
)
from connection_hub.delegated_gateway.models import canonical_digest

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources import (
    AuthoritySource,
    AvailabilityReason,
    DelegatedCardSnapshot,
    DelegatedResourceGrant,
    GatewayProjectionError,
    InvocationPolicyState,
    ResidentGatewayObservation,
    ResidentAgentCeiling,
    ResourceBinding,
    ResourceFamilyCeiling,
    compose_gateway_resource_facts,
    gateway_resident_projection,
    resolve_current_resident_runtime_projection,
    resolve_effective_resident_resources,
)

TENANT = "tenant-a"
PROJECT = "project-a"
APPLICATION = "workspace@1-0"
AGENT = "main"
USER = "user-a"
ACCESS_ID = "resident-card-1"
CLIENT_ID = "kdcube-agent:workspace@1-0:main"
RESOURCE = "urn:connection-hub:remote-mcp:connector-1"
REQUESTABLE = "urn:connection-hub:remote-mcp:connector-2"
NOW = 1_800_000_000


def _digest(label: str) -> str:
    return canonical_digest({"label": label})


def _grant() -> DelegatedResourceGrant:
    return DelegatedResourceGrant(
        resource_id=RESOURCE,
        resource_kind="remote_mcp",
        identity_scope="grantor",
        claims=("external_mcp:use",),
        operations=("delete", "search"),
        invocation_policies={
            "delete": InvocationPolicyState(mode="once", remaining=1, revision=2),
            "search": InvocationPolicyState(mode="always", revision=3),
        },
        operation_states={
            "delete": AvailabilityReason.OPERATION_DESCRIPTOR_CHANGED,
            "search": AvailabilityReason.AVAILABLE,
        },
        operation_accepted_digests={
            "delete": _digest("delete-v1"),
            "search": _digest("search-v1"),
        },
        operation_current_digests={
            "delete": _digest("delete-v2"),
            "search": _digest("search-v1"),
        },
        accepted_revision="1",
        current_revision="2",
        accepted_digest=_digest("resource-v1"),
        current_digest=_digest("resource-v2"),
    )


def _card(grant: DelegatedResourceGrant | None = None) -> DelegatedCardSnapshot:
    return DelegatedCardSnapshot(
        access_id=ACCESS_ID,
        client_id=CLIENT_ID,
        revision=7,
        tenant=TENANT,
        project=PROJECT,
        grantor_subject=USER,
        application=APPLICATION,
        agent_id=AGENT,
        identity_scope="grantor",
        source="resident_profile",
        resources=(grant or _grant(),),
        active=True,
        expires_at_epoch=NOW + 3600,
    )


def _binding() -> ResourceBinding:
    return ResourceBinding(
        mode="gateway",
        server_id="connection_hub_delegated",
        alias="connection_hub",
        transport="streamable-http",
        endpoint="https://hub.example.test/mcp/delegated_mcp_gateway",
    )


def _accepted(grant: DelegatedResourceGrant | None = None) -> AcceptedDescriptor:
    value = grant or _grant()
    return AcceptedDescriptor(
        revision=value.accepted_revision,
        digest=value.accepted_digest,
        operation_digests=value.operation_accepted_digests,
    )


def _route(
    operation: str = "search",
    *,
    grant: DelegatedResourceGrant | None = None,
) -> GatewayToolRoute:
    accepted = _accepted(grant)
    return GatewayToolRoute(
        resource_id=RESOURCE,
        resource_kind="remote_mcp",
        operation=operation,
        accepted_descriptor_identity=accepted.operation_identity(operation),
        provider_id="remote_mcp",
    )


def _tools(
    *,
    route: GatewayToolRoute | None = None,
    include_meta: bool = True,
) -> tuple[GatewayTool, ...]:
    selected = route or _route()
    rows = [
        GatewayTool(
            name=qualified_tool_name(selected),
            route=selected,
            title="Search fixture records",
            description="Search fixture records",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
            },
        )
    ]
    if include_meta:
        rows.append(
            GatewayTool(
                name=ACCESS_DESCRIBE_TOOL,
                route=None,
                title="Describe Connection Hub access",
                description="Describe this caller's delegated access.",
                input_schema={"type": "object", "additionalProperties": False},
            )
        )
    return tuple(rows)


def _access(grant: DelegatedResourceGrant | None = None) -> dict:
    value = grant or _grant()
    return {
        "schema": "connection_hub.delegated_gateway.access.v1",
        "caller": {
            "type": "resident",
            "profile_id": CLIENT_ID,
            "access_id": ACCESS_ID,
        },
        "card": {
            "revision": 7,
            "status": "active",
            "expires_at": NOW + 3600,
            "expired": False,
            "source": "resident_profile",
            "identity_scope": "grantor",
        },
        "resources": [
            {
                "resource_id": RESOURCE,
                "kind": "remote_mcp",
                "provider_id": "remote_mcp",
                "display_label": "Fixture",
                "endpoint_relation": "delegated_mcp_gateway",
                "identity_scope": "grantor",
                "state": "active",
                "grants": ["external_mcp:use"],
                "operations": ["delete", "search"],
                "accepted_descriptor": _accepted(value).to_public_dict(),
                "current_descriptor": {
                    "revision": value.current_revision,
                    "digest": value.current_digest,
                    "state": "current",
                },
                "invocation_policies": {
                    "delete": {
                        "mode": "once",
                        "state": "available",
                        "revision": 2,
                        "remaining": 1,
                    },
                    "search": {
                        "mode": "always",
                        "state": "available",
                        "revision": 3,
                    },
                },
                "unavailable_reason": "operation_descriptor_changed",
                "recovery": [
                    {
                        "code": "operation_descriptor_changed",
                        "href": "/connection-hub/card",
                    }
                ],
            }
        ],
        "requestable_discovery": "permitted",
        "requestable_resources": [
            {
                "resource_id": REQUESTABLE,
                "kind": "remote_mcp",
                "display_label": "Second fixture",
                "identity_scope": "grantor",
                "reason": "owner_delegable",
                "recovery": {"href": "/connection-hub/card"},
            }
        ],
    }


def _ceiling() -> ResidentAgentCeiling:
    return ResidentAgentCeiling(
        tenant=TENANT,
        project=PROJECT,
        application=APPLICATION,
        agent_id=AGENT,
        resource_families=(
            ResourceFamilyCeiling(
                family_id="user_external_mcp",
                resource_kinds=("remote_mcp",),
                authority_sources=(AuthoritySource.DELEGATED_CARD,),
                transports=("streamable-http",),
                resource_patterns=("urn:connection-hub:remote-mcp:*",),
                allowed_tool_patterns=("search", "delete"),
            ),
        ),
    )


def test_gateway_projection_preserves_qualified_route_and_operation_drift():
    projection = gateway_resident_projection(
        card=_card(),
        tools=_tools(),
        access=_access(),
        binding=_binding(),
    )

    assert [tool.name for tool in projection.meta_tools] == [ACCESS_DESCRIBE_TOOL]
    assert [item.resource_id for item in projection.requestable_resources] == [
        REQUESTABLE
    ]
    candidate = projection.candidates[0]
    assert candidate.provider_endpoint == ""
    assert candidate.family_id == ""
    search = next(tool for tool in candidate.tools if tool.operation == "search")
    delete = next(tool for tool in candidate.tools if tool.operation == "delete")
    assert search.name == qualified_tool_name(_route())
    assert search.output_schema == {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
    }
    assert delete.name == qualified_tool_name(_route("delete"))
    assert candidate.tool_status["search"].ready is True
    assert candidate.tool_status["delete"].reason is (
        AvailabilityReason.OPERATION_DESCRIPTOR_CHANGED
    )

    effective = resolve_effective_resident_resources(
        ceiling=_ceiling(),
        grantor_subject=USER,
        candidates=projection.candidates,
        card=_card(),
        now_epoch=NOW,
    )
    by_operation = {tool.operation: tool for tool in effective.resources[0].tools}
    assert by_operation["search"].available is True
    assert by_operation["search"].name == search.name
    assert by_operation["delete"].available is False
    assert by_operation["delete"].reason is (
        AvailabilityReason.OPERATION_DESCRIPTOR_CHANGED
    )


def test_one_gateway_observation_drives_inventory_runtime_and_requestable_view():
    facts = compose_gateway_resource_facts(
        observation=ResidentGatewayObservation(
            card=_card(),
            tools=_tools(),
            access=_access(),
            binding=_binding(),
        ),
        observed_at_epoch=NOW,
    )

    class _Loader:
        calls = 0

        async def load_current(self, *, ceiling, grantor_subject):
            self.calls += 1
            assert ceiling == _ceiling()
            assert grantor_subject == USER
            return facts

    loader = _Loader()
    resolved = asyncio.run(
        resolve_current_resident_runtime_projection(
            loader=loader,
            ceiling=_ceiling(),
            grantor_subject=USER,
        )
    )

    assert loader.calls == 1
    assert [row.resource_id for row in resolved.inventory.resources] == [RESOURCE]
    assert len(resolved.gateway_plan.connections) == 1
    assert set(resolved.gateway_plan.connections[0].tool_names) == {
        ACCESS_DESCRIBE_TOOL,
        qualified_tool_name(_route()),
    }
    assert [
        item.resource_id
        for item in resolved.requestable_resources_by_access_id[ACCESS_ID]
    ] == [REQUESTABLE]
    assert resolved.requestable_discovery_by_access_id == {
        ACCESS_ID: "permitted"
    }


def test_conversation_narrowing_removes_gateway_operation_from_every_runtime():
    facts = compose_gateway_resource_facts(
        observation=ResidentGatewayObservation(
            card=_card(),
            tools=_tools(),
            access=_access(),
            binding=_binding(),
        ),
        observed_at_epoch=NOW,
    )

    class _Loader:
        async def load_current(self, **_kwargs):
            return facts

    resolved = asyncio.run(
        resolve_current_resident_runtime_projection(
            loader=_Loader(),
            ceiling=_ceiling(),
            grantor_subject=USER,
            disabled_selection={"resources": {RESOURCE: ["search"]}},
        )
    )

    tools = resolved.gateway_plan.connections[0].tool_names
    assert tools == (ACCESS_DESCRIBE_TOOL,)
    assert qualified_tool_name(_route()) not in tools


def test_current_card_operation_missing_from_gateway_is_provider_unavailable():
    grant = replace(
        _grant(),
        operations=("search",),
        invocation_policies={"search": InvocationPolicyState("always", revision=3)},
        operation_states={"search": AvailabilityReason.AVAILABLE},
        operation_accepted_digests={"search": _digest("search-v1")},
        operation_current_digests={"search": _digest("search-v1")},
    )
    access = _access(grant)
    resource = access["resources"][0]
    resource["operations"] = ["search"]
    resource["invocation_policies"] = {
        "search": {"mode": "always", "state": "available", "revision": 3}
    }
    resource["unavailable_reason"] = ""

    projection = gateway_resident_projection(
        card=_card(grant),
        tools=_tools(include_meta=True)[1:],
        access=access,
        binding=_binding(),
    )

    assert projection.candidates[0].tool_status["search"].reason is (
        AvailabilityReason.RESOURCE_PROVIDER_UNAVAILABLE
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda access, tools: access["card"].update(revision=8),
            "gateway_access_card_mismatch",
        ),
        (
            lambda access, tools: access["resources"][0]["operations"].append(
                "ungranted"
            ),
            "gateway_access_resource_mismatch",
        ),
        (
            lambda access, tools: access["resources"][0][
                "current_descriptor"
            ].update(revision="stale"),
            "gateway_access_current_descriptor_mismatch",
        ),
        (
            lambda access, tools: access["resources"][0].update(
                provider_id="another_provider"
            ),
            "gateway_tool_provider_mismatch",
        ),
        (
            lambda access, tools: access["requestable_resources"][0][
                "recovery"
            ].update(access_token="must-not-cross"),
            "gateway_access_not_public",
        ),
    ],
)
def test_gateway_and_card_mismatches_fail_closed(mutate, reason):
    access = deepcopy(_access())
    tools = list(_tools())
    mutate(access, tools)

    with pytest.raises(GatewayProjectionError, match=reason):
        gateway_resident_projection(
            card=_card(),
            tools=tools,
            access=access,
            binding=_binding(),
        )


def test_tampered_qualified_name_and_changed_operation_route_fail_closed():
    route = _route()
    tampered = GatewayTool(
        name="ch_remote_mcp_0000000000000000__search_0000000000000000",
        route=route,
        title="Search",
        description="Search",
        input_schema={},
    )
    meta = _tools()[1]
    with pytest.raises(GatewayProjectionError, match="gateway_tool_authority_mismatch"):
        gateway_resident_projection(
            card=_card(),
            tools=(tampered, meta),
            access=_access(),
            binding=_binding(),
        )

    with pytest.raises(GatewayProjectionError, match="gateway_tool_not_current"):
        gateway_resident_projection(
            card=_card(),
            tools=_tools(route=_route("delete")),
            access=_access(),
            binding=_binding(),
        )


def test_gateway_projection_requires_caller_self_tool_and_safe_endpoint():
    with pytest.raises(
        GatewayProjectionError, match="gateway_access_describe_tool_missing"
    ):
        gateway_resident_projection(
            card=_card(),
            tools=_tools(include_meta=False),
            access=_access(),
            binding=_binding(),
        )

    unsafe = replace(_binding(), endpoint="https://user:secret@hub.example.test/mcp")
    with pytest.raises(GatewayProjectionError, match="gateway_binding_endpoint_invalid"):
        gateway_resident_projection(
            card=_card(), tools=_tools(), access=_access(), binding=unsafe
        )


def test_gateway_projection_contains_no_bearer_or_provider_credential():
    projection = gateway_resident_projection(
        card=_card(), tools=_tools(), access=_access(), binding=_binding()
    )
    rendered = repr(projection).lower()
    assert "access_token" not in rendered
    assert "refresh_token" not in rendered
    assert "authorization" not in rendered
    assert "provider_credential" not in rendered
    compose_gateway_resource_facts,
