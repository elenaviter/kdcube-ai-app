# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Bounded HTTP/MCP observation of the hosted delegated Gateway."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx2

from connection_hub.delegated_gateway import (
    ACCESS_DESCRIBE_TOOL,
    AcceptedDescriptor,
    GatewayTool,
    GatewayToolRoute,
)

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub._resident_gateway_contract import (
    ResidentGatewayHostError,
    clean,
    mapping,
    safe_bearer,
)
from kdcube_ai_app.apps.chat.sdk.runtime.mcp.client import open_mcp_client
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    DelegatedCardSnapshot,
)


_MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024
_ACCESS_OPERATION_ALIAS = "delegated_mcp_gateway_access"


def _model_mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="json", by_alias=True, exclude_none=True)
        return mapping(result, reason)
    raise ResidentGatewayHostError(reason)


def _gateway_access_payload(value: Any) -> Mapping[str, Any]:
    envelope = mapping(value, "resident_gateway_access_invalid")
    wrapped = envelope.get(_ACCESS_OPERATION_ALIAS)
    if wrapped is not None:
        envelope = mapping(wrapped, "resident_gateway_access_invalid")
    if envelope.get("ok") is not True:
        raise ResidentGatewayHostError("resident_gateway_access_unavailable")
    return mapping(envelope.get("access"), "resident_gateway_access_invalid")


async def read_resident_gateway_access(
    endpoint: str,
    access_token: str,
) -> Mapping[str, Any]:
    """Read caller-self Gateway facts without retaining transport state."""

    try:
        async with httpx2.AsyncClient(
            timeout=httpx2.Timeout(10.0, connect=5.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.get(
                endpoint,
                headers={"Authorization": f"Bearer {safe_bearer(access_token)}"},
            )
            body = response.content
        if response.status_code != 200 or len(body) > _MAX_HTTP_BODY_BYTES:
            raise ResidentGatewayHostError("resident_gateway_access_unavailable")
        return _gateway_access_payload(response.json())
    except ResidentGatewayHostError:
        raise
    except Exception:
        raise ResidentGatewayHostError("resident_gateway_access_unavailable") from None


async def read_resident_gateway_tools(
    endpoint: str,
    access_token: str,
) -> Sequence[Any]:
    """List raw MCP tool models through the maintained SDK client."""

    try:
        async with open_mcp_client(
            transport="streamable-http",
            endpoint=endpoint,
            headers={"Authorization": f"Bearer {safe_bearer(access_token)}"},
            read_timeout_seconds=10.0,
            follow_redirects=False,
            trust_env=False,
            http_timeout_seconds=10.0,
            http_read_timeout_seconds=10.0,
            http_connect_timeout_seconds=5.0,
            terminate_on_close=True,
        ) as client:
            response = await client.list_tools()
            return tuple(getattr(response, "tools", ()) or ())
    except ResidentGatewayHostError:
        raise
    except Exception:
        raise ResidentGatewayHostError("resident_gateway_tools_unavailable") from None


def gateway_tools_from_observation(
    raw_tools: Sequence[Any],
    *,
    card: DelegatedCardSnapshot,
    access: Mapping[str, Any],
) -> tuple[GatewayTool, ...]:
    """Bind public tool schemas to the exact Card/resource observation."""

    caller = mapping(access.get("caller"), "resident_gateway_access_invalid")
    card_state = mapping(access.get("card"), "resident_gateway_access_invalid")
    try:
        observed_revision = int(card_state.get("revision") or 0)
    except (TypeError, ValueError):
        raise ResidentGatewayHostError("resident_gateway_access_invalid") from None
    if (
        clean(caller.get("access_id")) != card.access_id
        or observed_revision != card.revision
    ):
        raise ResidentGatewayHostError("resident_gateway_observation_changed")

    grants = {item.resource_id: item for item in card.resources}
    raw_resources = access.get("resources")
    if not isinstance(raw_resources, list):
        raise ResidentGatewayHostError("resident_gateway_access_invalid")
    providers: dict[str, str] = {}
    for raw in raw_resources:
        row = mapping(raw, "resident_gateway_access_invalid")
        resource_id = clean(row.get("resource_id"))
        provider_id = clean(row.get("provider_id"))
        if not resource_id or not provider_id or resource_id in providers:
            raise ResidentGatewayHostError("resident_gateway_access_invalid")
        providers[resource_id] = provider_id

    tools: list[GatewayTool] = []
    for raw_tool in raw_tools:
        payload = _model_mapping(raw_tool, "resident_gateway_tool_invalid")
        name = clean(payload.get("name"))
        if not name:
            raise ResidentGatewayHostError("resident_gateway_tool_invalid")
        raw_input = payload.get("inputSchema")
        if raw_input is None:
            raw_input = payload.get("input_schema")
        input_schema = mapping(
            raw_input or {},
            "resident_gateway_tool_schema_invalid",
        )
        raw_output = payload.get("outputSchema")
        if raw_output is None:
            raw_output = payload.get("output_schema")
        output_schema = (
            None
            if raw_output is None
            else mapping(raw_output, "resident_gateway_tool_schema_invalid")
        )
        raw_meta = payload.get("_meta")
        if raw_meta is None:
            raw_meta = payload.get("meta")
        meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        raw_route = meta.get("connection_hub")
        route_data = raw_route if isinstance(raw_route, Mapping) else {}
        route: GatewayToolRoute | None = None
        if route_data:
            resource_id = clean(route_data.get("resource_id"))
            resource_kind = clean(route_data.get("resource_kind"))
            operation = clean(route_data.get("operation"))
            grant = grants.get(resource_id)
            provider_id = providers.get(resource_id, "")
            if (
                grant is None
                or resource_kind != grant.resource_kind
                or operation not in grant.operations
                or not provider_id
            ):
                raise ResidentGatewayHostError(
                    "resident_gateway_tool_route_invalid"
                )
            accepted = AcceptedDescriptor(
                revision=grant.accepted_revision,
                digest=grant.accepted_digest,
                operation_digests=grant.operation_accepted_digests,
            )
            route = GatewayToolRoute(
                resource_id=resource_id,
                resource_kind=resource_kind,
                operation=operation,
                accepted_descriptor_identity=accepted.operation_identity(operation),
                provider_id=provider_id,
            )
        elif name != ACCESS_DESCRIBE_TOOL:
            raise ResidentGatewayHostError("resident_gateway_tool_route_missing")
        tools.append(
            GatewayTool(
                name=name,
                route=route,
                title=clean(payload.get("title")) or name,
                description=clean(payload.get("description")),
                input_schema=dict(input_schema),
                output_schema=(
                    dict(output_schema) if output_schema is not None else None
                ),
            )
        )
    return tuple(tools)


__all__ = [
    "gateway_tools_from_observation",
    "read_resident_gateway_access",
    "read_resident_gateway_tools",
]
