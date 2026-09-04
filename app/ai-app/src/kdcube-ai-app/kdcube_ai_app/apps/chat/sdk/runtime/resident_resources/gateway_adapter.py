# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Adapt one current delegated Gateway observation into Projection facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from connection_hub.delegated_gateway import (
    ACCESS_DESCRIBE_TOOL,
    AcceptedDescriptor,
    GatewayTool,
    GatewayToolRoute,
    qualified_tool_name,
)
from connection_hub.delegated_gateway.models import public_mapping

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AuthoritySource,
    AvailabilityReason,
    DelegatedCardSnapshot,
    DelegatedResourceGrant,
    ResidentResourceCandidate,
    ResidentToolDescriptor,
    ResidentToolStatus,
    ResourceBinding,
)

_ACCESS_SCHEMA = "connection_hub.delegated_gateway.access.v1"
_REQUESTABLE_STATES = frozenset(
    {"not_requested", "not_permitted", "permitted", "unavailable"}
)
_RESOURCE_REASON_MAP = {
    "": AvailabilityReason.AVAILABLE,
    "available": AvailabilityReason.AVAILABLE,
    # The Card snapshot carries exact per-operation drift. A resource-level
    # drift reason must not hide an unchanged sibling operation.
    "descriptor_changed": AvailabilityReason.AVAILABLE,
    "operation_descriptor_changed": AvailabilityReason.AVAILABLE,
    "connector_disabled": AvailabilityReason.CONNECTOR_DISABLED,
    "connector_not_active": AvailabilityReason.CONNECTOR_DISABLED,
    "resource_disabled": AvailabilityReason.CONNECTOR_DISABLED,
    "credential_missing": AvailabilityReason.CREDENTIAL_MISSING,
    "descriptor_missing": AvailabilityReason.RESOURCE_NOT_CURRENT,
    "descriptor_unknown": AvailabilityReason.RESOURCE_NOT_CURRENT,
    "connected_account_missing": AvailabilityReason.CONNECTED_ACCOUNT_MISSING,
    "connected_account_claim_missing": (
        AvailabilityReason.CONNECTED_ACCOUNT_CLAIM_MISSING
    ),
    "connected_account_unavailable": (
        AvailabilityReason.CONNECTED_ACCOUNT_UNAVAILABLE
    ),
    "resource_provider_unavailable": (
        AvailabilityReason.RESOURCE_PROVIDER_UNAVAILABLE
    ),
}


class GatewayProjectionError(ValueError):
    """Gateway and Card observations cannot be joined without widening."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class GatewayRequestableResource:
    """One caller-visible, ungranted resource returned by the Gateway."""

    resource_id: str
    resource_kind: str
    display_name: str
    identity_scope: str
    reason: str
    recovery: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_kind": self.resource_kind,
            "display_name": self.display_name,
            "identity_scope": self.identity_scope,
            "reason": self.reason,
            "recovery": dict(self.recovery),
        }


@dataclass(frozen=True)
class GatewayResidentProjection:
    """Credential-free facts retained from one coherent Gateway observation."""

    candidates: tuple[ResidentResourceCandidate, ...]
    meta_tools: tuple[ResidentToolDescriptor, ...]
    requestable_resources: tuple[GatewayRequestableResource, ...]
    requestable_discovery: str


def _error(reason: str) -> GatewayProjectionError:
    return GatewayProjectionError(reason)


def _mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(reason)
    return value


def _string(value: Any, reason: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise _error(reason)
    return text


def _integer(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise _error(reason)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise _error(reason) from exc


def _strings(value: Any, reason: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise _error(reason)
    result = tuple(str(item or "").strip() for item in value)
    if "" in result or len(result) != len(set(result)):
        raise _error(reason)
    return tuple(sorted(result))


def _gateway_binding(binding: ResourceBinding) -> None:
    if not isinstance(binding, ResourceBinding) or binding.mode != "gateway":
        raise _error("gateway_binding_invalid")
    if not all(
        str(value or "").strip()
        for value in (
            binding.server_id,
            binding.alias,
            binding.transport,
            binding.endpoint,
        )
    ):
        raise _error("gateway_binding_incomplete")
    endpoint = urlsplit(binding.endpoint)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise _error("gateway_binding_endpoint_invalid")


def _accepted_descriptor(grant: DelegatedResourceGrant) -> AcceptedDescriptor:
    try:
        return AcceptedDescriptor(
            revision=grant.accepted_revision,
            digest=grant.accepted_digest,
            operation_digests=grant.operation_accepted_digests,
        )
    except Exception as exc:
        raise _error("gateway_card_accepted_descriptor_invalid") from exc


def _resource_reason(value: str) -> AvailabilityReason:
    return _RESOURCE_REASON_MAP.get(
        str(value or "").strip(),
        AvailabilityReason.RESOURCE_PROVIDER_UNAVAILABLE,
    )


def _recovery(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        raise _error("gateway_resource_recovery_invalid")
    links: list[dict[str, str]] = []
    codes: set[str] = set()
    for raw in value:
        item = _mapping(raw, "gateway_resource_recovery_invalid")
        code = _string(item.get("code"), "gateway_resource_recovery_invalid")
        href = _string(item.get("href"), "gateway_resource_recovery_invalid")
        if code in codes or not href.startswith(("https://", "/")):
            raise _error("gateway_resource_recovery_invalid")
        codes.add(code)
        links.append({"code": code, "href": href})
    return {"links": links} if links else {}


def _card_index(
    card: DelegatedCardSnapshot,
) -> dict[str, DelegatedResourceGrant]:
    result: dict[str, DelegatedResourceGrant] = {}
    for grant in card.resources:
        if not grant.resource_id or grant.resource_id in result:
            raise _error("gateway_card_resource_duplicate")
        if grant.identity_scope != card.identity_scope:
            raise _error("gateway_card_resource_scope_mismatch")
        result[grant.resource_id] = grant
    return result


def _validate_access_header(
    card: DelegatedCardSnapshot,
    access: Mapping[str, Any],
) -> None:
    if access.get("schema") != _ACCESS_SCHEMA:
        raise _error("gateway_access_schema_mismatch")
    caller = _mapping(access.get("caller"), "gateway_access_caller_invalid")
    if (
        caller.get("type") != "resident"
        or caller.get("profile_id") != card.client_id
        or caller.get("access_id") != card.access_id
    ):
        raise _error("gateway_access_caller_mismatch")
    public_card = _mapping(access.get("card"), "gateway_access_card_invalid")
    expected_expiry = card.expires_at_epoch or 0
    expected_status = "active" if card.active else "revoked"
    if (
        _integer(public_card.get("revision"), "gateway_access_card_invalid")
        != card.revision
        or public_card.get("status") != expected_status
        or _integer(public_card.get("expires_at"), "gateway_access_card_invalid")
        != expected_expiry
        or public_card.get("source") != card.source
        or public_card.get("identity_scope") != card.identity_scope
    ):
        raise _error("gateway_access_card_mismatch")


def _validate_accepted_descriptor(
    raw: Any,
    grant: DelegatedResourceGrant,
) -> AcceptedDescriptor:
    accepted = _mapping(raw, "gateway_access_accepted_descriptor_invalid")
    operations = _mapping(
        accepted.get("operation_digests"),
        "gateway_access_accepted_descriptor_invalid",
    )
    expected = _accepted_descriptor(grant)
    if (
        accepted.get("revision") != expected.revision
        or accepted.get("digest") != expected.digest
        or dict(operations) != dict(expected.operation_digests)
    ):
        raise _error("gateway_access_accepted_descriptor_mismatch")
    return expected


def _validate_policies(raw: Any, grant: DelegatedResourceGrant) -> None:
    policies = _mapping(raw, "gateway_access_policies_invalid")
    if set(policies) != set(grant.invocation_policies):
        raise _error("gateway_access_policies_mismatch")
    for operation, expected in grant.invocation_policies.items():
        policy = _mapping(policies.get(operation), "gateway_access_policy_invalid")
        remaining = policy.get("remaining")
        if (
            policy.get("mode") != expected.mode
            or _integer(policy.get("revision"), "gateway_access_policy_invalid")
            != expected.revision
            or (None if remaining is None else _integer(remaining, "gateway_access_policy_invalid"))
            != expected.remaining
            or not str(policy.get("state") or "").strip()
        ):
            raise _error("gateway_access_policy_mismatch")


def _current_descriptor(
    raw: Any,
    grant: DelegatedResourceGrant,
) -> tuple[str, str]:
    if raw is None:
        return grant.current_revision, grant.current_digest
    current = _mapping(raw, "gateway_access_current_descriptor_invalid")
    revision = _string(
        current.get("revision"), "gateway_access_current_descriptor_invalid"
    )
    digest = _string(current.get("digest"), "gateway_access_current_descriptor_invalid")
    _string(current.get("state"), "gateway_access_current_descriptor_invalid")
    if grant.current_revision and revision != grant.current_revision:
        raise _error("gateway_access_current_descriptor_mismatch")
    if grant.current_digest and digest != grant.current_digest:
        raise _error("gateway_access_current_descriptor_mismatch")
    return revision, digest


def _tool_index(
    tools: Iterable[GatewayTool],
    grants: Mapping[str, DelegatedResourceGrant],
    accepted: Mapping[str, AcceptedDescriptor],
    providers: Mapping[str, str],
) -> tuple[dict[tuple[str, str], GatewayTool], tuple[ResidentToolDescriptor, ...]]:
    routed: dict[tuple[str, str], GatewayTool] = {}
    meta: list[ResidentToolDescriptor] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, GatewayTool) or tool.name in names:
            raise _error("gateway_tool_invalid_or_duplicate")
        names.add(tool.name)
        route = tool.route
        if route is None:
            if tool.name != ACCESS_DESCRIBE_TOOL or meta:
                raise _error("gateway_meta_tool_invalid")
            meta.append(
                ResidentToolDescriptor(
                    name=tool.name,
                    operation=tool.name,
                    description=tool.description,
                    input_schema=dict(tool.input_schema),
                    output_schema=(
                        dict(tool.output_schema)
                        if tool.output_schema is not None
                        else None
                    ),
                )
            )
            continue
        if not isinstance(route, GatewayToolRoute):
            raise _error("gateway_tool_route_invalid")
        grant = grants.get(route.resource_id)
        if grant is None or route.operation not in set(grant.operations):
            raise _error("gateway_tool_outside_card")
        if route.resource_kind != grant.resource_kind:
            raise _error("gateway_tool_resource_kind_mismatch")
        provider_id = providers.get(route.resource_id, "")
        if route.provider_id != provider_id:
            raise _error("gateway_tool_provider_mismatch")
        try:
            expected_identity = accepted[route.resource_id].operation_identity(
                route.operation
            )
            expected_name = qualified_tool_name(route)
        except Exception as exc:
            raise _error("gateway_tool_authority_invalid") from exc
        if (
            route.accepted_descriptor_identity != expected_identity
            or tool.name != expected_name
        ):
            raise _error("gateway_tool_authority_mismatch")
        key = (route.resource_id, route.operation)
        if key in routed:
            raise _error("gateway_tool_route_duplicate")
        if (
            grant.operation_states.get(route.operation)
            is not AvailabilityReason.AVAILABLE
        ):
            raise _error("gateway_tool_not_current")
        accepted_digest = grant.operation_accepted_digests.get(route.operation, "")
        current_digest = grant.operation_current_digests.get(route.operation, "")
        if current_digest and current_digest != accepted_digest:
            raise _error("gateway_tool_descriptor_mismatch")
        routed[key] = tool
    if len(meta) != 1:
        raise _error("gateway_access_describe_tool_missing")
    return routed, tuple(meta)


def _requestable_resources(
    *,
    card: DelegatedCardSnapshot,
    raw: Any,
    state: str,
    granted_ids: set[str],
) -> tuple[GatewayRequestableResource, ...]:
    if not isinstance(raw, list):
        raise _error("gateway_requestable_resources_invalid")
    if state != "permitted" and raw:
        raise _error("gateway_requestable_resources_unexpected")
    result: list[GatewayRequestableResource] = []
    seen: set[str] = set()
    for value in raw:
        item = _mapping(value, "gateway_requestable_resource_invalid")
        resource_id = _string(
            item.get("resource_id"), "gateway_requestable_resource_invalid"
        )
        identity_scope = _string(
            item.get("identity_scope"), "gateway_requestable_resource_invalid"
        )
        if (
            resource_id in seen
            or resource_id in granted_ids
            or identity_scope != card.identity_scope
        ):
            raise _error("gateway_requestable_resource_mismatch")
        seen.add(resource_id)
        recovery = public_mapping(
            _mapping(
                item.get("recovery", {}), "gateway_requestable_recovery_invalid"
            ),
            reason="gateway_requestable_recovery_not_public",
        )
        result.append(
            GatewayRequestableResource(
                resource_id=resource_id,
                resource_kind=_string(
                    item.get("kind"), "gateway_requestable_resource_invalid"
                ),
                display_name=_string(
                    item.get("display_label"),
                    "gateway_requestable_resource_invalid",
                ),
                identity_scope=identity_scope,
                reason=_string(
                    item.get("reason"), "gateway_requestable_resource_invalid"
                ),
                recovery=recovery,
            )
        )
    return tuple(sorted(result, key=lambda item: item.resource_id))


def gateway_resident_projection(
    *,
    card: DelegatedCardSnapshot,
    tools: Iterable[GatewayTool],
    access: Mapping[str, Any],
    binding: ResourceBinding,
) -> GatewayResidentProjection:
    """Join Card authority and current Gateway facts from one turn.

    The access mapping and tool rows are credential-free Gateway outputs. Any
    disagreement with the independently loaded Card snapshot fails closed so
    an observation race can be retried on the next turn.
    """

    if not isinstance(card, DelegatedCardSnapshot):
        raise _error("gateway_card_snapshot_invalid")
    _gateway_binding(binding)
    try:
        public_access = public_mapping(
            _mapping(access, "gateway_access_invalid"),
            reason="gateway_access_not_public",
        )
    except GatewayProjectionError:
        raise
    except Exception as exc:
        raise _error("gateway_access_not_public") from exc
    _validate_access_header(card, public_access)

    grants = _card_index(card)
    raw_resources = public_access.get("resources")
    if not isinstance(raw_resources, list):
        raise _error("gateway_access_resources_invalid")
    rows: dict[str, Mapping[str, Any]] = {}
    accepted: dict[str, AcceptedDescriptor] = {}
    providers: dict[str, str] = {}
    resource_reasons: dict[str, AvailabilityReason] = {}
    resource_recovery: dict[str, dict[str, Any]] = {}
    current_revisions: dict[str, str] = {}
    current_digests: dict[str, str] = {}

    for raw in raw_resources:
        row = _mapping(raw, "gateway_access_resource_invalid")
        resource_id = _string(
            row.get("resource_id"), "gateway_access_resource_invalid"
        )
        if resource_id in rows:
            raise _error("gateway_access_resource_duplicate")
        grant = grants.get(resource_id)
        if grant is None:
            raise _error("gateway_access_resource_outside_card")
        if (
            row.get("kind") != grant.resource_kind
            or row.get("identity_scope") != card.identity_scope
            or _strings(row.get("grants"), "gateway_access_grants_invalid")
            != tuple(sorted(grant.claims))
            or _strings(row.get("operations"), "gateway_access_operations_invalid")
            != tuple(sorted(grant.operations))
        ):
            raise _error("gateway_access_resource_mismatch")
        state = _string(row.get("state"), "gateway_access_resource_invalid")
        if state not in {"active", "disabled"}:
            raise _error("gateway_access_resource_state_invalid")
        reason_text = str(row.get("unavailable_reason") or "").strip()
        reason = _resource_reason(reason_text)
        if state == "disabled" and reason is AvailabilityReason.AVAILABLE:
            reason = AvailabilityReason.CONNECTOR_DISABLED
        accepted[resource_id] = _validate_accepted_descriptor(
            row.get("accepted_descriptor"), grant
        )
        _validate_policies(row.get("invocation_policies"), grant)
        current_revision, current_digest = _current_descriptor(
            row.get("current_descriptor"), grant
        )
        rows[resource_id] = row
        providers[resource_id] = _string(
            row.get("provider_id"), "gateway_access_provider_invalid"
        )
        resource_reasons[resource_id] = reason
        resource_recovery[resource_id] = _recovery(row.get("recovery"))
        current_revisions[resource_id] = current_revision
        current_digests[resource_id] = current_digest

    if set(rows) != set(grants):
        raise _error("gateway_access_card_resources_mismatch")

    routed, meta_tools = _tool_index(tools, grants, accepted, providers)
    candidates: list[ResidentResourceCandidate] = []
    for resource_id in sorted(grants):
        grant = grants[resource_id]
        row = rows[resource_id]
        reason = resource_reasons[resource_id]
        recovery = resource_recovery[resource_id]
        descriptors: list[ResidentToolDescriptor] = []
        statuses: dict[str, ResidentToolStatus] = {}
        for operation in sorted(grant.operations):
            state = grant.operation_states.get(operation)
            if state is None:
                raise _error("gateway_card_operation_state_missing")
            listed = routed.get((resource_id, operation))
            if listed is not None:
                descriptor = ResidentToolDescriptor(
                    name=listed.name,
                    operation=operation,
                    description=listed.description,
                    input_schema=dict(listed.input_schema),
                    output_schema=(
                        dict(listed.output_schema)
                        if listed.output_schema is not None
                        else None
                    ),
                )
                status = ResidentToolStatus(ready=True)
            else:
                route = GatewayToolRoute(
                    resource_id=resource_id,
                    resource_kind=grant.resource_kind,
                    operation=operation,
                    accepted_descriptor_identity=(
                        accepted[resource_id].operation_identity(operation)
                    ),
                    provider_id=providers[resource_id],
                )
                descriptor = ResidentToolDescriptor(
                    name=qualified_tool_name(route),
                    operation=operation,
                )
                unavailable = state
                if unavailable is AvailabilityReason.AVAILABLE:
                    unavailable = (
                        reason
                        if reason is not AvailabilityReason.AVAILABLE
                        else AvailabilityReason.RESOURCE_PROVIDER_UNAVAILABLE
                    )
                status = ResidentToolStatus(
                    ready=False,
                    reason=unavailable,
                    recovery=(
                        grant.operation_recovery.get(operation)
                        or grant.operation_recovery.get("*")
                        or recovery
                    ),
                )
            descriptors.append(descriptor)
            statuses[operation] = status

        candidates.append(
            ResidentResourceCandidate(
                resource_id=resource_id,
                resource_kind=grant.resource_kind,
                server_id=binding.server_id,
                alias=binding.alias,
                display_name=_string(
                    row.get("display_label"), "gateway_access_resource_invalid"
                ),
                authority_source=AuthoritySource.DELEGATED_CARD,
                tools=tuple(descriptors),
                binding=binding,
                tenant=card.tenant,
                project=card.project,
                application=card.application,
                agent_id=card.agent_id,
                grantor_subject=card.grantor_subject,
                identity_scope=card.identity_scope,
                required_claims=grant.claims,
                descriptor_revision=grant.accepted_revision,
                provider_revision=current_revisions[resource_id],
                unavailable_reason=reason,
                tool_status=statuses,
                enabled=row.get("state") == "active",
                credential_ready=reason is not AvailabilityReason.CREDENTIAL_MISSING,
                descriptor_accepted=True,
                recovery=recovery,
            )
        )

    requestable_state = _string(
        public_access.get("requestable_discovery"),
        "gateway_requestable_discovery_invalid",
    )
    if requestable_state not in _REQUESTABLE_STATES:
        raise _error("gateway_requestable_discovery_invalid")
    requestable = _requestable_resources(
        card=card,
        raw=public_access.get("requestable_resources"),
        state=requestable_state,
        granted_ids=set(grants),
    )
    return GatewayResidentProjection(
        candidates=tuple(candidates),
        meta_tools=meta_tools,
        requestable_resources=requestable,
        requestable_discovery=requestable_state,
    )


__all__ = [
    "GatewayProjectionError",
    "GatewayRequestableResource",
    "GatewayResidentProjection",
    "gateway_resident_projection",
]
