# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Adapt Connection Hub's public card view for resident projection."""

from __future__ import annotations

from typing import Any, Mapping

from connection_hub.delegated_credentials.cards.model import (
    CARD_STATE_ACTIVE,
    CARD_STATE_REVOKED,
)
from connection_hub.delegated_credentials.cards.read_model import (
    CALLER_KIND_RESIDENT,
    OPERATION_STATE_CHANGED,
    OPERATION_STATE_CURRENT,
    OPERATION_STATE_REMOVED,
    OPERATION_STATE_UNKNOWN,
    CardOperationView,
    CardResourceView,
    DelegatedCardView,
)

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AvailabilityReason,
    DelegatedCardSnapshot,
    DelegatedResourceGrant,
    InvocationPolicyState,
)


class ResidentCardAdapterError(ValueError):
    """The public card view cannot be projected without widening authority."""


_RESOURCE_STATES = {
    OPERATION_STATE_CURRENT: AvailabilityReason.AVAILABLE,
    # Operation rows carry the exact changed/removed state. Keeping the
    # resource available preserves unaffected selected operations.
    OPERATION_STATE_CHANGED: AvailabilityReason.AVAILABLE,
    OPERATION_STATE_REMOVED: AvailabilityReason.RESOURCE_NOT_CURRENT,
    # Unknown operation rows fail closed individually, including wildcard
    # rows, while retaining their more precise reason.
    OPERATION_STATE_UNKNOWN: AvailabilityReason.AVAILABLE,
}

_OPERATION_STATES = {
    OPERATION_STATE_CURRENT: AvailabilityReason.AVAILABLE,
    OPERATION_STATE_CHANGED: AvailabilityReason.OPERATION_DESCRIPTOR_CHANGED,
    OPERATION_STATE_REMOVED: AvailabilityReason.OPERATION_REMOVED,
    OPERATION_STATE_UNKNOWN: AvailabilityReason.OPERATION_STATE_UNKNOWN,
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _error(code: str, *parts: Any) -> ResidentCardAdapterError:
    suffix = ":".join(_clean(part) for part in parts if _clean(part))
    return ResidentCardAdapterError(f"{code}:{suffix}" if suffix else code)


def _mapped_state(
    state: str,
    mapping: Mapping[str, AvailabilityReason],
    *,
    code: str,
    resource: str,
    operation: str = "",
) -> AvailabilityReason:
    normalized = _clean(state).lower()
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise _error(code, resource, operation, normalized) from exc


def _policy(
    value: Mapping[str, Any] | None,
    *,
    resource: str,
    operation: str,
) -> InvocationPolicyState | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _error("resident_card_policy_invalid", resource, operation)

    mode = _clean(value.get("mode")).lower()
    if mode not in {"always", "once"}:
        raise _error("resident_card_policy_mode_invalid", resource, operation, mode)
    raw_revision = value.get("revision")
    if isinstance(raw_revision, bool):
        raise _error("resident_card_policy_revision_invalid", resource, operation)
    try:
        revision = int(raw_revision)
    except (TypeError, ValueError) as exc:
        raise _error(
            "resident_card_policy_revision_invalid", resource, operation
        ) from exc
    if revision < 1:
        raise _error("resident_card_policy_revision_invalid", resource, operation)

    remaining: int | None = None
    if mode == "once":
        raw_remaining = value.get("remaining")
        if isinstance(raw_remaining, bool):
            raise _error("resident_card_policy_remaining_invalid", resource, operation)
        try:
            remaining = int(raw_remaining)
        except (TypeError, ValueError) as exc:
            raise _error(
                "resident_card_policy_remaining_invalid", resource, operation
            ) from exc
        if remaining not in {0, 1}:
            raise _error("resident_card_policy_remaining_invalid", resource, operation)
    elif value.get("remaining") is not None:
        raise _error("resident_card_policy_remaining_invalid", resource, operation)

    return InvocationPolicyState(
        mode=mode,
        remaining=remaining,
        revision=revision,
    )


def _operation_views(
    resource: CardResourceView,
) -> tuple[
    tuple[str, ...],
    dict[str, AvailabilityReason],
    dict[str, InvocationPolicyState],
    dict[str, str],
    dict[str, str],
]:
    names: list[str] = []
    states: dict[str, AvailabilityReason] = {}
    policies: dict[str, InvocationPolicyState] = {}
    accepted_digests: dict[str, str] = {}
    current_digests: dict[str, str] = {}
    for operation in sorted(resource.operations, key=lambda item: _clean(item.name)):
        if not isinstance(operation, CardOperationView):
            raise _error("resident_card_operation_invalid", resource.resource)
        name = _clean(operation.name)
        if not name:
            raise _error("resident_card_operation_name_required", resource.resource)
        if name in states:
            raise _error("resident_card_operation_duplicate", resource.resource, name)
        names.append(name)
        states[name] = _mapped_state(
            operation.state,
            _OPERATION_STATES,
            code="resident_card_operation_state_invalid",
            resource=resource.resource,
            operation=name,
        )
        policy = _policy(
            operation.policy,
            resource=resource.resource,
            operation=name,
        )
        if policy is not None:
            policies[name] = policy
        accepted = _clean(operation.accepted_digest)
        current = _clean(operation.current_digest)
        if accepted:
            accepted_digests[name] = accepted
        if current:
            current_digests[name] = current
    return (
        tuple(names),
        states,
        policies,
        accepted_digests,
        current_digests,
    )


def _resource_grant(
    resource: CardResourceView,
    *,
    identity_scope: str,
) -> DelegatedResourceGrant:
    resource_id = _clean(resource.resource)
    if not resource_id:
        raise _error("resident_card_resource_required")
    resource_scope = _clean(resource.identity_scope) or "grantor"
    if resource_scope != identity_scope:
        raise _error(
            "resident_card_resource_identity_scope_mismatch",
            resource_id,
            resource_scope,
            identity_scope,
        )
    operations, states, policies, accepted, current = _operation_views(resource)
    named_service_operations = {
        _clean(namespace): tuple(
            sorted({_clean(operation) for operation in values if _clean(operation)})
        )
        for namespace, values in sorted(resource.named_service_operations.items())
        if _clean(namespace)
    }
    return DelegatedResourceGrant(
        resource_id=resource_id,
        resource_kind=_clean(resource.kind),
        identity_scope=resource_scope,
        claims=tuple(sorted({_clean(item) for item in resource.grants if _clean(item)})),
        operations=operations,
        invocation_policies=policies,
        resource_state=_mapped_state(
            resource.state,
            _RESOURCE_STATES,
            code="resident_card_resource_state_invalid",
            resource=resource_id,
        ),
        operation_states=states,
        operation_accepted_digests=accepted,
        operation_current_digests=current,
        accepted_revision=_clean(resource.accepted_revision),
        current_revision=_clean(resource.current_revision),
        accepted_digest=_clean(resource.accepted_digest),
        current_digest=_clean(resource.current_digest),
        named_service_operations=named_service_operations,
    )


def delegated_card_snapshot_from_view(
    view: DelegatedCardView,
    *,
    tenant: str,
    project: str,
) -> DelegatedCardSnapshot:
    """Convert Card's non-secret read model without reproducing its identity."""

    profile = view.profile
    if view.caller_kind != CALLER_KIND_RESIDENT or profile is None:
        raise _error("resident_card_profile_required", view.access_id)
    identity_scope = _clean(view.identity_scope) or "grantor"
    if (
        _clean(view.grantor_subject) != profile.grantor_subject
        or _clean(view.client_id) != profile.client_id
    ):
        raise _error("resident_card_profile_mismatch", view.access_id)
    access_id = _clean(view.access_id)
    if not access_id:
        raise _error("resident_card_access_id_required")
    selected_tenant = _clean(tenant)
    selected_project = _clean(project)
    if not selected_tenant or not selected_project:
        raise _error("resident_card_deployment_scope_required")
    card_state = _clean(view.state).lower()
    if card_state not in {CARD_STATE_ACTIVE, CARD_STATE_REVOKED}:
        raise _error("resident_card_state_invalid", access_id, card_state)
    if isinstance(view.card_revision, bool):
        raise _error("resident_card_revision_invalid", access_id)
    try:
        card_revision = int(view.card_revision)
        raw_expiry = int(view.expires_at)
    except (TypeError, ValueError) as exc:
        raise _error("resident_card_revision_or_expiry_invalid", access_id) from exc
    if card_revision < 1 or raw_expiry < 0:
        raise _error("resident_card_revision_or_expiry_invalid", access_id)
    # A legacy resource-derived access id remains valid until Card folds it
    # into profile.access_id. The public view is the identity authority here.
    resources = tuple(
        _resource_grant(resource, identity_scope=identity_scope)
        for resource in sorted(
            view.resources,
            key=lambda item: (_clean(item.identity_scope), _clean(item.resource)),
        )
    )
    expires_at = raw_expiry if raw_expiry > 0 else None
    account_scope = {
        _clean(provider): {
            _clean(account): tuple(
                sorted({_clean(claim) for claim in claims if _clean(claim)})
            )
            for account, claims in sorted(accounts.items())
            if _clean(account)
        }
        for provider, accounts in sorted(view.account_scope.items())
        if _clean(provider)
    }
    return DelegatedCardSnapshot(
        access_id=access_id,
        client_id=profile.client_id,
        revision=card_revision,
        tenant=selected_tenant,
        project=selected_project,
        grantor_subject=profile.grantor_subject,
        application=profile.application,
        agent_id=profile.agent_id,
        identity_scope=identity_scope,
        source=_clean(view.source),
        catalog_version=_clean(view.catalog_version),
        resources=resources,
        account_scope=account_scope,
        active=card_state == CARD_STATE_ACTIVE,
        expires_at_epoch=expires_at,
    )


__all__ = [
    "ResidentCardAdapterError",
    "delegated_card_snapshot_from_view",
]
