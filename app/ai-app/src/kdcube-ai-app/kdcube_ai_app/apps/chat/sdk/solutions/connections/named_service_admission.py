# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Connection Hub implementations of named-service admission."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.solutions.connections.agent_account_scope import (
    bind_agent_account_scope,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.catalog.authorization import (
    CAPABILITY_NAMED_SERVICE_OPERATION,
    ActiveCatalogCapabilities,
    CapabilityRequest,
    CardProvenance,
    authorize_current_capability,
    card_boundary_denial,
    catalog_unavailable_denial,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.named_service_policy import (
    boundary_permits_operation,
    configured_named_service_operations,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.admission import (
    ADMISSION_MODE_APPLICATION,
    DELEGATED_SELECTOR_AGENT,
    DELEGATED_SELECTOR_BEARER,
    NamedServiceAdmission,
    NamedServiceAdmissionDecision,
    NamedServiceAdmissionSelector,
    effective_named_service_operation,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    NamedServiceError,
    NamedServiceRequest,
    NamedServiceResponse,
)

MANAGED_ADMISSION_STATE_ATTR = "named_service_admission_snapshot"
DELEGATED_CARD_BINDING_SCHEMA = "connection_hub.delegated_card_binding.v1"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _denial_response(
    payload: Mapping[str, Any],
    *,
    default_status: int = 403,
) -> NamedServiceResponse:
    body = dict(payload or {})
    error = body.get("error")
    error = dict(error) if isinstance(error, Mapping) else {}
    details = error.get("details")
    details = dict(details) if isinstance(details, Mapping) else {}
    for key in ("where", "retryable"):
        if key in error:
            details.setdefault(key, error[key])
    if isinstance(body.get("consent"), Mapping):
        details.setdefault("consent", dict(body["consent"]))
    status = int(body.get("status") or details.get("status") or default_status)
    return NamedServiceResponse(
        ok=False,
        status=status,
        ret=dict(body.get("ret") or {}) if isinstance(body.get("ret"), Mapping) else {},
        error=NamedServiceError(
            code=_clean(error.get("code")) or "named_service_admission_denied",
            message=_clean(error.get("message")) or "Named-service admission was denied.",
            details={**details, "status": status},
            fix=dict(error.get("fix") or {}) if isinstance(error.get("fix"), Mapping) else {},
        ),
    )


@dataclass(frozen=True)
class DelegatedAccountExecutionScope:
    account_scope: Mapping[str, Any]
    client_id: str
    resource: str

    def bind(self):
        return bind_agent_account_scope(
            self.account_scope,
            client_id=self.client_id,
            resource=self.resource,
        )


@dataclass(frozen=True)
class ManagedNamedServiceAdmissionSnapshot:
    catalog: Any
    access_id: str
    client_id: str
    grantor_user_id: str
    delegate_identity: str
    expires_at: int
    resource: str
    request_resource: str
    outer_operation: str
    card_revision: int
    card_catalog_version: str
    named_services: Mapping[str, Any]
    named_services_present: bool
    account_scope: Mapping[str, Any]

    def selector(self) -> NamedServiceAdmissionSelector:
        return NamedServiceAdmissionSelector(
            mode="delegated",
            source="managed_mcp.named_services",
            delegated_kind=DELEGATED_SELECTOR_BEARER,
            access_id=self.access_id,
            client_id=self.client_id,
            grantor_user_id=self.grantor_user_id,
            delegate_identity=self.delegate_identity,
            expires_at=self.expires_at,
        )


class ManagedNamedServiceAdmissionAuthorizer:
    def __init__(self, snapshot: ManagedNamedServiceAdmissionSnapshot) -> None:
        self._snapshot = snapshot

    async def authorize(self, request: NamedServiceRequest) -> NamedServiceAdmissionDecision:
        snapshot = self._snapshot
        operation = effective_named_service_operation(request)
        capability = CapabilityRequest(
            kind=CAPABILITY_NAMED_SERVICE_OPERATION,
            resource=snapshot.resource,
            request_resource=snapshot.request_resource,
            surface="named_service",
            outer_operation=snapshot.outer_operation,
            namespace=request.namespace or "",
            operation=operation,
        )
        provenance = CardProvenance(
            access_id=snapshot.access_id,
            card_revision=snapshot.card_revision,
            catalog_version=snapshot.card_catalog_version,
        )
        removed = authorize_current_capability(
            catalog=snapshot.catalog,
            provenance=provenance,
            request=capability,
        )
        if removed is not None:
            return NamedServiceAdmissionDecision.deny(_denial_response(removed))
        if not snapshot.named_services_present:
            return NamedServiceAdmissionDecision.deny(
                NamedServiceResponse.error_response(
                    code="delegated_card_boundary_unavailable",
                    message="The delegated card does not carry a materialized named-service boundary.",
                    status=503,
                    namespace=request.namespace,
                    details={"retryable": False, "access_id": snapshot.access_id},
                )
            )
        if not boundary_permits_operation(
            snapshot.named_services,
            namespace=request.namespace or "",
            operation=operation,
        ):
            return NamedServiceAdmissionDecision.deny(
                _denial_response(
                    card_boundary_denial(provenance=provenance, request=capability)
                )
            )
        return NamedServiceAdmissionDecision.allow(
            execution_scope=DelegatedAccountExecutionScope(
                account_scope=snapshot.account_scope,
                client_id=snapshot.client_id,
                resource=snapshot.resource,
            ),
            audit={
                "mode": "delegated",
                "source": "managed_mcp.named_services",
                "access_id": snapshot.access_id,
                "card_revision": snapshot.card_revision,
                "card_catalog_version": snapshot.card_catalog_version,
                "active_catalog_version": snapshot.catalog.version,
            },
        )


class _ResolvedHubStateAuthorizer:
    def __init__(
        self,
        *,
        selector: NamedServiceAdmissionSelector,
        state: Mapping[str, Any],
    ) -> None:
        self._selector = selector
        self._state = dict(state or {})

    async def authorize(self, request: NamedServiceRequest) -> NamedServiceAdmissionDecision:
        del request
        state = self._state
        if not state.get("granted"):
            return NamedServiceAdmissionDecision.deny(
                _hub_state_denial(state, selector=self._selector)
            )
        return NamedServiceAdmissionDecision.allow(
            execution_scope=DelegatedAccountExecutionScope(
                account_scope=(
                    state.get("account_scope")
                    if isinstance(state.get("account_scope"), Mapping)
                    else {}
                ),
                client_id=self._selector.client_id,
                resource=_clean(state.get("resource")),
            ),
            audit=_hub_state_audit(state, selector=self._selector),
        )


class HubNamedServiceAdmissionAuthorizer:
    """Resolve current card/catalog authority through Connection Hub."""

    def __init__(self, selector: NamedServiceAdmissionSelector) -> None:
        self._selector = selector

    async def authorize(self, request: NamedServiceRequest) -> NamedServiceAdmissionDecision:
        from kdcube_ai_app.apps.chat.sdk.infra.bundle_operations import call_bundle_named_service
        from kdcube_ai_app.apps.chat.sdk.solutions.connections.connection_edges import (
            DEFAULT_CONNECTION_HUB_BUNDLE_ID,
        )
        from kdcube_ai_app.apps.chat.sdk.solutions.connections.contract import (
            AGENT_GRANT_CHECK,
            NAMESPACE as CONNECTIONS_NAMESPACE,
        )

        selector = self._selector
        operation = effective_named_service_operation(request)
        payload = {
            "client_id": selector.client_id,
            "namespace": request.namespace or "",
            "operation": operation,
        }
        if selector.access_id:
            payload["access_id"] = selector.access_id
        if selector.delegate_identity:
            payload["delegate_identity"] = selector.delegate_identity
        try:
            result = await call_bundle_named_service(
                bundle_id=DEFAULT_CONNECTION_HUB_BUNDLE_ID,
                request={
                    "namespace": CONNECTIONS_NAMESPACE,
                    "operation": AGENT_GRANT_CHECK,
                    "payload": payload,
                },
            )
            value = getattr(result, "value", None)
            response = NamedServiceResponse.coerce(value) if value is not None else None
        except Exception:
            response = None
        if response is None or not response.ok:
            denial = catalog_unavailable_denial("agent_grant_check_unavailable")
            return NamedServiceAdmissionDecision.deny(
                _denial_response(denial, default_status=503)
            )
        state = dict(response.object or {})
        if state.get("granted"):
            return NamedServiceAdmissionDecision.allow(
                execution_scope=DelegatedAccountExecutionScope(
                    account_scope=(
                        state.get("account_scope")
                        if isinstance(state.get("account_scope"), Mapping)
                        else {}
                    ),
                    client_id=selector.client_id,
                    resource=_clean(state.get("resource")),
                ),
                audit=_hub_state_audit(state, selector=selector),
            )
        return NamedServiceAdmissionDecision.deny(
            _hub_state_denial(state, selector=selector)
        )


def _hub_state_audit(
    state: Mapping[str, Any],
    *,
    selector: NamedServiceAdmissionSelector,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "mode": "delegated",
            "source": selector.source,
            "access_id": state.get("access_id") or selector.access_id,
            "card_revision": state.get("card_revision"),
            "card_catalog_version": state.get("card_catalog_version"),
            "active_catalog_version": state.get("active_catalog_version"),
        }.items()
        if value not in (None, "", 0)
    }


def _hub_state_denial(
    state: Mapping[str, Any],
    *,
    selector: NamedServiceAdmissionSelector,
) -> NamedServiceResponse:
    unavailable = _clean(state.get("unavailable"))
    if unavailable:
        return _denial_response(catalog_unavailable_denial(unavailable), default_status=503)
    removed = state.get("removed")
    if isinstance(removed, Mapping):
        return _denial_response(removed)
    not_granted = state.get("not_granted")
    if isinstance(not_granted, Mapping):
        return _denial_response(not_granted)
    card_error = _clean(state.get("card_error"))
    if card_error:
        return NamedServiceResponse.error_response(
            code="delegated_card_not_active",
            message="The delegated access card is absent, expired, revoked, or does not match this caller.",
            status=403,
            details={
                "reason": card_error,
                "access_id": state.get("access_id") or selector.access_id,
            },
        )
    if not state.get("governed"):
        return NamedServiceResponse.error_response(
            code="delegated_named_service_not_governed",
            message="The active delegated-service catalog does not publish this named-service capability.",
            status=403,
            details={"retryable": False},
        )
    return NamedServiceResponse.error_response(
        code="delegated_capability_not_granted",
        message="The delegated caller has not been granted this named-service capability.",
        status=403,
        details={
            "resource": state.get("resource") or "",
            "claims": list(state.get("missing_claims") or state.get("claims") or []),
            "client_id": selector.client_id,
        },
    )


def store_managed_named_service_admission_snapshot(
    request: Any,
    *,
    catalog: Any,
    grant_record: Mapping[str, Any],
    credential: Any,
    resource: str,
    request_resource: str,
    outer_operation: str = "",
) -> ManagedNamedServiceAdmissionSnapshot:
    attrs = getattr(credential, "attrs", None)
    attrs = dict(attrs) if isinstance(attrs, Mapping) else {}
    access_id = _clean(grant_record.get("registry_access_id"))
    client_id = _clean(grant_record.get("client_id") or attrs.get("client_id"))
    grantor_user_id = _clean(
        grant_record.get("grantor_subject")
        or attrs.get("grantor_subject")
        or attrs.get("grantor_user_id")
    )
    delegate_identity = _clean(
        grant_record.get("delegate_subject") or getattr(credential, "subject", "")
    )
    snapshot = ManagedNamedServiceAdmissionSnapshot(
        catalog=catalog,
        access_id=access_id,
        client_id=client_id,
        grantor_user_id=grantor_user_id,
        delegate_identity=delegate_identity,
        expires_at=int(grant_record.get("expires_at") or 0),
        resource=_clean(resource),
        request_resource=_clean(request_resource),
        outer_operation=_clean(outer_operation),
        card_revision=int(grant_record.get("card_revision") or 0),
        card_catalog_version=_clean(grant_record.get("catalog_version")),
        named_services=copy.deepcopy(dict(grant_record.get("named_services") or {})),
        named_services_present=isinstance(grant_record.get("named_services"), Mapping),
        account_scope=copy.deepcopy(dict(grant_record.get("account_scope") or {})),
    )
    setattr(request.state, MANAGED_ADMISSION_STATE_ATTR, snapshot)
    return snapshot


def managed_named_service_admission(request: Any) -> NamedServiceAdmission:
    snapshot = getattr(getattr(request, "state", None), MANAGED_ADMISSION_STATE_ATTR, None)
    if not isinstance(snapshot, ManagedNamedServiceAdmissionSnapshot):
        raise ValueError("Managed named-service admission snapshot is unavailable")
    return NamedServiceAdmission.delegated(
        selector=snapshot.selector(),
        authorizer=ManagedNamedServiceAdmissionAuthorizer(snapshot),
    )


def managed_named_service_catalog_operations(
    request: Any,
) -> dict[str, set[str]]:
    snapshot = getattr(getattr(request, "state", None), MANAGED_ADMISSION_STATE_ATTR, None)
    if not isinstance(snapshot, ManagedNamedServiceAdmissionSnapshot):
        raise ValueError("Managed named-service admission snapshot is unavailable")
    capabilities = snapshot.catalog
    resource_cfg = capabilities.resource_config(
        CapabilityRequest(
            kind=CAPABILITY_NAMED_SERVICE_OPERATION,
            resource=snapshot.resource,
            request_resource=snapshot.request_resource,
            surface="named_service",
            namespace="-",
            operation="-",
        )
    )
    named_services = getattr(resource_cfg, "named_services", None)
    if not isinstance(named_services, Mapping):
        return {}
    offered = configured_named_service_operations(named_services)
    card = configured_named_service_operations(snapshot.named_services)
    return {
        namespace: set(operations) & set(card.get(namespace) or ())
        for namespace, operations in offered.items()
        if set(operations) & set(card.get(namespace) or ())
    }


def delegated_card_binding_from_request(request: Any) -> dict[str, Any]:
    snapshot = getattr(getattr(request, "state", None), MANAGED_ADMISSION_STATE_ATTR, None)
    if not isinstance(snapshot, ManagedNamedServiceAdmissionSnapshot):
        return {}
    if not snapshot.access_id:
        return {}
    return {
        "schema": DELEGATED_CARD_BINDING_SCHEMA,
        "access_id": snapshot.access_id,
        "client_id": snapshot.client_id,
        "grantor_user_id": snapshot.grantor_user_id,
        "delegate_identity": snapshot.delegate_identity,
        "expires_at": snapshot.expires_at,
    }


def native_agent_admission_selector(
    *,
    source_bundle_id: str,
    source_agent_id: str,
    client_id: str,
    grantor_user_id: str,
) -> NamedServiceAdmissionSelector:
    return NamedServiceAdmissionSelector(
        mode="delegated",
        source="named_services.client_tool",
        delegated_kind=DELEGATED_SELECTOR_AGENT,
        client_id=client_id,
        grantor_user_id=grantor_user_id,
        source_bundle_id=source_bundle_id,
        source_agent_id=source_agent_id,
    )


def native_agent_admission_from_state(
    *,
    selector: NamedServiceAdmissionSelector,
    state: Mapping[str, Any],
) -> NamedServiceAdmission:
    return NamedServiceAdmission.delegated(
        selector=selector,
        authorizer=_ResolvedHubStateAuthorizer(selector=selector, state=state),
    )


class NamedServiceAdmissionResolutionError(ValueError):
    pass


def admission_from_relay_selector(
    value: Mapping[str, Any],
    *,
    actor: Mapping[str, Any],
) -> NamedServiceAdmission:
    selector = NamedServiceAdmissionSelector.from_mapping(value)
    source_bundle = _clean(actor.get("source_bundle_id"))
    if selector.mode == ADMISSION_MODE_APPLICATION:
        if not source_bundle:
            raise NamedServiceAdmissionResolutionError(
                "Application admission relay requires a trusted source bundle identity"
        )
        return NamedServiceAdmission.application(source=selector.source)

    actor_user_id = _clean(actor.get("user_id") or actor.get("fingerprint"))
    if selector.grantor_user_id and actor_user_id != selector.grantor_user_id:
        raise NamedServiceAdmissionResolutionError(
            "Relayed delegated selector does not match the carried grantor identity"
        )
    if selector.delegated_kind == DELEGATED_SELECTOR_AGENT:
        if source_bundle != selector.source_bundle_id or _clean(
            actor.get("source_agent_id")
        ) != selector.source_agent_id:
            raise NamedServiceAdmissionResolutionError(
                "Relayed agent-card selector does not match the carried caller"
            )
    elif selector.delegated_kind == DELEGATED_SELECTOR_BEARER:
        if not selector.access_id:
            raise NamedServiceAdmissionResolutionError(
                "Relayed bearer-card admission requires an exact access_id"
            )
        authority = actor.get("identity_authority")
        authority = authority if isinstance(authority, Mapping) else {}
        binding = authority.get("delegated_card_binding")
        binding = binding if isinstance(binding, Mapping) else {}
        expected = {
            "access_id": selector.access_id,
            "client_id": selector.client_id,
            "grantor_user_id": selector.grantor_user_id,
            "delegate_identity": selector.delegate_identity,
        }
        if _clean(binding.get("schema")) != DELEGATED_CARD_BINDING_SCHEMA or any(
            _clean(binding.get(key)) != _clean(expected_value)
            for key, expected_value in expected.items()
            if _clean(expected_value)
        ):
            raise NamedServiceAdmissionResolutionError(
                "Relayed bearer-card selector does not match the authenticated session binding"
            )
    return NamedServiceAdmission.delegated(
        selector=selector,
        authorizer=HubNamedServiceAdmissionAuthorizer(selector),
    )


__all__ = [
    "DELEGATED_CARD_BINDING_SCHEMA",
    "DelegatedAccountExecutionScope",
    "HubNamedServiceAdmissionAuthorizer",
    "MANAGED_ADMISSION_STATE_ATTR",
    "ManagedNamedServiceAdmissionSnapshot",
    "NamedServiceAdmissionResolutionError",
    "admission_from_relay_selector",
    "delegated_card_binding_from_request",
    "managed_named_service_admission",
    "managed_named_service_catalog_operations",
    "native_agent_admission_from_state",
    "native_agent_admission_selector",
    "store_managed_named_service_admission_snapshot",
]
