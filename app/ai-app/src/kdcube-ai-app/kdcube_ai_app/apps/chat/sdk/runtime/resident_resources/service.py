# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Per-turn orchestration for resident-agent resource projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AvailabilityReason,
    DelegatedCardSnapshot,
    EffectiveResidentInventory,
    ResidentAgentCeiling,
    ResidentResourceCandidate,
    ResidentToolDescriptor,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.gateway_adapter import (
    GatewayRequestableResource,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.resolver import (
    conversation_narrowing_from_selection,
    resolve_effective_resident_resources,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.runtime_binding import (
    GatewayRuntimePlan,
    gateway_runtime_plan,
)


@dataclass(frozen=True)
class CurrentResidentResourceFacts:
    """Credential-free card and provider facts observed for one resolution."""

    candidates: tuple[ResidentResourceCandidate, ...]
    card: DelegatedCardSnapshot | None
    observed_at_epoch: int
    card_unavailable_reasons: Mapping[str, AvailabilityReason] = field(
        default_factory=dict
    )
    gateway_meta_tools_by_access_id: Mapping[
        str, tuple[ResidentToolDescriptor, ...]
    ] = field(default_factory=dict)
    gateway_requestable_resources_by_access_id: Mapping[
        str, tuple[GatewayRequestableResource, ...]
    ] = field(default_factory=dict)
    gateway_requestable_discovery_by_access_id: Mapping[str, str] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class EffectiveResidentRuntimeProjection:
    """One coherent effective inventory and all of its runtime translations."""

    inventory: EffectiveResidentInventory
    gateway_plan: GatewayRuntimePlan
    requestable_resources_by_access_id: Mapping[
        str, tuple[GatewayRequestableResource, ...]
    ] = field(default_factory=dict)
    requestable_discovery_by_access_id: Mapping[str, str] = field(
        default_factory=dict
    )


class ResidentResourceFactsLoader(Protocol):
    """Load current facts for one scoped resident caller.

    Implementations resolve live state on every call. They do not return or
    retain delegated bearers or provider credentials.
    """

    async def load_current(
        self,
        *,
        ceiling: ResidentAgentCeiling,
        grantor_subject: str,
    ) -> CurrentResidentResourceFacts: ...


async def resolve_current_resident_resources(
    *,
    loader: ResidentResourceFactsLoader,
    ceiling: ResidentAgentCeiling,
    grantor_subject: str,
    disabled_selection: Mapping[str, Any] | None = None,
) -> EffectiveResidentInventory:
    """Load fresh facts and resolve the effective inventory for this turn."""

    facts = await loader.load_current(
        ceiling=ceiling,
        grantor_subject=grantor_subject,
    )
    return resolve_resident_resource_facts(
        facts=facts,
        ceiling=ceiling,
        grantor_subject=grantor_subject,
        disabled_selection=disabled_selection,
    )


def resolve_resident_resource_facts(
    *,
    facts: CurrentResidentResourceFacts,
    ceiling: ResidentAgentCeiling,
    grantor_subject: str,
    disabled_selection: Mapping[str, Any] | None = None,
) -> EffectiveResidentInventory:
    """Resolve an already loaded observation without reading mutable state."""

    narrowing = conversation_narrowing_from_selection(
        disabled_selection,
        facts.candidates,
    )
    return resolve_effective_resident_resources(
        ceiling=ceiling,
        grantor_subject=grantor_subject,
        candidates=facts.candidates,
        card=facts.card,
        card_unavailable_reasons=facts.card_unavailable_reasons,
        conversation=narrowing,
        now_epoch=facts.observed_at_epoch,
    )


async def resolve_current_resident_runtime_projection(
    *,
    loader: ResidentResourceFactsLoader,
    ceiling: ResidentAgentCeiling,
    grantor_subject: str,
    disabled_selection: Mapping[str, Any] | None = None,
) -> EffectiveResidentRuntimeProjection:
    """Load once, intersect once, then derive every maintained runtime view."""

    facts = await loader.load_current(
        ceiling=ceiling,
        grantor_subject=grantor_subject,
    )
    inventory = resolve_resident_resource_facts(
        facts=facts,
        ceiling=ceiling,
        grantor_subject=grantor_subject,
        disabled_selection=disabled_selection,
    )
    effective_access_ids = {
        resource.access_id
        for resource in inventory.resources
        if resource.access_id
    }
    meta = {
        access_id: tuple(tools)
        for access_id, tools in facts.gateway_meta_tools_by_access_id.items()
        if access_id in effective_access_ids
    }
    return EffectiveResidentRuntimeProjection(
        inventory=inventory,
        gateway_plan=gateway_runtime_plan(
            inventory,
            meta_tools_by_access_id=meta,
        ),
        requestable_resources_by_access_id={
            access_id: tuple(resources)
            for access_id, resources in (
                facts.gateway_requestable_resources_by_access_id.items()
            )
            if access_id in effective_access_ids
        },
        requestable_discovery_by_access_id={
            access_id: state
            for access_id, state in (
                facts.gateway_requestable_discovery_by_access_id.items()
            )
            if access_id in effective_access_ids
        },
    )


__all__ = [
    "CurrentResidentResourceFacts",
    "EffectiveResidentRuntimeProjection",
    "ResidentResourceFactsLoader",
    "resolve_current_resident_resources",
    "resolve_current_resident_runtime_projection",
    "resolve_resident_resource_facts",
]
