# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Resident-agent effective resource projection."""

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.card_adapter import (
    ResidentCardAdapterError,
    delegated_card_snapshot_from_view,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.card_loader import (
    CardBackedResidentResourceFactsLoader,
    ResidentCardViewReader,
    ResidentResourceCandidateLoader,
)

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.descriptor import (
    ResidentResourceDescriptorError,
    parse_resource_family_ceiling,
    resource_family_catalog_from_bundle_props,
    resource_family_ceilings_from_bundle_props,
    resident_agent_ceiling_from_bundle_props,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.gateway_adapter import (
    GatewayProjectionError,
    GatewayRequestableResource,
    GatewayResidentProjection,
    gateway_resident_projection,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.gateway_observation import (
    GatewayObservationError,
    ResidentGatewayObservation,
    compose_gateway_resource_facts,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AuthoritySource,
    AvailabilityReason,
    ConnectedAccountStatus,
    ConversationNarrowing,
    DelegatedCardSnapshot,
    DelegatedResourceGrant,
    EffectiveResidentInventory,
    InvocationPolicyState,
    RejectedResidentResource,
    ResidentAgentCeiling,
    ResidentResourceCandidate,
    ResidentToolStatus,
    ResidentToolDescriptor,
    ResolvedResidentResource,
    ResolvedResidentTool,
    ResourceBinding,
    ResourceFamilyCeiling,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.projection import (
    attach_effective_resource_catalog,
    effective_resource_catalog,
    resident_projection_base_catalog,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.resolver import (
    conversation_narrowing_from_selection,
    resolve_effective_resident_resources,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.service import (
    CurrentResidentResourceFacts,
    EffectiveResidentRuntimeProjection,
    ResidentResourceFactsLoader,
    resolve_current_resident_resources,
    resolve_current_resident_runtime_projection,
    resolve_resident_resource_facts,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.runtime_binding import (
    GatewayRuntimeConnection,
    GatewayRuntimeHeadersProvider,
    GatewayRuntimePlan,
    ResidentRuntimeBindingError,
    apply_gateway_runtime_connections,
    bind_gateway_runtime_context,
    delegated_mcp_bindings_from_catalog,
    gateway_connection_descriptors,
    gateway_runtime_connections,
    gateway_runtime_plan,
    gateway_services_config,
    gateway_tool_overrides,
    merge_gateway_services_config,
    remove_direct_delegated_mcp_bindings,
)

__all__ = [
    "AuthoritySource",
    "AvailabilityReason",
    "CardBackedResidentResourceFactsLoader",
    "ConnectedAccountStatus",
    "ConversationNarrowing",
    "CurrentResidentResourceFacts",
    "DelegatedCardSnapshot",
    "DelegatedResourceGrant",
    "EffectiveResidentInventory",
    "EffectiveResidentRuntimeProjection",
    "GatewayObservationError",
    "GatewayProjectionError",
    "GatewayRequestableResource",
    "GatewayResidentProjection",
    "GatewayRuntimeConnection",
    "GatewayRuntimeHeadersProvider",
    "GatewayRuntimePlan",
    "InvocationPolicyState",
    "RejectedResidentResource",
    "ResidentAgentCeiling",
    "ResidentCardAdapterError",
    "ResidentCardViewReader",
    "ResidentResourceCandidate",
    "ResidentResourceDescriptorError",
    "ResidentResourceFactsLoader",
    "ResidentRuntimeBindingError",
    "ResidentResourceCandidateLoader",
    "ResidentToolDescriptor",
    "ResidentToolStatus",
    "ResolvedResidentResource",
    "ResolvedResidentTool",
    "ResourceBinding",
    "ResourceFamilyCeiling",
    "attach_effective_resource_catalog",
    "apply_gateway_runtime_connections",
    "bind_gateway_runtime_context",
    "conversation_narrowing_from_selection",
    "compose_gateway_resource_facts",
    "delegated_card_snapshot_from_view",
    "delegated_mcp_bindings_from_catalog",
    "effective_resource_catalog",
    "resident_projection_base_catalog",
    "gateway_resident_projection",
    "gateway_connection_descriptors",
    "gateway_runtime_connections",
    "gateway_runtime_plan",
    "gateway_services_config",
    "gateway_tool_overrides",
    "merge_gateway_services_config",
    "remove_direct_delegated_mcp_bindings",
    "parse_resource_family_ceiling",
    "resource_family_catalog_from_bundle_props",
    "resource_family_ceilings_from_bundle_props",
    "resident_agent_ceiling_from_bundle_props",
    "resolve_effective_resident_resources",
    "resolve_current_resident_resources",
    "resolve_current_resident_runtime_projection",
    "resolve_resident_resource_facts",
    "ResidentGatewayObservation",
]
