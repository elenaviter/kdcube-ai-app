# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Descriptor parser for resident-agent dynamic resource ceilings."""

from __future__ import annotations

from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.event_identity import normalize_agent_id
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AuthoritySource,
    ResidentAgentCeiling,
    ResourceFamilyCeiling,
)


class ResidentResourceDescriptorError(ValueError):
    """Raised when a configured dynamic-resource ceiling is ambiguous."""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return ()
    result: list[str] = []
    for item in value:
        text = _clean(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _agent_block(bundle_props: Mapping[str, Any], agent_id: str) -> Mapping[str, Any]:
    surfaces = bundle_props.get("surfaces")
    consumer = surfaces.get("as_consumer") if isinstance(surfaces, Mapping) else None
    agents = consumer.get("agents") if isinstance(consumer, Mapping) else None
    if not isinstance(agents, Mapping):
        return {}
    requested = _clean(agent_id)
    candidates = (
        requested,
        normalize_agent_id(requested),
        requested.replace(".", "_").replace("-", "_"),
    )
    for candidate in candidates:
        block = agents.get(candidate) if candidate else None
        if isinstance(block, Mapping):
            return block
    return {}


def _positive_int(raw: Any, *, field_name: str, default: int) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise ResidentResourceDescriptorError(f"{field_name}_must_be_positive_integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ResidentResourceDescriptorError(
            f"{field_name}_must_be_positive_integer"
        ) from exc
    if value <= 0:
        raise ResidentResourceDescriptorError(f"{field_name}_must_be_positive_integer")
    return value


def _authority_sources(raw: Any) -> tuple[AuthoritySource, ...]:
    values = _strings(raw) or (AuthoritySource.DELEGATED_CARD.value,)
    result: list[AuthoritySource] = []
    for value in values:
        try:
            source = AuthoritySource(value)
        except ValueError as exc:
            raise ResidentResourceDescriptorError(
                f"unknown_resource_authority_source:{value}"
            ) from exc
        if source not in result:
            result.append(source)
    return tuple(result)


def parse_resource_family_ceiling(raw: Mapping[str, Any]) -> ResourceFamilyCeiling:
    family_id = _clean(raw.get("id") or raw.get("family"))
    if not family_id:
        raise ResidentResourceDescriptorError("resource_family_id_required")
    resource_kinds = _strings(raw.get("resource_kinds") or raw.get("kinds"))
    if not resource_kinds:
        raise ResidentResourceDescriptorError(f"resource_family_kinds_required:{family_id}")
    transports = tuple(
        value.lower().replace("_", "-")
        for value in _strings(raw.get("transports"))
    )
    if not transports:
        raise ResidentResourceDescriptorError(
            f"resource_family_transports_required:{family_id}"
        )
    resource_patterns = _strings(raw.get("resource_patterns"))
    if not resource_patterns:
        raise ResidentResourceDescriptorError(
            f"resource_family_patterns_required:{family_id}"
        )
    return ResourceFamilyCeiling(
        family_id=family_id,
        resource_kinds=resource_kinds,
        authority_sources=_authority_sources(raw.get("authority_sources")),
        transports=transports,
        resource_patterns=resource_patterns,
        allowed_tool_patterns=_strings(raw.get("allowed_tools")) or ("*",),
        endpoint_schemes=tuple(value.lower() for value in _strings(raw.get("endpoint_schemes"))),
        endpoint_hosts=tuple(value.lower() for value in _strings(raw.get("endpoint_hosts"))),
        max_resources=_positive_int(
            raw.get("max_resources"),
            field_name=f"resource_family_{family_id}_max_resources",
            default=8,
        ),
        max_tools_per_resource=_positive_int(
            raw.get("max_tools_per_resource"),
            field_name=f"resource_family_{family_id}_max_tools_per_resource",
            default=64,
        ),
    )


def resident_agent_ceiling_from_bundle_props(
    bundle_props: Mapping[str, Any] | None,
    *,
    tenant: str,
    project: str,
    application: str,
    agent_id: str,
    declared_resource_ids: tuple[str, ...] = (),
    descriptor_revision: str = "",
) -> ResidentAgentCeiling:
    """Read the agent's dynamic family ceilings from effective bundle props.

    ``delegated_resource_families`` describes classes of user-owned resources;
    exact connector ids remain in Connection Hub and never enter a descriptor.
    """

    families = resource_family_ceilings_from_bundle_props(bundle_props, agent_id=agent_id)
    return ResidentAgentCeiling(
        tenant=_clean(tenant),
        project=_clean(project),
        application=_clean(application),
        agent_id=_clean(agent_id),
        declared_resource_ids=tuple(
            sorted({_clean(value) for value in declared_resource_ids if _clean(value)})
        ),
        resource_families=families,
        descriptor_revision=_clean(descriptor_revision),
    )


def resource_family_ceilings_from_bundle_props(
    bundle_props: Mapping[str, Any] | None,
    *,
    agent_id: str,
) -> tuple[ResourceFamilyCeiling, ...]:
    """Parse only the family rows for catalog and runtime adapters."""

    block = _agent_block(bundle_props or {}, agent_id)
    raw_families = block.get("delegated_resource_families")
    if raw_families is None:
        raw_families = []
    if not isinstance(raw_families, list):
        raise ResidentResourceDescriptorError(
            "delegated_resource_families_must_be_list"
        )
    if any(not isinstance(item, Mapping) for item in raw_families):
        raise ResidentResourceDescriptorError(
            "delegated_resource_family_must_be_mapping"
        )
    families = tuple(parse_resource_family_ceiling(item) for item in raw_families)
    ids = [family.family_id for family in families]
    if len(ids) != len(set(ids)):
        raise ResidentResourceDescriptorError("duplicate_resource_family_id")
    return families


def resource_family_catalog_from_bundle_props(
    bundle_props: Mapping[str, Any] | None,
    *,
    agent_id: str,
) -> list[dict[str, Any]]:
    """Credential-free wire projection of dynamic family ceilings."""

    return [
        {
            "id": family.family_id,
            "resource_kinds": list(family.resource_kinds),
            "authority_sources": [source.value for source in family.authority_sources],
            "transports": list(family.transports),
            "resource_patterns": list(family.resource_patterns),
            "allowed_tools": list(family.allowed_tool_patterns),
            "endpoint_schemes": list(family.endpoint_schemes),
            "endpoint_hosts": list(family.endpoint_hosts),
            "max_resources": family.max_resources,
            "max_tools_per_resource": family.max_tools_per_resource,
        }
        for family in resource_family_ceilings_from_bundle_props(
            bundle_props,
            agent_id=agent_id,
        )
    ]


__all__ = [
    "ResidentResourceDescriptorError",
    "parse_resource_family_ceiling",
    "resource_family_catalog_from_bundle_props",
    "resource_family_ceilings_from_bundle_props",
    "resident_agent_ceiling_from_bundle_props",
]
