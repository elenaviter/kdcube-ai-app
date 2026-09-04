# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Compose coherent Card and Gateway observations into Projection facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from connection_hub.delegated_gateway import GatewayTool

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.gateway_adapter import (
    gateway_resident_projection,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AvailabilityReason,
    DelegatedCardSnapshot,
    ResidentResourceCandidate,
    ResourceBinding,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.service import (
    CurrentResidentResourceFacts,
)


class GatewayObservationError(ValueError):
    """A current observation cannot identify one unambiguous resident Card."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ResidentGatewayObservation:
    """Card and Gateway facts read for one access id in the same host pass."""

    card: DelegatedCardSnapshot
    tools: tuple[GatewayTool, ...]
    access: Mapping[str, Any]
    binding: ResourceBinding


def compose_gateway_resource_facts(
    *,
    application_candidates: Iterable[ResidentResourceCandidate] = (),
    observation: ResidentGatewayObservation | None = None,
    observed_at_epoch: int,
    card_unavailable_reasons: Mapping[
        str, AvailabilityReason
    ] | None = None,
) -> CurrentResidentResourceFacts:
    """Join one current Gateway observation without another Card read."""

    try:
        observed_at = int(observed_at_epoch)
    except (TypeError, ValueError):
        raise GatewayObservationError("gateway_observation_time_invalid") from None
    if observed_at < 0:
        raise GatewayObservationError("gateway_observation_time_invalid")

    candidates = list(application_candidates)
    meta: dict[str, tuple] = {}
    requestable: dict[str, tuple] = {}
    discovery: dict[str, str] = {}
    card = None
    if observation is not None:
        if not isinstance(observation, ResidentGatewayObservation):
            raise GatewayObservationError("gateway_observation_invalid")
        access_id = str(observation.card.access_id or "").strip()
        if not access_id:
            raise GatewayObservationError("gateway_observation_access_invalid")
        projection = gateway_resident_projection(
            card=observation.card,
            tools=observation.tools,
            access=observation.access,
            binding=observation.binding,
        )
        card = observation.card
        candidates.extend(projection.candidates)
        meta[access_id] = projection.meta_tools
        requestable[access_id] = projection.requestable_resources
        discovery[access_id] = projection.requestable_discovery

    return CurrentResidentResourceFacts(
        candidates=tuple(candidates),
        card=card,
        observed_at_epoch=observed_at,
        card_unavailable_reasons=dict(card_unavailable_reasons or {}),
        gateway_meta_tools_by_access_id=meta,
        gateway_requestable_resources_by_access_id=requestable,
        gateway_requestable_discovery_by_access_id=discovery,
    )


__all__ = [
    "GatewayObservationError",
    "ResidentGatewayObservation",
    "compose_gateway_resource_facts",
]
