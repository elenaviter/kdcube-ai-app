# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Per-turn composition of current candidates and one resident Card."""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from connection_hub.delegated_credentials.cards.identity import resident_client_id
from connection_hub.delegated_credentials.cards.read_model import DelegatedCardView

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.card_adapter import (
    delegated_card_snapshot_from_view,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    AuthoritySource,
    AvailabilityReason,
    ResidentAgentCeiling,
    ResidentResourceCandidate,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.service import (
    CurrentResidentResourceFacts,
)


class ResidentResourceCandidateLoader(Protocol):
    """Load current provider facts; Gateway supplies the production adapter."""

    async def load_candidates(
        self,
        *,
        ceiling: ResidentAgentCeiling,
        grantor_subject: str,
    ) -> tuple[ResidentResourceCandidate, ...]: ...


class ResidentCardViewReader(Protocol):
    """The Card service method Projection consumes."""

    async def resident_profile_card(
        self,
        *,
        grantor_subject: str,
        client_id: str,
    ) -> DelegatedCardView | None: ...


def _identity_scope(value: Any) -> str:
    return str(value or "").strip() or "grantor"


class CardBackedResidentResourceFactsLoader:
    """Load fresh candidates and the resident profile's one Card per turn.

    A Card read failure is projected onto every delegated-card candidate scope
    instead of raising a process-wide failure. Provider candidate loading
    remains the candidate adapter's responsibility and is never silently
    replaced with an empty inventory.
    """

    def __init__(
        self,
        *,
        candidate_loader: ResidentResourceCandidateLoader,
        card_reader: ResidentCardViewReader,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._candidates = candidate_loader
        self._card = card_reader
        self._clock = clock

    async def load_current(
        self,
        *,
        ceiling: ResidentAgentCeiling,
        grantor_subject: str,
    ) -> CurrentResidentResourceFacts:
        candidates = tuple(
            await self._candidates.load_candidates(
                ceiling=ceiling,
                grantor_subject=grantor_subject,
            )
        )
        scopes = sorted(
            {
                _identity_scope(candidate.identity_scope)
                for candidate in candidates
                if candidate.authority_source is AuthoritySource.DELEGATED_CARD
            }
        )
        client_id = resident_client_id(ceiling.application, ceiling.agent_id)
        card = None
        unavailable: dict[str, AvailabilityReason] = {}
        if scopes:
            try:
                view = await self._card.resident_profile_card(
                    grantor_subject=grantor_subject,
                    client_id=client_id,
                )
                if view is not None:
                    card = delegated_card_snapshot_from_view(
                        view,
                        tenant=ceiling.tenant,
                        project=ceiling.project,
                    )
            except Exception:
                unavailable = {
                    scope: AvailabilityReason.CARD_AUTHORITY_UNAVAILABLE
                    for scope in scopes
                }
        return CurrentResidentResourceFacts(
            candidates=candidates,
            card=card,
            observed_at_epoch=int(self._clock()),
            card_unavailable_reasons=unavailable,
        )


__all__ = [
    "CardBackedResidentResourceFactsLoader",
    "ResidentCardViewReader",
    "ResidentResourceCandidateLoader",
]
