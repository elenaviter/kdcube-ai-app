# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube composition adapter for Prokura's durable card persistence."""

from __future__ import annotations

from typing import Any

from prokura.delegated_credentials.cards.persistence import (
    CardPersistence,
    DurableCardPersistence as _ProkuraDurableCardPersistence,
    LoadedCard,
)
from prokura.delegated_credentials.cards.service import CardMutationLock
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.cards.service import (
    _kdcube_card_mutation_lock,
)


class DurableCardPersistence(_ProkuraDurableCardPersistence):
    """Prokura card persistence bound to KDCube's shared-storage lock."""

    def __init__(
        self,
        *,
        redis: Any,
        tenant: str,
        project: str,
        card_store: Any,
        settings: Any = None,
        mutation_lock: CardMutationLock | None = None,
    ) -> None:
        super().__init__(
            redis=redis,
            tenant=tenant,
            project=project,
            card_store=card_store,
            settings=settings,
            mutation_lock=mutation_lock or _kdcube_card_mutation_lock,
        )


__all__ = ["CardPersistence", "DurableCardPersistence", "LoadedCard"]
