# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube lock adapter for Prokura's delegated-card mutation service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from prokura.delegated_credentials.cards.service import (
    CARD_LOCK_WAIT_SECONDS,
    CardCommitFailed,
    CardConflict,
    CardMutationLock,
    CardMutationLockTimeout,
    CardServingUnavailable,
    DelegatedCardService as _ProkuraDelegatedCardService,
    replace_state,
)
from kdcube_ai_app.storage.observed_file_locks import (
    ObservedFileLockTimeout,
    observed_file_lock_async,
)


@asynccontextmanager
async def _kdcube_card_mutation_lock(**kwargs: Any) -> AsyncIterator[Any]:
    try:
        async with observed_file_lock_async(**kwargs) as metadata:
            yield metadata
    except ObservedFileLockTimeout as exc:
        raise CardMutationLockTimeout(str(exc)) from exc


class DelegatedCardService(_ProkuraDelegatedCardService):
    """Prokura's state machine bound to KDCube's shared-storage lock."""

    def __init__(
        self,
        *,
        store: Any,
        cache: Any,
        settings: Any = None,
        mutation_lock: CardMutationLock | None = None,
    ) -> None:
        super().__init__(
            store=store,
            cache=cache,
            settings=settings,
            mutation_lock=mutation_lock or _kdcube_card_mutation_lock,
        )


__all__ = [
    "CARD_LOCK_WAIT_SECONDS",
    "CardCommitFailed",
    "CardConflict",
    "CardMutationLock",
    "CardMutationLockTimeout",
    "CardServingUnavailable",
    "DelegatedCardService",
    "replace_state",
]
