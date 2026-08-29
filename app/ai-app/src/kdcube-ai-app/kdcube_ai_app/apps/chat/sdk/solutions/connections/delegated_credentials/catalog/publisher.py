# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube shared-operation adapter for Prokura catalog publication."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

from prokura.delegated_credentials.catalog.publisher import (
    CATALOG_OPERATION,
    SIGNATURE_FILENAME,
    CatalogPublicationError,
    CatalogPublicationResult,
    SharedStorageOperationRunner,
    ensure_delegated_catalog as _ensure_delegated_catalog,
)
from kdcube_ai_app.infra.plugin.bundle_once import run_once_for_shared_bundle_storage


async def ensure_delegated_catalog(
    *,
    connections: Mapping[str, Any],
    store: Any,
    cache: Any,
    reread: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
    reason: str = "",
    settings: Any = None,
    logger: Any = None,
    operation_runner: SharedStorageOperationRunner | None = None,
) -> CatalogPublicationResult:
    return await _ensure_delegated_catalog(
        connections=connections,
        store=store,
        cache=cache,
        operation_runner=operation_runner or run_once_for_shared_bundle_storage,
        reread=reread,
        reason=reason,
        settings=settings,
        logger=logger,
    )


__all__ = [
    "CATALOG_OPERATION",
    "CatalogPublicationError",
    "CatalogPublicationResult",
    "SIGNATURE_FILENAME",
    "SharedStorageOperationRunner",
    "ensure_delegated_catalog",
]
