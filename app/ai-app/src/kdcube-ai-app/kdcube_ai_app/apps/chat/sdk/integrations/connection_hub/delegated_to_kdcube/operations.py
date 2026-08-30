# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube account-store binding for Connection Hub connection operations."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_to_kdcube.store import (
    DelegatedToKdcubeStore,
)


_core = import_module("connection_hub.delegated_to_kdcube.operations")

DelegatedToKdcubeOperations = _core.DelegatedToKdcubeOperations


def operations_for_user(
    *,
    user_id: str,
    config: Any,
    bundle_id: str = _core.CONNECTION_HUB_BUNDLE_ID,
    store: Any | None = None,
    consent_granted_notifier: Callable[..., Any] | None = None,
) -> Any:
    return _core.operations_for_user(
        user_id=user_id,
        config=config,
        bundle_id=bundle_id,
        store=store,
        store_factory=DelegatedToKdcubeStore,
        consent_granted_notifier=consent_granted_notifier,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = ["DelegatedToKdcubeOperations", "operations_for_user"]
