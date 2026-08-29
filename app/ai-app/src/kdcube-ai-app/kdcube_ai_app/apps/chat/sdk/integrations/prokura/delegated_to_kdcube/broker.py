# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube account-store binding for the Prokura credential broker."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_to_kdcube.store import (
    DelegatedToKdcubeStore,
)


_core = import_module("prokura.delegated_to_kdcube.broker")

DelegatedToKdcubeBroker = _core.DelegatedToKdcubeBroker


def broker_for_user(
    *,
    user_id: str,
    config: Any,
    bundle_id: str = "",
    store: Any | None = None,
    client_secret_resolver: Callable[..., Any] | None = None,
) -> Any:
    return _core.broker_for_user(
        user_id=user_id,
        config=config,
        bundle_id=bundle_id,
        store=store,
        store_factory=DelegatedToKdcubeStore,
        client_secret_resolver=client_secret_resolver,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = ["DelegatedToKdcubeBroker", "broker_for_user"]
