# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube portable-context binding for Prokura connector-app selection."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import (
    get_current_bundle_call_context,
    update_current_bundle_call_context,
)


_core = import_module("prokura.connector_app_resolution")


def set_service_connector_apps(mapping: Mapping[str, str] | None) -> None:
    _core.set_service_connector_apps(
        mapping,
        context_updater=update_current_bundle_call_context,
    )


def resolve_connector_app_id(provider_id: str) -> str:
    return _core.resolve_connector_app_id(
        provider_id,
        context_reader=get_current_bundle_call_context,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = ["resolve_connector_app_id", "set_service_connector_apps"]
