# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube host composition surface for delegated credential services."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AutomationAccessRecord": "kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.automation_access",
    "AutomationAccessService": "kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.automation_access",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
