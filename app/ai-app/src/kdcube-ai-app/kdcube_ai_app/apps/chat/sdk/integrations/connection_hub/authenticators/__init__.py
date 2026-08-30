# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube-bound request-authenticator client plus Connection Hub contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ConnectionHubAuthenticatorsClient": f"{__name__}.client",
    "DEFAULT_CONNECTION_HUB_BUNDLE_ID": f"{__name__}.client",
    "REQUEST_AUTHENTICATE_OPERATION": f"{__name__}.client",
    "AuthRequestHints": "connection_hub.authenticators.authority",
    "AuthorityIdentity": "connection_hub.authenticators.authority",
    "SurfaceGuardRequirement": "connection_hub.authenticators.authority",
    "select_authenticator_candidates": "connection_hub.authenticators.authority",
    "AuthenticatedRequest": "connection_hub.authenticators.models",
    "AuthenticatorRegistration": "connection_hub.authenticators.models",
    "RequestEnvelope": "connection_hub.authenticators.models",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
