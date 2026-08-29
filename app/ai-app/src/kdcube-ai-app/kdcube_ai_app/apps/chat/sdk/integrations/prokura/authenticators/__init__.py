# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube-bound request-authenticator client plus Prokura contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ConnectionHubAuthenticatorsClient": f"{__name__}.client",
    "DEFAULT_CONNECTION_HUB_BUNDLE_ID": f"{__name__}.client",
    "REQUEST_AUTHENTICATE_OPERATION": f"{__name__}.client",
    "AuthRequestHints": "prokura.authenticators.authority",
    "AuthorityIdentity": "prokura.authenticators.authority",
    "SurfaceGuardRequirement": "prokura.authenticators.authority",
    "select_authenticator_candidates": "prokura.authenticators.authority",
    "AuthenticatedRequest": "prokura.authenticators.models",
    "AuthenticatorRegistration": "prokura.authenticators.models",
    "RequestEnvelope": "prokura.authenticators.models",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
