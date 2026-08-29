# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Lazy compatibility surface for request-authenticator contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BASE = "kdcube_ai_app.apps.chat.sdk.solutions.connections.authenticators"
_EXPORTS = {
    "ConnectionHubAuthenticatorsClient": f"{_BASE}.client",
    "DEFAULT_CONNECTION_HUB_BUNDLE_ID": f"{_BASE}.client",
    "REQUEST_AUTHENTICATE_OPERATION": f"{_BASE}.client",
    "AuthRequestHints": f"{_BASE}.authority",
    "AuthorityIdentity": f"{_BASE}.authority",
    "SurfaceGuardRequirement": f"{_BASE}.authority",
    "select_authenticator_candidates": f"{_BASE}.authority",
    "AuthenticatedRequest": f"{_BASE}.models",
    "AuthenticatorRegistration": f"{_BASE}.models",
    "RequestEnvelope": f"{_BASE}.models",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
