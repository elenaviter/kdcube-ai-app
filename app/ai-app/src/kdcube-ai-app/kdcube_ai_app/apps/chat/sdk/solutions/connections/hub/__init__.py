# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Lazy compatibility surface for Connection Hub runtime services."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BASE = "kdcube_ai_app.apps.chat.sdk.solutions.connections.hub"
_EXPORTS = {
    "AuthenticatorStore": f"{_BASE}.authenticator_store",
    "authenticate_request": f"{_BASE}.authenticators",
    "descriptor_authenticator_rows": f"{_BASE}.authenticators",
    "merged_authenticator_rows": f"{_BASE}.authenticators",
    "matching_authenticator_rows": f"{_BASE}.authenticators",
    "supported_authenticator_providers": f"{_BASE}.authenticators",
    "ConnectionEdgeStore": f"{_BASE}.edges",
    "resolve_principal_roles": f"{_BASE}.edges",
    "BUNDLE_ID": f"{_BASE}.provider_impl",
    "ConnectionHubProvider": f"{_BASE}.provider_impl",
    "DEFAULT_DELEGATED_IDENTITY_SCOPE": f"{_BASE}.resolver",
    "IDENTITY_SCOPE_GRANTOR": f"{_BASE}.resolver",
    "IDENTITY_SCOPE_GRANTOR_FAMILY": f"{_BASE}.resolver",
    "IDENTITY_SCOPE_SELECTED_IDENTITIES": f"{_BASE}.resolver",
    "actor_user_id_for_identity": f"{_BASE}.resolver",
    "delegated_primary_user_id": f"{_BASE}.resolver",
    "normalize_delegated_identity_scope": f"{_BASE}.resolver",
    "parse_actor_user_id": f"{_BASE}.resolver",
    "resolve_delegated_authority_projection": f"{_BASE}.resolver",
    "resolve_delegated_identity_scope": f"{_BASE}.resolver",
    "resolve_identity_family": f"{_BASE}.resolver",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
