# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube Connection Hub bindings plus host-neutral Prokura hub contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AuthenticatorStore": "prokura.hub.authenticator_store",
    "authenticate_request": f"{__name__}.authenticators",
    "descriptor_authenticator_rows": f"{__name__}.authenticators",
    "merged_authenticator_rows": f"{__name__}.authenticators",
    "matching_authenticator_rows": f"{__name__}.authenticators",
    "supported_authenticator_providers": f"{__name__}.authenticators",
    "ConnectionEdgeStore": "prokura.hub.edges",
    "resolve_principal_roles": "prokura.hub.edges",
    "BUNDLE_ID": f"{__name__}.provider_impl",
    "ConnectionHubProvider": f"{__name__}.provider_impl",
    "DEFAULT_DELEGATED_IDENTITY_SCOPE": "prokura.hub.resolver",
    "IDENTITY_SCOPE_GRANTOR": "prokura.hub.resolver",
    "IDENTITY_SCOPE_GRANTOR_FAMILY": "prokura.hub.resolver",
    "IDENTITY_SCOPE_SELECTED_IDENTITIES": "prokura.hub.resolver",
    "actor_user_id_for_identity": "prokura.hub.resolver",
    "delegated_primary_user_id": "prokura.hub.resolver",
    "normalize_delegated_identity_scope": "prokura.hub.resolver",
    "parse_actor_user_id": "prokura.hub.resolver",
    "resolve_delegated_authority_projection": "prokura.hub.resolver",
    "resolve_delegated_identity_scope": "prokura.hub.resolver",
    "resolve_identity_family": "prokura.hub.resolver",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
