# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube Connection Hub bindings plus host-neutral Connection Hub hub contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AuthenticatorStore": "connection_hub.hub.authenticator_store",
    "authenticate_request": f"{__name__}.authenticators",
    "descriptor_authenticator_rows": f"{__name__}.authenticators",
    "merged_authenticator_rows": f"{__name__}.authenticators",
    "matching_authenticator_rows": f"{__name__}.authenticators",
    "supported_authenticator_providers": f"{__name__}.authenticators",
    "ConnectionEdgeStore": "connection_hub.hub.edges",
    "resolve_principal_roles": "connection_hub.hub.edges",
    "BUNDLE_ID": f"{__name__}.provider_impl",
    "ConnectionHubProvider": f"{__name__}.provider_impl",
    "DEFAULT_DELEGATED_IDENTITY_SCOPE": "connection_hub.hub.resolver",
    "IDENTITY_SCOPE_GRANTOR": "connection_hub.hub.resolver",
    "IDENTITY_SCOPE_GRANTOR_FAMILY": "connection_hub.hub.resolver",
    "IDENTITY_SCOPE_SELECTED_IDENTITIES": "connection_hub.hub.resolver",
    "actor_user_id_for_identity": "connection_hub.hub.resolver",
    "delegated_primary_user_id": "connection_hub.hub.resolver",
    "normalize_delegated_identity_scope": "connection_hub.hub.resolver",
    "parse_actor_user_id": "connection_hub.hub.resolver",
    "resolve_delegated_authority_projection": "connection_hub.hub.resolver",
    "resolve_delegated_identity_scope": "connection_hub.hub.resolver",
    "resolve_identity_family": "connection_hub.hub.resolver",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
