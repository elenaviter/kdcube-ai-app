# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Lazy compatibility surface for delegated OAuth."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BASE = "kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.oauth"
_EXPORTS = {
    "DELEGATED_CLIENT_AUDIENCE": f"{_BASE}.authority",
    "DELEGATED_CLIENT_CREDENTIAL_KIND": f"{_BASE}.authority",
    "OAuthDelegatedClientAuthorityProvider": f"{_BASE}.authority",
    "build_delegated_client_credential": f"{_BASE}.authority",
    "delegated_client_authority_spec": f"{_BASE}.authority",
    "register_delegated_client_authority": f"{_BASE}.authority",
    "OAuthDelegatedClientConfig": f"{_BASE}.config",
    "OAuthDelegatedConsentUIConfig": f"{_BASE}.config",
    "oauth_delegated_config": f"{_BASE}.config",
    "ACCESS_TOKEN_TTL_SECONDS": f"{_BASE}.grants",
    "DELEGATED_CLIENT_ROLE": f"{_BASE}.grants",
    "integration_subject": f"{_BASE}.grants",
    "mint_delegated_client_access_token": f"{_BASE}.grants",
    "authorization_server_metadata": f"{_BASE}.metadata",
    "protected_resource_metadata": f"{_BASE}.metadata",
    "protected_resource_metadata_url": f"{_BASE}.metadata",
    "GrantStore": f"{_BASE}.store",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
