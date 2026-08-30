# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube descriptor adapter for Connection Hub's delegated OAuth configuration."""

from __future__ import annotations

from typing import Any

from connection_hub.delegated_credentials.oauth.config import (
    DEFAULT_CLAUDE_REDIRECT_URIS,
    DEFAULT_DCR_REDIRECT_URIS,
    OAuthDelegatedAccountRequirement,
    OAuthDelegatedCapabilityConfig,
    OAuthDelegatedClientConfig,
    OAuthDelegatedClientMetadataDocumentsConfig,
    OAuthDelegatedConsentUIConfig,
    OAuthDelegatedDynamicClientRegistrationConfig,
    OAuthDelegatedPublicClientConfig,
    OAuthDelegatedResourceConfig,
    OAuthDelegatedToolConfig,
    oauth_delegated_config as _portable_oauth_delegated_config,
    oauth_delegated_config_from_connections,
)
from kdcube_ai_app.apps.chat.sdk.config import get_settings


def oauth_delegated_config(source: Any | None = None) -> OAuthDelegatedClientConfig:
    """Resolve app/request overrides with KDCube descriptor defaults."""

    request_state = getattr(source, "state", None) if source is not None else None
    if request_state is not None and hasattr(request_state, "oauth_delegated_config"):
        return _portable_oauth_delegated_config(source)
    app = (getattr(source, "app", None) or source) if source is not None else None
    app_state = getattr(app, "state", None)
    if app_state is not None and hasattr(app_state, "oauth_delegated_config"):
        return _portable_oauth_delegated_config(source)
    return _portable_oauth_delegated_config(source, settings=get_settings())


__all__ = [
    "DEFAULT_CLAUDE_REDIRECT_URIS",
    "DEFAULT_DCR_REDIRECT_URIS",
    "OAuthDelegatedAccountRequirement",
    "OAuthDelegatedCapabilityConfig",
    "OAuthDelegatedClientConfig",
    "OAuthDelegatedClientMetadataDocumentsConfig",
    "OAuthDelegatedConsentUIConfig",
    "OAuthDelegatedDynamicClientRegistrationConfig",
    "OAuthDelegatedPublicClientConfig",
    "OAuthDelegatedResourceConfig",
    "OAuthDelegatedToolConfig",
    "oauth_delegated_config",
    "oauth_delegated_config_from_connections",
]
