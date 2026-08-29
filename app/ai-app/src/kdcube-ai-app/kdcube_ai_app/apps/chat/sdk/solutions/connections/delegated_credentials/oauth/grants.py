# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube session-authority adapter for Prokura delegated OAuth grants."""

from __future__ import annotations

from typing import Any, List, Mapping

from prokura.delegated_credentials.oauth.grants import (
    ACCESS_TOKEN_TTL_SECONDS,
    DELEGATED_CLIENT_ROLE,
    SessionAuthorityFactory,
    integration_subject,
    mint_delegated_client_access_token as _prokura_mint,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.oauth.config import (
    oauth_delegated_config,
)
from kdcube_ai_app.auth.bundle import get_bundle_session_authority


def oauth_tenant_project(source: Any | None = None) -> tuple[str, str]:
    config = oauth_delegated_config(source)
    return config.tenant, config.project


async def mint_delegated_client_access_token(
    sub: str,
    scopes: List[str],
    *,
    authority: Any = None,
    authority_factory: SessionAuthorityFactory | None = None,
    client_id: str = "",
    operations: List[str] | None = None,
    credential: Mapping[str, Any] | None = None,
    ttl_seconds: int = ACCESS_TOKEN_TTL_SECONDS,
) -> dict:
    resolved_factory = authority_factory
    config = None
    if authority is None:
        resolved_factory = resolved_factory or get_bundle_session_authority
        config = oauth_delegated_config()
    return await _prokura_mint(
        sub,
        scopes,
        authority=authority,
        authority_factory=resolved_factory,
        config=config,
        client_id=client_id,
        operations=operations,
        credential=credential,
        ttl_seconds=ttl_seconds,
    )


__all__ = [
    "ACCESS_TOKEN_TTL_SECONDS",
    "DELEGATED_CLIENT_ROLE",
    "SessionAuthorityFactory",
    "integration_subject",
    "mint_delegated_client_access_token",
    "oauth_tenant_project",
]
