# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube session, storage, and secret bindings for Prokura federated tokens."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable, Mapping

from kdcube_ai_app.auth.sessions import RequestContext, UserType
from kdcube_ai_app.infra.namespaces import ns_key


_core = import_module("prokura.federated_tokens.data_bus")

FEDERATED_TOKEN_DEFAULT_TTL_SECONDS = _core.FEDERATED_TOKEN_DEFAULT_TTL_SECONDS
FEDERATED_TOKEN_MAX_TTL_SECONDS = _core.FEDERATED_TOKEN_MAX_TTL_SECONDS
FEDERATED_TOKEN_SCHEMA = _core.FEDERATED_TOKEN_SCHEMA
FEDERATED_TOKEN_SECRET_KEY = _core.FEDERATED_TOKEN_SECRET_KEY
FederatedTokenError = _core.FederatedTokenError
FederatedTokenExpired = _core.FederatedTokenExpired
FederatedTokenGrant = _core.FederatedTokenGrant
FederatedTokenInvalid = _core.FederatedTokenInvalid
FederatedTokenVerification = _core.FederatedTokenVerification


def _request_context_from_request(request: Any) -> RequestContext:
    gateway_adapter = getattr(getattr(request, "app", None), "state", None)
    gateway_adapter = getattr(gateway_adapter, "gateway_adapter", None)
    extractor = getattr(gateway_adapter, "_extract_context", None)
    if callable(extractor):
        try:
            return extractor(request)
        except Exception:
            pass

    headers = getattr(request, "headers", {}) or {}
    client = getattr(request, "client", None)
    client_ip = getattr(client, "host", None) or "unknown"
    return RequestContext(
        client_ip=client_ip,
        user_agent=headers.get("user-agent", "") if hasattr(headers, "get") else "",
    )


def _session_manager_from_request(request: Any) -> Any:
    state = getattr(getattr(request, "app", None), "state", None)
    gateway_adapter = getattr(state, "gateway_adapter", None)
    session_manager = getattr(
        getattr(gateway_adapter, "gateway", None),
        "session_manager",
        None,
    )
    if session_manager is None:
        raise FederatedTokenInvalid("session manager is unavailable")
    return session_manager


def _redis_from_request_or_session_manager(
    request: Any,
    session_manager: Any,
) -> Any:
    state = getattr(getattr(request, "app", None), "state", None)
    redis = getattr(state, "redis_async", None)
    if redis is not None:
        return redis
    return getattr(session_manager, "redis", None)


async def _secret_loader(key: str) -> str | bytes | None:
    from kdcube_ai_app.apps.chat.sdk.config import get_secret

    return await get_secret(key, default=None)


async def issue_federated_data_bus_token(
    *,
    request: Any,
    tenant: str,
    project: str,
    bundle_id: str,
    user_id: str,
    user_type: str | UserType = UserType.EXTERNAL,
    username: str | None = None,
    email: str | None = None,
    roles: Iterable[str] | None = None,
    permissions: Iterable[str] | None = None,
    identity_authority: Mapping[str, Any] | None = None,
    ttl_seconds: int = FEDERATED_TOKEN_DEFAULT_TTL_SECONDS,
    secret: str | bytes | None = None,
) -> Any:
    user_type_value = (
        user_type.value
        if isinstance(user_type, UserType)
        else str(user_type or "").strip().lower()
    )
    resolved_user_type = UserType(user_type_value or UserType.EXTERNAL.value)
    session_manager = _session_manager_from_request(request)
    return await _core.issue_federated_data_bus_token(
        session_manager=session_manager,
        request_context=_request_context_from_request(request),
        redis=_redis_from_request_or_session_manager(request, session_manager),
        tenant=tenant,
        project=project,
        bundle_id=bundle_id,
        user_id=user_id,
        user_type=resolved_user_type,
        username=username,
        email=email,
        roles=roles,
        permissions=permissions,
        identity_authority=identity_authority,
        ttl_seconds=ttl_seconds,
        secret=secret,
        secret_loader=_secret_loader,
        token_key_builder=ns_key,
    )


async def verify_federated_data_bus_token(
    *,
    token: str,
    tenant: str,
    project: str,
    bundle_id: str,
    redis: Any,
    session_manager: Any,
    secret: str | bytes | None = None,
    now: int | None = None,
) -> Any:
    return await _core.verify_federated_data_bus_token(
        token=token,
        tenant=tenant,
        project=project,
        bundle_id=bundle_id,
        redis=redis,
        session_manager=session_manager,
        secret=secret,
        now=now,
        secret_loader=_secret_loader,
        token_key_builder=ns_key,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = [
    "FEDERATED_TOKEN_DEFAULT_TTL_SECONDS",
    "FEDERATED_TOKEN_MAX_TTL_SECONDS",
    "FEDERATED_TOKEN_SCHEMA",
    "FEDERATED_TOKEN_SECRET_KEY",
    "FederatedTokenError",
    "FederatedTokenExpired",
    "FederatedTokenGrant",
    "FederatedTokenInvalid",
    "FederatedTokenVerification",
    "issue_federated_data_bus_token",
    "verify_federated_data_bus_token",
]
