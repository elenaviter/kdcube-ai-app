# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Distributed CSRF tokens for cookie-authenticated bundle operations."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import Request

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.auth.sessions import UserSession


OPERATION_CSRF_HEADER = "X-KDCube-CSRF-Token"
OPERATION_CSRF_TTL_SECONDS = 600
_OPERATION_CSRF_SCHEMA = "kdcube.bundle_operation_csrf.v1"
_ATOMIC_GETDEL = """
local value = redis.call('GET', KEYS[1])
if not value then
    return nil
end
redis.call('DEL', KEYS[1])
return value
"""


@dataclass(frozen=True)
class OperationCsrfValidation:
    ok: bool
    reason: str


def authenticated_session_subject(session: UserSession) -> str:
    """Return the stable platform subject used to bind one CSRF token."""

    for value in (getattr(session, "user_id", None), getattr(session, "username", None)):
        text = str(value or "").strip()
        if text:
            return text
    authority = getattr(session, "identity_authority", None)
    if isinstance(authority, Mapping):
        for key in ("subject", "sub", "user_id"):
            text = str(authority.get(key) or "").strip()
            if text:
                return text
    return ""


def request_uses_cookie_auth(request: Request) -> bool:
    """True when browser cookies, rather than explicit headers, carry auth."""

    auth = get_settings().AUTH
    explicit = bool(
        str(request.headers.get("authorization") or "").strip()
        or str(
            request.headers.get(auth.ID_TOKEN_HEADER_NAME)
            or request.headers.get(auth.ID_TOKEN_HEADER_NAME.lower())
            or ""
        ).strip()
    )
    if explicit:
        return False
    return any(
        bool(request.cookies.get(name))
        for name in (
            auth.AUTH_TOKEN_COOKIE_NAME,
            auth.ID_TOKEN_COOKIE_NAME,
            auth.MASQUERADED_TOKEN_COOKIE_NAME,
        )
        if name
    )


def _key(*, tenant: str, project: str, token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return (
        f"{str(tenant or '').strip()}:{str(project or '').strip()}:"
        f"kdcube:bundle-operation-csrf:{digest}"
    )


def _context(
    *,
    subject: str,
    tenant: str,
    project: str,
    bundle_id: str,
    operation: str,
    method: str,
) -> dict[str, str]:
    return {
        "schema": _OPERATION_CSRF_SCHEMA,
        "subject": str(subject or "").strip(),
        "tenant": str(tenant or "").strip(),
        "project": str(project or "").strip(),
        "bundle_id": str(bundle_id or "").strip(),
        "operation": str(operation or "").strip(),
        "method": str(method or "POST").strip().upper(),
    }


def _request_state_store(request: Request) -> Any:
    """Resolve shared state below the HTTP dispatch boundary."""

    state_store = getattr(getattr(request, "state", None), "operation_csrf_store", None)
    if state_store is not None:
        return state_store
    state_store = getattr(getattr(request.app, "state", None), "redis_async", None)
    if state_store is not None:
        return state_store

    from kdcube_ai_app.infra.redis.client import get_async_redis_client

    return get_async_redis_client(get_settings().REDIS_URL)


async def mint_operation_csrf_token(
    redis: Any,
    *,
    subject: str,
    tenant: str,
    project: str,
    bundle_id: str,
    operation: str,
    method: str = "POST",
) -> str:
    if not str(subject or "").strip():
        raise ValueError("authenticated subject is required")
    token = secrets.token_urlsafe(32)
    payload = _context(
        subject=subject,
        tenant=tenant,
        project=project,
        bundle_id=bundle_id,
        operation=operation,
        method=method,
    )
    await redis.setex(
        _key(tenant=tenant, project=project, token=token),
        OPERATION_CSRF_TTL_SECONDS,
        json.dumps(payload, separators=(",", ":")),
    )
    return token


async def consume_operation_csrf_token(
    redis: Any,
    token: str,
    *,
    subject: str,
    tenant: str,
    project: str,
    bundle_id: str,
    operation: str,
    method: str = "POST",
) -> OperationCsrfValidation:
    token_value = str(token or "").strip()
    if not token_value:
        return OperationCsrfValidation(False, "missing")
    try:
        raw = await redis.eval(
            _ATOMIC_GETDEL,
            1,
            _key(tenant=tenant, project=project, token=token_value),
        )
    except Exception:
        return OperationCsrfValidation(False, "store_unavailable")
    if raw is None:
        return OperationCsrfValidation(False, "not_found")
    try:
        payload = json.loads(raw)
    except Exception:
        return OperationCsrfValidation(False, "malformed")
    if not isinstance(payload, Mapping):
        return OperationCsrfValidation(False, "malformed")
    expected = _context(
        subject=subject,
        tenant=tenant,
        project=project,
        bundle_id=bundle_id,
        operation=operation,
        method=method,
    )
    for key, value in expected.items():
        if not secrets.compare_digest(
            str(payload.get(key) or "").encode("utf-8"),
            value.encode("utf-8"),
        ):
            return OperationCsrfValidation(False, f"{key}_mismatch")
    return OperationCsrfValidation(True, "ok")


async def mint_request_operation_csrf_token(
    request: Request,
    **context: Any,
) -> str:
    """Mint through the state backend owned by the current runtime."""

    return await mint_operation_csrf_token(_request_state_store(request), **context)


async def consume_request_operation_csrf_token(
    request: Request,
    token: str,
    **context: Any,
) -> OperationCsrfValidation:
    """Consume through the current runtime backend and fail closed on outage."""

    try:
        state_store = _request_state_store(request)
    except Exception:
        return OperationCsrfValidation(False, "store_unavailable")
    return await consume_operation_csrf_token(state_store, token, **context)


__all__ = [
    "OPERATION_CSRF_HEADER",
    "OPERATION_CSRF_TTL_SECONDS",
    "OperationCsrfValidation",
    "authenticated_session_subject",
    "consume_operation_csrf_token",
    "consume_request_operation_csrf_token",
    "mint_operation_csrf_token",
    "mint_request_operation_csrf_token",
    "request_uses_cookie_auth",
]
