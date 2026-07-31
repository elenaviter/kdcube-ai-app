# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers

from kdcube_ai_app.apps.chat.proc.rest.integrations import operation_csrf


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.fail_eval = False

    async def setex(self, key: str, _ttl: int, value: str) -> bool:
        self.values[key] = value
        return True

    async def eval(self, _script: str, _numkeys: int, key: str):
        if self.fail_eval:
            raise RuntimeError("redis unavailable")
        return self.values.pop(key, None)


def _request(*, headers=None, cookies=None):
    return SimpleNamespace(
        headers=Headers(headers or {}),
        cookies=dict(cookies or {}),
    )


@pytest.fixture(autouse=True)
def _auth_settings(monkeypatch):
    monkeypatch.setattr(
        operation_csrf,
        "get_settings",
        lambda: SimpleNamespace(
            AUTH=SimpleNamespace(
                ID_TOKEN_HEADER_NAME="X-ID-Token",
                AUTH_TOKEN_COOKIE_NAME="__Secure-LATC",
                ID_TOKEN_COOKIE_NAME="__Secure-LITC",
                MASQUERADED_TOKEN_COOKIE_NAME="__Secure-LMTC",
            )
        ),
    )


def test_cookie_auth_requires_csrf_but_explicit_credentials_do_not():
    cookie_request = _request(cookies={"__Secure-LATC": "ambient-session"})
    bearer_request = _request(
        headers={"Authorization": "Bearer explicit"},
        cookies={"__Secure-LATC": "ambient-session"},
    )
    id_header_request = _request(
        headers={"X-ID-Token": "explicit"},
        cookies={"__Secure-LATC": "ambient-session"},
    )

    assert operation_csrf.request_uses_cookie_auth(cookie_request) is True
    assert operation_csrf.request_uses_cookie_auth(bearer_request) is False
    assert operation_csrf.request_uses_cookie_auth(id_header_request) is False


@pytest.mark.asyncio
async def test_operation_csrf_token_is_context_bound_and_single_use():
    redis = _Redis()
    token = await operation_csrf.mint_operation_csrf_token(
        redis,
        subject="user@example.test",
        tenant="tenant-a",
        project="project-a",
        bundle_id="connection-hub@1-0",
        operation="delegated_access_create",
    )

    wrong_operation = await operation_csrf.consume_operation_csrf_token(
        redis,
        token,
        subject="user@example.test",
        tenant="tenant-a",
        project="project-a",
        bundle_id="connection-hub@1-0",
        operation="delegated_access_revoke",
    )
    assert wrong_operation.ok is False
    assert wrong_operation.reason == "operation_mismatch"

    # A mismatched attempt consumes the one-time token as well.
    replay = await operation_csrf.consume_operation_csrf_token(
        redis,
        token,
        subject="user@example.test",
        tenant="tenant-a",
        project="project-a",
        bundle_id="connection-hub@1-0",
        operation="delegated_access_create",
    )
    assert replay.ok is False
    assert replay.reason == "not_found"


@pytest.mark.asyncio
async def test_operation_csrf_token_accepts_exact_context_once():
    redis = _Redis()
    token = await operation_csrf.mint_operation_csrf_token(
        redis,
        subject="user-123",
        tenant="tenant-a",
        project="project-a",
        bundle_id="connection-hub@1-0",
        operation="delegated_access_create",
    )
    context = {
        "subject": "user-123",
        "tenant": "tenant-a",
        "project": "project-a",
        "bundle_id": "connection-hub@1-0",
        "operation": "delegated_access_create",
    }

    accepted = await operation_csrf.consume_operation_csrf_token(redis, token, **context)
    replay = await operation_csrf.consume_operation_csrf_token(redis, token, **context)

    assert accepted.ok is True
    assert replay.ok is False
    assert replay.reason == "not_found"


@pytest.mark.asyncio
async def test_operation_csrf_store_failure_is_distinct_from_bad_token():
    redis = _Redis()
    redis.fail_eval = True

    result = await operation_csrf.consume_operation_csrf_token(
        redis,
        "present-token",
        subject="user-1",
        tenant="tenant-a",
        project="project-a",
        bundle_id="connection-hub@1-0",
        operation="delegated_access_create",
    )

    assert result.ok is False
    assert result.reason == "store_unavailable"
