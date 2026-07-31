# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import json
import time

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.automation_access import (
    ACCESS_SOURCE_OAUTH,
    AutomationAccessRecord,
    automation_record_key,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.live_grant import (
    LiveGrantCardError,
    live_grants_for_resource,
    resolve_live_grant_card,
)


TENANT = "tenant-a"
PROJECT = "project-a"
ACCESS_ID = "oauth-access-1"
RESOURCE = "https://runtime.example.test/mcp/productivity"
GRANTOR = "user-1"
CLIENT = "claude"
DELEGATE = "integration:claude:user-1"


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.fail_get = False

    async def get(self, key: str):
        if self.fail_get:
            raise RuntimeError("redis unavailable")
        return self.values.get(key)


def _record(
    *,
    operations=("sheets_read",),
    resource_grants=None,
    expires_at=None,
) -> AutomationAccessRecord:
    return AutomationAccessRecord(
        access_id=ACCESS_ID,
        label="Claude productivity",
        client_id=CLIENT,
        grantor_subject=GRANTOR,
        delegate_subject=DELEGATE,
        operations=tuple(operations),
        resource_grants=resource_grants
        if resource_grants is not None
        else {RESOURCE: ("sheets:read",)},
        expires_at=int(expires_at if expires_at is not None else time.time() + 3600),
        source=ACCESS_SOURCE_OAUTH,
    )


def _key() -> str:
    return automation_record_key(TENANT, PROJECT, ACCESS_ID)


async def _resolve(redis: _Redis):
    return await resolve_live_grant_card(
        redis,
        tenant=TENANT,
        project=PROJECT,
        access_id=ACCESS_ID,
        expected_client_id=CLIENT,
        expected_grantor_subject=GRANTOR,
        expected_delegate_subject=DELEGATE,
    )


@pytest.mark.asyncio
async def test_live_grant_resolves_current_valid_card():
    redis = _Redis()
    redis.values[_key()] = json.dumps(_record().to_dict())

    resolved = await _resolve(redis)

    assert resolved is not None
    assert resolved.operations == ("sheets_read",)
    assert resolved.resource_grants == {RESOURCE: ("sheets:read",)}


@pytest.mark.asyncio
async def test_live_grant_absent_or_expired_is_revoked():
    redis = _Redis()
    assert await _resolve(redis) is None

    redis.values[_key()] = json.dumps(
        _record(expires_at=int(time.time()) - 1).to_dict()
    )
    assert await _resolve(redis) is None


@pytest.mark.asyncio
async def test_live_grant_lookup_failure_is_not_a_snapshot_fallback():
    redis = _Redis()
    redis.fail_get = True

    with pytest.raises(LiveGrantCardError, match="lookup_unavailable") as exc_info:
        await _resolve(redis)

    assert exc_info.value.reason == "lookup_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("{", "malformed_json"),
        (json.dumps([]), "record_not_object"),
        (json.dumps({"schema": "wrong"}), "schema_mismatch"),
        (
            json.dumps(
                {
                    **_record().to_dict(),
                    "operations": "sheets_read",
                }
            ),
            "operations_invalid",
        ),
    ],
)
async def test_live_grant_malformed_or_invalid_card_fails_closed(payload, reason):
    redis = _Redis()
    redis.values[_key()] = payload

    with pytest.raises(LiveGrantCardError) as exc_info:
        await _resolve(redis)

    assert exc_info.value.reason == reason


@pytest.mark.asyncio
async def test_live_grant_binding_mismatch_fails_closed():
    redis = _Redis()
    payload = _record().to_dict()
    payload["client_id"] = "different-client"
    redis.values[_key()] = json.dumps(payload)

    with pytest.raises(LiveGrantCardError) as exc_info:
        await _resolve(redis)

    assert exc_info.value.reason == "client_id_mismatch"


def test_live_resource_grants_preserve_explicit_empty_narrowing():
    record = _record(resource_grants={RESOURCE: ()})

    assert live_grants_for_resource(record, RESOURCE) == ()
    assert live_grants_for_resource(record, "https://runtime.example.test/mcp/other") is None
