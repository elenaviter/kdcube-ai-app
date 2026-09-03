from __future__ import annotations

import json

import pytest

from kdcube_ai_app.apps.chat.proc.rest.management.contracts import (
    INSPECT_OPERATION,
    management_request_digest,
    management_resource,
    validate_application_id,
    validate_invocation_id,
)
from kdcube_ai_app.apps.chat.proc.rest.management.effect_ledger import (
    ACTION_CONFLICT,
    ACTION_EXECUTE,
    ACTION_PENDING,
    ACTION_REPLAY,
    ACTION_UNKNOWN,
    RedisEffectLedger,
)


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(self, key, value, *, nx=False):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def eval(self, _script, _keys, key, state, owner, digest, replacement):
        current = json.loads(self.values[key])
        if current["state"] != state:
            return -1
        if current["owner"] != owner:
            return -2
        if current["request_digest"] != digest:
            return -3
        self.values[key] = replacement
        return 1


def test_resource_and_request_identity_are_canonical() -> None:
    resource = management_resource("tenant one", "project/a")
    first = management_request_digest(
        resource=resource,
        operation=INSPECT_OPERATION,
        body={},
    )
    second = management_request_digest(
        operation=INSPECT_OPERATION,
        body={},
        resource=resource,
    )

    assert resource == "urn:kdcube:management:deployment:tenant%20one:project%2Fa"
    assert first == second
    assert len(first) == 64
    assert validate_application_id("workspace@2026-03-31") == "workspace@2026-03-31"
    assert validate_invocation_id("reload-request-1") == "reload-request-1"
    with pytest.raises(ValueError):
        validate_application_id("*")
    with pytest.raises(ValueError):
        validate_application_id("path/to/app")
    with pytest.raises(ValueError):
        validate_invocation_id("contains space")


@pytest.mark.asyncio
async def test_redis_effect_ledger_is_shared_idempotency_authority() -> None:
    redis = _Redis()
    now = [1000.0]
    ledger = RedisEffectLedger(
        redis,
        tenant="tenant-a",
        project="project-a",
        pending_seconds=10,
        clock=lambda: now[0],
    )
    args = {
        "access_id": "access-1",
        "resource": "urn:kdcube:management:deployment:tenant-a:project-a",
        "operation": INSPECT_OPERATION,
        "invocation_id": "inspect-1",
        "request_digest": "a" * 64,
        "audit": {"access_id": "access-1"},
    }

    first = await ledger.reserve(**args)
    pending = await ledger.reserve(**args)
    conflict = await ledger.reserve(**{**args, "request_digest": "b" * 64})
    now[0] = 1011.0
    unknown = await ledger.reserve(**args)
    await ledger.finish(
        access_id=args["access_id"],
        resource=args["resource"],
        operation=args["operation"],
        invocation_id=args["invocation_id"],
        request_digest=args["request_digest"],
        owner=first.owner,
        status_code=200,
        response={"ok": True},
    )
    replay = await ledger.reserve(**args)

    assert first.action == ACTION_EXECUTE
    assert pending.action == ACTION_PENDING
    assert conflict.action == ACTION_CONFLICT
    assert unknown.action == ACTION_UNKNOWN
    assert replay.action == ACTION_REPLAY
    assert replay.record["response"] == {"ok": True}


@pytest.mark.asyncio
async def test_redis_effect_ledger_scopes_invocation_ids_to_access_id() -> None:
    redis = _Redis()
    ledger = RedisEffectLedger(
        redis,
        tenant="tenant-a",
        project="project-a",
    )
    common = {
        "resource": "urn:kdcube:management:deployment:tenant-a:project-a",
        "operation": INSPECT_OPERATION,
        "invocation_id": "shared-by-two-callers",
        "request_digest": "a" * 64,
    }

    first = await ledger.reserve(
        **common,
        access_id="access-1",
        audit={"access_id": "access-1"},
    )
    second = await ledger.reserve(
        **common,
        access_id="access-2",
        audit={"access_id": "access-2"},
    )

    assert first.action == ACTION_EXECUTE
    assert second.action == ACTION_EXECUTE
    assert len(redis.values) == 2
