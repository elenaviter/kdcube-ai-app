# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import pytest

from kdcube_ai_app.apps.chat.sdk.events.event_bus import EventLaneStateLockTimeout
from kdcube_ai_app.apps.chat.sdk.events.event_bus.state import RedisEventLaneStateTable


class _AckDelayedRedis:
    """Redis-shaped fake that separates server application from client ACK."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.server_applied = asyncio.Event()
        self.release_ack = asyncio.Event()
        self.delay_lock_key = ""
        self.delay_release = False
        self.release_server_applied = asyncio.Event()
        self.release_release_ack = asyncio.Event()

    async def get(self, key: str) -> Any:
        return self.data.get(str(key))

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        del ex
        key = str(key)
        if nx and key in self.data:
            return False
        self.data[key] = value
        if nx and key == self.delay_lock_key:
            self.server_applied.set()
            await self.release_ack.wait()
        return True

    async def delete(self, key: str) -> int:
        return int(self.data.pop(str(key), None) is not None)

    async def eval(self, script: str, _numkeys: int, key: str, token: str, *args: str) -> int:
        key = str(key)
        if self.data.get(key) != token:
            return 0
        if "DEL" in script:
            del self.data[key]
            if self.delay_release:
                self.release_server_applied.set()
                await self.release_release_ack.wait()
            return 1
        if "EXPIRE" in script:
            return 1
        raise AssertionError(f"unsupported script args={args}")

    async def pttl(self, key: str) -> int:
        return 10_000 if str(key) in self.data else -2


@pytest.mark.asyncio
async def test_cancel_after_server_acquires_lock_cleans_exact_owner_token() -> None:
    redis = _AckDelayedRedis()
    table = RedisEventLaneStateTable(
        redis=redis,
        state_key="lane:state",
        lock_ttl_seconds=10,
    )
    redis.delay_lock_key = table.lock_key

    async def _enter_lock() -> None:
        async with table.lock(operation="cancel-window-regression"):
            raise AssertionError("cancelled acquisition must not enter the critical section")

    task = asyncio.create_task(_enter_lock())
    await asyncio.wait_for(redis.server_applied.wait(), timeout=1)
    assert table.lock_key in redis.data

    task.cancel()
    await asyncio.sleep(0)
    redis.release_ack.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert table.lock_key not in redis.data


@pytest.mark.asyncio
async def test_cancel_after_server_releases_lock_waits_for_known_outcome() -> None:
    redis = _AckDelayedRedis()
    table = RedisEventLaneStateTable(redis=redis, state_key="lane:state")
    entered = asyncio.Event()
    leave_body = asyncio.Event()
    redis.delay_release = True

    async def _hold_lock() -> None:
        async with table.lock(operation="release-cancel-window"):
            entered.set()
            await leave_body.wait()

    task = asyncio.create_task(_hold_lock())
    await asyncio.wait_for(entered.wait(), timeout=1)
    leave_body.set()
    await asyncio.wait_for(redis.release_server_applied.wait(), timeout=1)
    assert table.lock_key not in redis.data

    task.cancel()
    await asyncio.sleep(0)
    redis.release_release_ack.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert table.lock_key not in redis.data


@pytest.mark.asyncio
async def test_lock_timeout_reports_holder_operation_without_exposing_token() -> None:
    redis = _AckDelayedRedis()
    table = RedisEventLaneStateTable(redis=redis, state_key="lane:state")
    redis.data[table.lock_key] = (
        '{"acquired_at":"2026-07-28T09:46:45Z","operation":"mark_consumer_active",'
        '"pid":44,"task":"react-timeline-events:conversation:turn","token":"secret-token"}'
    )

    with pytest.raises(EventLaneStateLockTimeout) as raised:
        async with table.lock(timeout_seconds=0.05, operation="schedule_consumer_from_wake"):
            pass

    error = raised.value
    assert error.operation == "schedule_consumer_from_wake"
    assert error.holder_operation == "mark_consumer_active"
    assert error.holder_pid == "44"
    assert error.pttl_ms == 10_000
    assert "secret-token" not in str(error)


class _DelayedRealRedis:
    def __init__(self, inner: Any, *, lock_key: str) -> None:
        self.inner = inner
        self.lock_key = lock_key
        self.server_applied = asyncio.Event()
        self.release_ack = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def set(self, key: str, value: str, **kwargs: Any) -> Any:
        result = await self.inner.set(key, value, **kwargs)
        if str(key) == self.lock_key and kwargs.get("nx") and result:
            self.server_applied.set()
            await self.release_ack.wait()
        return result


@pytest.mark.asyncio
async def test_real_redis_cancel_after_server_acquire_does_not_leave_ttl_lock() -> None:
    redis_url = str(os.getenv("KDCUBE_TEST_REDIS_URL") or "").strip()
    if not redis_url:
        pytest.skip("set KDCUBE_TEST_REDIS_URL to run the disposable-Redis regression")

    redis_asyncio = pytest.importorskip("redis.asyncio")
    client = redis_asyncio.from_url(redis_url, decode_responses=True)
    state_key = f"test:event-lane:{uuid.uuid4().hex}:state"
    lock_key = f"{state_key}:lock"
    redis = _DelayedRealRedis(client, lock_key=lock_key)
    table = RedisEventLaneStateTable(
        redis=redis,
        state_key=state_key,
        lock_key=lock_key,
        lock_ttl_seconds=10,
    )

    async def _enter_lock() -> None:
        async with table.lock(operation="real-redis-cancel-window"):
            raise AssertionError("cancelled acquisition must not enter the critical section")

    try:
        task = asyncio.create_task(_enter_lock())
        await asyncio.wait_for(redis.server_applied.wait(), timeout=2)
        assert await client.pttl(lock_key) > 0
        task.cancel()
        await asyncio.sleep(0)
        redis.release_ack.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await client.pttl(lock_key) == -2
        async with table.lock(operation="real-redis-next-wake"):
            assert await client.pttl(lock_key) > 0
        assert await client.pttl(lock_key) == -2
    finally:
        await client.delete(state_key, lock_key)
        closer = getattr(client, "aclose", None) or getattr(client, "close")
        await closer()
