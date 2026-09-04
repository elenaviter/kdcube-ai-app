from __future__ import annotations

from types import SimpleNamespace

import pytest

from kdcube_ai_app.infra.accounting import FileAccountingStorage, _get_storage
from kdcube_ai_app.infra.accounting.envelope import bind_accounting, build_envelope_from_session


class _Backend:
    async def write_text_a(self, _path: str, _content: str) -> None:
        return None


@pytest.mark.asyncio
async def test_bind_accounting_can_disable_redis_turn_cache() -> None:
    envelope = build_envelope_from_session(
        SimpleNamespace(user_id="user", session_id="session", user_type="regular", timezone="UTC"),
        tenant_id="tenant",
        project_id="project",
        request_id="request",
        component="standalone-test",
    )

    async with bind_accounting(
        envelope,
        _Backend(),
        redis_turn_cache=False,
    ):
        storage = _get_storage()
        assert isinstance(storage, FileAccountingStorage)
        assert storage.turn_cache is None


@pytest.mark.asyncio
async def test_bind_accounting_preserves_hosted_redis_default() -> None:
    envelope = build_envelope_from_session(
        SimpleNamespace(user_id="user", session_id="session", user_type="regular", timezone="UTC"),
        tenant_id="tenant",
        project_id="project",
        request_id="request",
        component="hosted-test",
    )

    async with bind_accounting(envelope, _Backend()):
        storage = _get_storage()
        assert isinstance(storage, FileAccountingStorage)
        assert storage.turn_cache is not None


@pytest.mark.asyncio
async def test_bind_accounting_accepts_direct_redis_configuration() -> None:
    envelope = build_envelope_from_session(
        SimpleNamespace(user_id="user", session_id="session", user_type="regular", timezone="UTC"),
        tenant_id="tenant",
        project_id="project",
        request_id="request",
        component="direct-sdk-test",
    )

    async with bind_accounting(
        envelope,
        _Backend(),
        redis_url="redis://:secret@127.0.0.1:56379/7",
        turn_cache_ttl_s=91,
    ):
        storage = _get_storage()
        assert isinstance(storage, FileAccountingStorage)
        assert storage.turn_cache is not None
        assert storage.turn_cache.ttl_seconds == 91
        connection = storage.turn_cache.redis.connection_pool.connection_kwargs
        assert connection["host"] == "127.0.0.1"
        assert connection["port"] == 56379
        assert connection["db"] == 7
