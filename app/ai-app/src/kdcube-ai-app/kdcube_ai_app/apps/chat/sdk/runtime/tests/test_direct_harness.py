from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime import direct_harness as module
from kdcube_ai_app.apps.chat.sdk.runtime.direct_harness import (
    DirectAgentHarness,
    DirectAgentHarnessConfig,
)


def _config(**overrides):
    values = {
        "tenant": "tenant",
        "project": "project",
        "user_id": "user",
        "user_type": "regular",
        "session_id": "session",
        "bundle_id": "example@1-0",
        "agent_id": "agent",
        "postgres_url": "postgresql://example",
        "redis_url": "redis://example",
        "storage_uri": "file:///tmp/example",
    }
    values.update(overrides)
    return DirectAgentHarnessConfig(**values)


def test_config_requires_every_direct_host_boundary() -> None:
    with pytest.raises(ValueError, match="storage_uri"):
        _config(storage_uri="")
    with pytest.raises(ValueError, match="turn_cache_ttl_seconds"):
        _config(turn_cache_ttl_seconds=0)


@pytest.mark.asyncio
async def test_turn_records_once_and_requires_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def passthrough(*_args, **_kwargs):
        yield

    monkeypatch.setattr(module, "bind_accounting", passthrough)
    monkeypatch.setattr(module, "with_accounting", passthrough)
    monkeypatch.setattr(module, "get_turn_events", AsyncMock(return_value=[{"usage": 1}]))
    record = AsyncMock(return_value=True)
    monkeypatch.setattr(module, "record_minimal_turn_log_if_absent", record)

    emitter = SimpleNamespace(emit=AsyncMock())
    harness = DirectAgentHarness(config=_config(), model_service=None, emitter=emitter)
    harness._store = SimpleNamespace(backend=object())
    harness._conversation_client = object()

    async with harness.turn(conversation_id="conversation", turn_id="turn") as turn:
        await turn.complete(prompt="question", final_answer="answer")

    record.assert_awaited_once()
    assert turn.accounting_events == [{"usage": 1}]
    assert turn.finished is True


@pytest.mark.asyncio
async def test_turn_fails_when_caller_forgets_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def passthrough(*_args, **_kwargs):
        yield

    monkeypatch.setattr(module, "bind_accounting", passthrough)
    monkeypatch.setattr(module, "with_accounting", passthrough)
    monkeypatch.setattr(module, "get_turn_events", AsyncMock(return_value=[{"usage": 1}]))

    harness = DirectAgentHarness(
        config=_config(),
        model_service=None,
        emitter=SimpleNamespace(emit=AsyncMock()),
    )
    harness._store = SimpleNamespace(backend=object())
    harness._conversation_client = object()

    with pytest.raises(RuntimeError, match="turn.complete"):
        async with harness.turn(conversation_id="conversation", turn_id="turn"):
            pass


@pytest.mark.asyncio
async def test_completed_model_turn_requires_accounting_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def passthrough(*_args, **_kwargs):
        yield

    monkeypatch.setattr(module, "bind_accounting", passthrough)
    monkeypatch.setattr(module, "with_accounting", passthrough)
    monkeypatch.setattr(module, "get_turn_events", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        module,
        "record_minimal_turn_log_if_absent",
        AsyncMock(return_value=True),
    )

    harness = DirectAgentHarness(
        config=_config(),
        model_service=None,
        emitter=SimpleNamespace(emit=AsyncMock()),
    )
    harness._store = SimpleNamespace(backend=object())
    harness._conversation_client = object()

    with pytest.raises(RuntimeError, match="no Redis accounting evidence"):
        async with harness.turn(conversation_id="conversation", turn_id="turn") as turn:
            await turn.complete(prompt="question", final_answer="answer")


@pytest.mark.asyncio
async def test_verify_conversation_requires_postgres_and_storage_evidence() -> None:
    recent = AsyncMock(
        return_value={
            "items": [
                {
                    "turn_id": "turn-1",
                    "hosted_uri": "file:///turn-1.json",
                    "payload": {"blocks": []},
                }
            ]
        }
    )
    harness = DirectAgentHarness(
        config=_config(),
        model_service=None,
        emitter=SimpleNamespace(emit=AsyncMock()),
    )
    harness._conversation_client = SimpleNamespace(recent=recent)

    rows = await harness.verify_conversation(
        conversation_id="conversation",
        expected_turn_ids=("turn-1",),
    )

    assert rows[0]["turn_id"] == "turn-1"
    recent.assert_awaited_once()
