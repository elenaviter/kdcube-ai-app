# SPDX-License-Identifier: MIT

"""Regression tests for the /api/cp-frontend-config handler.

These guard the fix for the chat-ingress event-loop freeze: the config is
built by synchronous, blocking I/O (reads assembly.yaml, typically from EFS),
so it must run off the event loop, be time-bounded, and fall back to the
last-known-good value instead of hanging when the descriptor read stalls.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from kdcube_ai_app.apps.chat.ingress.control_plane import config as cp_config


@pytest.fixture(autouse=True)
def _reset_cache():
    cp_config._config_cache = None
    cp_config._config_expiry = 0.0
    yield
    cp_config._config_cache = None
    cp_config._config_expiry = 0.0


@pytest.mark.asyncio
async def test_config_is_cached_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _build():
        calls["n"] += 1
        return {"v": calls["n"]}

    monkeypatch.setattr(cp_config, "build_frontend_config", _build)
    monkeypatch.setattr(cp_config, "_CONFIG_TTL_SECONDS", 60.0)

    first = await cp_config._get_frontend_config()
    second = await cp_config._get_frontend_config()

    assert first == {"v": 1}
    assert second == {"v": 1}  # served from cache
    assert calls["n"] == 1  # built exactly once within the TTL


@pytest.mark.asyncio
async def test_stalled_rebuild_serves_last_known_good(monkeypatch):
    # Prime the cache with a good build.
    monkeypatch.setattr(cp_config, "build_frontend_config", lambda: {"ok": True})
    monkeypatch.setattr(cp_config, "_CONFIG_TTL_SECONDS", 60.0)
    assert await cp_config._get_frontend_config() == {"ok": True}

    # Force a refresh whose build blocks far longer than the timeout.
    cp_config._config_expiry = 0.0
    monkeypatch.setattr(cp_config, "_CONFIG_BUILD_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(cp_config, "_CONFIG_RETRY_SECONDS", 0.0)

    def _stalled():
        time.sleep(5)  # simulate a hung EFS read
        return {"ok": "new"}

    monkeypatch.setattr(cp_config, "build_frontend_config", _stalled)

    started = time.monotonic()
    result = await cp_config._get_frontend_config()
    elapsed = time.monotonic() - started

    assert result == {"ok": True}  # last-known-good, not a hang or a 500
    assert elapsed < 2.0  # returned promptly instead of blocking on the read


@pytest.mark.asyncio
async def test_blocking_build_does_not_block_event_loop(monkeypatch):
    monkeypatch.setattr(cp_config, "_CONFIG_BUILD_TIMEOUT_SECONDS", 2.0)

    def _slow():
        time.sleep(0.5)  # blocking, but must run off the loop
        return {"ok": True}

    monkeypatch.setattr(cp_config, "build_frontend_config", _slow)

    ticks = 0

    async def _ticker():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.02)
            ticks += 1

    _, cfg = await asyncio.gather(_ticker(), cp_config._get_frontend_config())

    assert cfg == {"ok": True}
    # The event loop kept ticking while the build slept in a worker thread.
    assert ticks >= 10
