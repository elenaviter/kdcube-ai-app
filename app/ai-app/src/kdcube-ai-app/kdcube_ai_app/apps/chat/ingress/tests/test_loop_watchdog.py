# SPDX-License-Identifier: MIT

"""Tests for the chat-ingress event-loop liveness watchdog."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from kdcube_ai_app.apps.chat.ingress.loop_watchdog import LoopWatchdog


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CHAT_INGRESS_LOOP_WATCHDOG_ENABLED", raising=False)
    wd = LoopWatchdog()
    assert wd.enabled is False
    wd.start()  # no-op
    assert wd._running is False
    assert wd._thread is None


def test_should_kill_logic():
    wd = LoopWatchdog(enabled=True, stall_seconds=1.0)
    wd._running = True

    wd._last_tick = time.monotonic()  # fresh
    assert wd._should_kill() is False

    wd._last_tick = time.monotonic() - 5.0  # stale
    assert wd._should_kill() is True

    wd._running = False  # never kill while stopped
    assert wd._should_kill() is False


def test_watch_loop_fires_on_stall():
    # Ticker intentionally not started, so the heartbeat stays stale.
    wd = LoopWatchdog(
        enabled=True, stall_seconds=0.05, tick_interval=100.0, poll_interval=0.02
    )
    fired = threading.Event()
    wd._on_stall = fired.set  # observe instead of os._exit
    wd._running = True
    wd._last_tick = time.monotonic() - 10.0

    t = threading.Thread(target=wd._watch_loop, daemon=True)
    t.start()
    try:
        assert fired.wait(2.0) is True
    finally:
        wd._running = False
        t.join(2.0)


@pytest.mark.asyncio
async def test_ticker_keeps_heartbeat_fresh():
    wd = LoopWatchdog(
        enabled=True, stall_seconds=1.0, tick_interval=0.02, poll_interval=100.0
    )
    fired = {"n": 0}
    wd._on_stall = lambda: fired.__setitem__("n", fired["n"] + 1)

    wd.start()
    try:
        await asyncio.sleep(0.1)  # loop is healthy; ticker keeps bumping
        assert wd._stalled_for() < 0.5
        assert fired["n"] == 0
    finally:
        await wd.stop()
    assert wd._running is False


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_cancels_ticker():
    wd = LoopWatchdog(enabled=True, tick_interval=0.02, poll_interval=100.0)
    wd.start()
    await wd.stop()
    await wd.stop()  # second stop must not raise
    assert wd._ticker_task is None
