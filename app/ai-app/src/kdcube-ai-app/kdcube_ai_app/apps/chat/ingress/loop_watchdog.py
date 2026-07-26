# SPDX-License-Identifier: MIT

"""Event-loop liveness watchdog for chat-ingress uvicorn workers.

A blocking call executed on the asyncio event loop (e.g. synchronous I/O in a
request handler) freezes a worker: it stops serving *every* route while the
process still looks alive. Because chat-ingress runs multiple uvicorn workers
behind a single port, the container health check (``curl /health``) can
round-robin to a surviving worker and keep passing, so a partially-frozen
service is never recycled by ECS.

This watchdog runs one daemon thread per worker. The event loop bumps a
monotonic heartbeat every ``tick_interval`` seconds; if the heartbeat goes
stale beyond ``stall_seconds`` — i.e. the loop has been blocked that long — the
watchdog dumps all thread stacks (via faulthandler) and force-exits the worker
so the uvicorn supervisor respawns a fresh one.

Disabled by default. Enable with ``CHAT_INGRESS_LOOP_WATCHDOG_ENABLED=1`` only
after a staging soak, and keep ``stall_seconds`` comfortably above the longest
expected synchronous section so legitimately busy workers are not killed.
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class LoopWatchdog:
    """Force-exits a worker whose asyncio event loop has stalled."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        stall_seconds: float | None = None,
        tick_interval: float | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self.enabled = (
            _env_flag("CHAT_INGRESS_LOOP_WATCHDOG_ENABLED", False)
            if enabled is None
            else enabled
        )
        self.stall_seconds = (
            _env_float("CHAT_INGRESS_LOOP_WATCHDOG_STALL_SECONDS", 30.0)
            if stall_seconds is None
            else stall_seconds
        )
        self.tick_interval = (
            _env_float("CHAT_INGRESS_LOOP_WATCHDOG_TICK_SECONDS", 1.0)
            if tick_interval is None
            else tick_interval
        )
        self.poll_interval = (
            _env_float("CHAT_INGRESS_LOOP_WATCHDOG_POLL_SECONDS", 5.0)
            if poll_interval is None
            else poll_interval
        )
        self._last_tick = time.monotonic()
        self._running = False
        self._thread: threading.Thread | None = None
        self._ticker_task: asyncio.Task | None = None
        # Test seam: overridden so tests can observe a stall without exiting.
        self._on_stall = self._default_on_stall

    def _stalled_for(self) -> float:
        return time.monotonic() - self._last_tick

    def _should_kill(self) -> bool:
        return self._running and self._stalled_for() >= self.stall_seconds

    async def _tick_loop(self) -> None:
        while self._running:
            self._last_tick = time.monotonic()
            await asyncio.sleep(self.tick_interval)

    def _watch_loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            if self._should_kill():
                logger.critical(
                    "chat-ingress event loop stalled for %.1fs (>= %.1fs); "
                    "dumping stacks and exiting worker pid=%s",
                    self._stalled_for(),
                    self.stall_seconds,
                    os.getpid(),
                )
                self._on_stall()
                return

    def _default_on_stall(self) -> None:
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:  # pragma: no cover - best effort diagnostics
            pass
        # The loop is frozen, so a graceful shutdown cannot run. Force-exit and
        # let the uvicorn supervisor respawn the worker.
        os._exit(1)

    def start(self) -> None:
        """Start the ticker task and watcher thread (no-op if disabled)."""
        if not self.enabled or self._running:
            return
        self._running = True
        self._last_tick = time.monotonic()
        self._ticker_task = asyncio.ensure_future(self._tick_loop())
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="chat-ingress-loop-watchdog",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "chat-ingress loop watchdog started: stall=%.1fs tick=%.1fs poll=%.1fs pid=%s",
            self.stall_seconds,
            self.tick_interval,
            self.poll_interval,
            os.getpid(),
        )

    async def stop(self) -> None:
        """Stop the watcher so shutdown/drain never trips a force-exit."""
        if not self._running:
            return
        self._running = False
        task = self._ticker_task
        self._ticker_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive
                logger.debug("loop watchdog ticker raised on cancel", exc_info=True)
