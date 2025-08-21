# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
import traceback
from typing import Optional, Dict, Any, Iterable

from kdcube_ai_app.apps.chat.emitters import SocketIOEmitter, NoopEmitter
from kdcube_ai_app.infra.availability.health_and_heartbeat import MultiprocessDistributedMiddleware, logger
from kdcube_ai_app.storage.storage import create_storage_backend

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class EnhancedChatRequestProcessor:
    """
    Queue worker that:
      - Pops tasks fairly from multiple queues
      - Acquires + renews a per-task Redis lock
      - Emits chat_* events via an injected emitter (Socket.IO or custom)
      - Enforces per-task timeout
      - Handles graceful shutdown
    """

    QUEUE_ORDER: Iterable[str] = ("privileged", "registered", "anonymous")

    def __init__(
            self,
            middleware: MultiprocessDistributedMiddleware,
            chat_handler,
            *,
            process_id: Optional[int] = None,
            emitter: Optional[Any] = None,     # anything with `await emit(event, data, room=...)`
            socketio=None,                     # convenience: if you pass sio, we wrap it
            max_concurrent: Optional[int] = None,
            task_timeout_sec: Optional[int] = None,
            lock_ttl_sec: int = 300,
            lock_renew_sec: int = 60,
    ):
        self.middleware = middleware
        self.chat_handler = chat_handler
        self.process_id = process_id or os.getpid()
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_CHAT", str(max_concurrent or 5)))
        self.task_timeout_sec = int(os.getenv("CHAT_TASK_TIMEOUT_SEC", str(task_timeout_sec or 600)))
        self.lock_ttl_sec = lock_ttl_sec
        self.lock_renew_sec = lock_renew_sec

        self._emitter = (
                emitter
                or (SocketIOEmitter(socketio) if socketio is not None else NoopEmitter())
        )

        self._processor_task: Optional[asyncio.Task] = None
        self._active_tasks: set[asyncio.Task] = set()
        self._current_load = 0
        self._stop_event = asyncio.Event()

        # round-robin index
        self._queue_idx = 0

    # ---------------- Public API ----------------

    async def start_processing(self):
        if self._processor_task and not self._processor_task.done():
            return
        self._processor_task = asyncio.create_task(self._processing_loop(), name="chat-processing-loop")

    async def stop_processing(self):
        self._stop_event.set()
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        if self._active_tasks:
            await asyncio.gather(*list(self._active_tasks), return_exceptions=True)

    def get_current_load(self) -> int:
        return self._current_load

    # ---------------- Core loop ----------------

    async def _processing_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._current_load >= self.max_concurrent:
                    await asyncio.sleep(0.05)
                    continue

                task_data = await self._pop_any_queue_fair()
                if not task_data:
                    await asyncio.sleep(0.05)
                    continue

                task = asyncio.create_task(self._process_task(task_data), name=f"chat-task:{task_data.get('task_id')}")
                self._active_tasks.add(task)
                # remove from set when done
                task.add_done_callback(lambda t: self._active_tasks.discard(t))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.5)

    async def _pop_any_queue_fair(self) -> Optional[Dict[str, Any]]:
        """
        Round-robin across QUEUE_ORDER, try a non-blocking pop with very small timeout.
        """
        for _ in range(len(self.QUEUE_ORDER)):
            user_type = self.QUEUE_ORDER[self._queue_idx]
            self._queue_idx = (self._queue_idx + 1) % len(self.QUEUE_ORDER)

            if self._current_load >= self.max_concurrent:
                return None

            queue_key = f"{self.middleware.QUEUE_PREFIX}:{user_type}"
            # tiny blocking timeframe to avoid busy-loop
            raw = await self.middleware.redis.brpop(queue_key, timeout=0.1)
            if not raw:
                continue

            try:
                task_dict = json.loads(raw[1])
            except Exception:
                logger.error("Invalid task payload (not JSON); dropping")
                continue

            task_id = task_dict.get("task_id")
            if not task_id:
                logger.error("Task missing task_id; dropping")
                continue

            # Attempt lock
            lock_key = f"{self.middleware.LOCK_PREFIX}:{task_id}"
            acquired = await self.middleware.redis.set(
                lock_key,
                f"{self.middleware.instance_id}:{self.process_id}",
                nx=True,
                ex=self.lock_ttl_sec,
            )
            if acquired:
                self._current_load += 1
                logger.info(f"Process {self.process_id} acquired task {task_id} ({user_type})")
                task_dict["_lock_key"] = lock_key
                task_dict["_queue_key"] = queue_key
                return task_dict

            # Put back if someone else locked first
            await self.middleware.redis.lpush(queue_key, json.dumps(task_dict))
        return None

    # ---------------- Per-task execution ----------------

    @asynccontextmanager
    async def _lock_renewer(self, lock_key: str):
        """
        Background coroutine that keeps the Redis lock alive while a task runs.
        """
        cancelled = False

        async def renewer():
            nonlocal cancelled
            try:
                while not self._stop_event.is_set():
                    await asyncio.sleep(self.lock_renew_sec)
                    # if lock was deleted, break
                    ttl = await self.middleware.redis.ttl(lock_key)
                    if ttl is None or ttl < 0:
                        # key expired or no TTL -> stop renewing
                        break
                    await self.middleware.redis.expire(lock_key, self.lock_ttl_sec)
            except asyncio.CancelledError:
                cancelled = True

        task = asyncio.create_task(renewer(), name=f"lock-renewer:{lock_key}")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _process_task(self, task_data: Dict[str, Any]):
        task_id = task_data["task_id"]
        session_id = task_data.get("session_id")
        config = task_data.get("config") or {}
        chat_history = task_data.get("chat_history") or []
        message = task_data.get("message") or ""
        acct_dict = task_data.get("acct") or {}
        lock_key = task_data.get("_lock_key")

        # accounting + storage
        from kdcube_ai_app.infra.accounting.envelope import AccountingEnvelope, bind_accounting
        from kdcube_ai_app.infra.accounting import with_accounting

        envelope = AccountingEnvelope.from_dict(acct_dict)
        kdcube_path = task_data.get("kdcube_path") or os.environ.get("KDCUBE_STORAGE_PATH")
        storage_backend = create_storage_backend(kdcube_path, **{})

        # Emit "chat_start" up-front (so UI shows a banner immediately)
        await self._emit("chat_start", {
            "task_id": task_id,
            "message": (message[:100] + "...") if len(message) > 100 else message,
            "timestamp": _utc_now_iso(),
            "user_type": task_data.get("user_type", "unknown"),
            "queue_stats": {},
        }, room=session_id)

        # Also emit a synthetic "workflow_start" step (optional but nice)
        await self._emit_step("workflow_start", "started", session_id, data={
            "model": (config or {}).get("selected_model"),
        })

        try:
            async with bind_accounting(envelope, storage_backend, enabled=True):
                async with with_accounting("chat.orchestrator", metadata={"task_id": task_id}):
                    # Renew the lock while running the job
                    async with self._lock_renewer(lock_key):
                        # Enforce timeout
                        result = await asyncio.wait_for(
                            # self._invoke_handler(task_id, session_id, message, config, chat_history),
                            self._invoke_handler(task_id, session_id, message, config, chat_history),
                            timeout=self.task_timeout_sec,
                        )

            # Emit completion
            await self._emit_result(task_id, session_id, result)

        except asyncio.TimeoutError:
            tb = "Task timed out"
            await self._emit_error(task_id, session_id, tb)
        except Exception:
            tb = traceback.format_exc()
            await self._emit_error(task_id, session_id, tb)
        finally:
            try:
                if lock_key:
                    await self.middleware.redis.delete(lock_key)
            finally:
                self._current_load = max(0, self._current_load - 1)

    async def _invoke_handler(
            self,
            task_id: str,
            session_id: str,
            message: str,
            config: Dict[str, Any],
            chat_history: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calls the chat handler. If your handler supports callbacks or an emitter,
        we pass our emitter so it can stream `chat_delta` etc. through this processor.
        """
        # Best-effort: some handlers accept **kwargs like emitter / task_id
        kwargs = {}
        try:
            kwargs["emitter"] = self._emitter
            kwargs["task_id"] = task_id
        except Exception:
            pass

        result = await self.chat_handler(
            message,
            session_id,
            config,
            chat_history,
            **kwargs,  # ignored by handlers that don't take them
        )
        return result or {}

    # ---------------- Emission helpers ----------------

    async def _emit_step(self, step: str, status: str, room: Optional[str], *, data: Optional[dict] = None, error: Optional[str] = None):
        payload = {
            "step": step,
            "status": status,
            "timestamp": _utc_now_iso(),
            "elapsed_time": None,
            "error": error,
            "data": data or {},
        }
        await self._emit("chat_step", payload, room=room)

    async def _emit_result(self, task_id: str, session_id: str, result: Dict[str, Any]):
        logger.info(f"Task {task_id} completed for session {session_id}")

        # Close the synthetic workflow step
        await self._emit_step("workflow_complete", "completed", session_id)

        # Your handler already returns {"final_answer": ..., "session_id": ...} in your passthrough.
        # Normalize a bit, but don’t force a schema.
        if not result:
            result = {}
        result["task_id"] = task_id
        result["session_id"] = session_id
        result["timestamp"] = _utc_now_iso()

        await self._emit("chat_complete", result, room=session_id)

    async def _emit_error(self, task_id: str, session_id: str, error: str):
        logger.error(f"Task {task_id} failed for session {session_id}: {error}")
        await self._emit_step("workflow_error", "error", session_id, error=error, data={"message": "Workflow aborted"})

        payload = {
            "task_id": task_id,
            "error": error,
            "timestamp": _utc_now_iso(),
        }
        await self._emit("chat_error", payload, room=session_id)

    async def _emit(self, event: str, data: dict, *, room: Optional[str] = None):
        try:
            await self._emitter.emit(event, data, room=room)
        except Exception as e:
            logger.error(f"Emitter error for event '{event}': {e}")
