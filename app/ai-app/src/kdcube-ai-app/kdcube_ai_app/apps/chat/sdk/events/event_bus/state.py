# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from kdcube_ai_app.apps.chat.sdk.events.event_bus.exceptions import EventLaneStateLockTimeout
from kdcube_ai_app.apps.chat.sdk.events.semantics import event_is_active_turn_control


_STATE_TTL_SECONDS = 7 * 24 * 3600
_LOCK_TTL_SECONDS = 10

logger = logging.getLogger(__name__)


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _decode(raw: Any) -> Any:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return raw


def _parse_json(raw: Any) -> dict[str, Any]:
    raw = _decode(raw)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
            return dict(data) if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


async def _complete_task_despite_cancellation(
    task: asyncio.Task,
) -> tuple[Any, BaseException | None, BaseException | None]:
    """Wait for a Redis command to reach a known outcome before propagating cancellation."""

    cancelled: BaseException | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
    try:
        return task.result(), cancelled, None
    except BaseException as exc:
        return None, cancelled, exc


def normalize_timestamp(value: Any) -> str:
    if value is None:
        return ""
    text = str(value or "").strip()
    if not text:
        return ""
    return text


def _timestamp_epoch(value: Any) -> float:
    text = normalize_timestamp(value)
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        pass
    try:
        parse_text = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = _dt.datetime.fromisoformat(parse_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return 0.0


def later_timestamp(left: Any, right: Any) -> str:
    left_text = normalize_timestamp(left)
    right_text = normalize_timestamp(right)
    if not left_text:
        return right_text
    if not right_text:
        return left_text
    if _timestamp_epoch(right_text) >= _timestamp_epoch(left_text):
        return right_text
    return left_text


def timestamp_lte(left: Any, right: Any) -> bool:
    left_text = normalize_timestamp(left)
    right_text = normalize_timestamp(right)
    if not left_text or not right_text:
        return False
    return _timestamp_epoch(left_text) <= _timestamp_epoch(right_text)


def timestamp_lt(left: Any, right: Any) -> bool:
    left_text = normalize_timestamp(left)
    right_text = normalize_timestamp(right)
    if not right_text:
        return False
    if not left_text:
        return True
    return _timestamp_epoch(left_text) < _timestamp_epoch(right_text)


def timestamp_age_ms(*, now: Any, since: Any) -> float:
    now_text = normalize_timestamp(now)
    since_text = normalize_timestamp(since)
    if not now_text or not since_text:
        return float("inf")
    return max(0.0, (_timestamp_epoch(now_text) - _timestamp_epoch(since_text)) * 1000.0)


def timestamp_is_fresh(*, now: Any, since: Any, ttl_ms: int) -> bool:
    ttl = max(0, int(ttl_ms or 0))
    if ttl <= 0:
        return False
    return timestamp_age_ms(now=now, since=since) <= ttl


def event_timestamp(event: Any) -> str:
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict):
        accepted = payload.get("event")
        if isinstance(accepted, dict):
            ts = normalize_timestamp(accepted.get("timestamp") or accepted.get("ts"))
            if ts:
                return ts
    task_payload = getattr(event, "task_payload", None)
    if isinstance(task_payload, dict):
        request = task_payload.get("request")
        if isinstance(request, dict):
            for item in request.get("external_events") or []:
                if isinstance(item, dict):
                    ts = normalize_timestamp(item.get("timestamp") or item.get("ts"))
                    if ts:
                        return ts
    return normalize_timestamp(getattr(event, "created_at", None))


def event_id(event: Any) -> str:
    text = str(getattr(event, "message_id", "") or getattr(event, "event_id", "") or "").strip()
    if text:
        return text
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict):
        accepted = payload.get("event")
        if isinstance(accepted, dict):
            text = str(accepted.get("event_id") or accepted.get("message_id") or "").strip()
            if text:
                return text
    task_payload = getattr(event, "task_payload", None)
    if isinstance(task_payload, dict):
        event_meta = task_payload.get("event")
        if isinstance(event_meta, dict):
            text = str(event_meta.get("event_id") or event_meta.get("message_id") or "").strip()
            if text:
                return text
    return ""


def event_is_reactive(event: Any) -> bool:
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict):
        accepted = payload.get("event")
        if isinstance(accepted, dict) and accepted.get("reactive") is not None:
            return bool(accepted.get("reactive"))
    task_payload = getattr(event, "task_payload", None)
    if isinstance(task_payload, dict):
        request = task_payload.get("request")
        if isinstance(request, dict):
            for item in request.get("external_events") or []:
                if isinstance(item, dict) and item.get("reactive") is not None:
                    return bool(item.get("reactive"))
    return bool(getattr(event, "is_continuation", False))


@dataclass
class EventLaneState:
    handler_turn_id: str = ""
    handler_status: str = ""
    handler_status_at: str = ""
    last_processed_reactive_event_timestamp: str = ""
    last_processed_event_timestamp: str = ""
    last_processed_event_id: str = ""
    consumer_turn_id: str = ""
    consumer_status: str = ""
    consumer_status_at: str = ""

    @classmethod
    def from_any(cls, raw: Any) -> "EventLaneState":
        data = _parse_json(raw)
        if not data:
            return cls()
        fields = {name: data.get(name) for name in cls.__dataclass_fields__.keys()}
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_present(self) -> bool:
        return self.handler_status in {"open", "closed"}

    def is_open_for(self, turn_id: str) -> bool:
        return self.handler_status == "open" and bool(turn_id) and self.handler_turn_id == str(turn_id or "")

    def event_was_processed(self, event: Any) -> bool:
        ts = event_timestamp(event)
        if event_is_reactive(event):
            return timestamp_lte(ts, self.last_processed_reactive_event_timestamp)
        return timestamp_lte(ts, self.last_processed_event_timestamp)


def wake_ignore_reason(event: Any, state: "EventLaneState") -> str:
    """Promote-only-if-unconsumed: why a lane wakeup for ``event`` must be
    acked instead of promoted. ``""`` means promote.

    A live turn that folded the event recorded consumption on the event
    itself (``consumed_at``, via ``mark_consumed_up_to``) and advanced the
    lane's processed-event cursors — either record is enough to ack, so a
    promotable event starts at most one turn (exactly-once)."""
    if getattr(event, "consumed_at", None) is not None:
        return "event_already_consumed"
    if getattr(event, "promoted_at", None) is not None:
        return "event_already_promoted"
    if getattr(event, "failed_at", None) is not None:
        return "event_failed"
    if event_is_active_turn_control(event):
        return "active_turn_control_not_promotable"
    if timestamp_lte(event_timestamp(event), state.last_processed_reactive_event_timestamp):
        return "wake_already_processed"
    if state.event_was_processed(event):
        return "wake_already_processed"
    return ""


class RedisEventLaneStateTable:
    """Redis-backed event-lane coordination record.

    The logical table row is stored as one JSON value under ``state_key``. The
    short-lived lock key serializes updates between ingress, proc, and runtime
    readers.
    """

    def __init__(
        self,
        *,
        redis: Any,
        state_key: str,
        lock_key: Optional[str] = None,
        ttl_seconds: int = _STATE_TTL_SECONDS,
        lock_ttl_seconds: int = _LOCK_TTL_SECONDS,
    ) -> None:
        self.redis = redis
        self.state_key = str(state_key or "")
        self.lock_key = str(lock_key or f"{self.state_key}:lock")
        self.ttl_seconds = max(1, int(ttl_seconds or _STATE_TTL_SECONDS))
        self.lock_ttl_seconds = max(1, int(lock_ttl_seconds or _LOCK_TTL_SECONDS))

    @classmethod
    def for_source(cls, source: Any) -> "RedisEventLaneStateTable":
        return cls(
            redis=getattr(source, "redis"),
            state_key=f"{getattr(source, 'log_key')}:state",
            lock_key=f"{getattr(source, 'log_key')}:state:lock",
        )

    async def get(self) -> EventLaneState:
        raw = await self.redis.get(self.state_key)
        return EventLaneState.from_any(raw)

    async def put(self, state: EventLaneState) -> EventLaneState:
        payload = json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True)
        task = asyncio.create_task(
            self._put_payload(payload),
            name=f"event-lane-state-put:{self.state_key}",
        )
        _result, cancelled, error = await _complete_task_despite_cancellation(task)
        if error is not None:
            if cancelled is not None:
                raise cancelled from error
            raise error
        if cancelled is not None:
            raise cancelled
        return state

    async def _put_payload(self, payload: str) -> None:
        setter = getattr(self.redis, "set", None)
        if callable(setter):
            try:
                await setter(self.state_key, payload, ex=self.ttl_seconds)
                return
            except TypeError:
                await setter(self.state_key, payload)
                return
        await self.redis.setex(self.state_key, self.ttl_seconds, payload)

    @contextlib.asynccontextmanager
    async def lock(self, *, timeout_seconds: float = 2.0, operation: str = "unspecified"):
        operation = str(operation or "unspecified")
        current_task = asyncio.current_task()
        ownership_token = f"lock_{uuid.uuid4().hex}"
        token = ""
        deadline = time.monotonic() + max(0.05, float(timeout_seconds or 2.0))
        acquired = False
        while time.monotonic() < deadline:
            token = json.dumps(
                {
                    "token": ownership_token,
                    "operation": operation,
                    "pid": os.getpid(),
                    "task": current_task.get_name() if current_task is not None else "",
                    "acquired_at": utc_timestamp(),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            acquire_task = asyncio.create_task(
                self._acquire_lock_once(token),
                name=f"event-lane-lock-acquire:{operation}:{self.lock_key}",
            )
            result, cancelled, error = await _complete_task_despite_cancellation(acquire_task)
            if error is not None or cancelled is not None:
                await self._cleanup_uncertain_acquire(token=token, operation=operation)
                if cancelled is not None:
                    logger.warning(
                        "[event-bus.lock] acquisition cancelled after Redis command "
                        "operation=%s state_key=%s lock_key=%s command_result=%s command_error=%s",
                        operation,
                        self.state_key,
                        self.lock_key,
                        bool(result),
                        type(error).__name__ if error is not None else "none",
                    )
                    if error is not None:
                        raise cancelled from error
                    raise cancelled
                raise error
            acquired = bool(result)
            if acquired:
                break
            await asyncio.sleep(0.01)
        if not acquired:
            diagnostics = await self._lock_diagnostics()
            error = EventLaneStateLockTimeout(
                state_key=self.state_key,
                lock_key=self.lock_key,
                operation=operation,
                timeout_seconds=timeout_seconds,
                holder_operation=str(diagnostics.get("operation") or ""),
                holder_pid=str(diagnostics.get("pid") or ""),
                holder_task=str(diagnostics.get("task") or ""),
                holder_acquired_at=str(diagnostics.get("acquired_at") or ""),
                pttl_ms=diagnostics.get("pttl_ms"),
            )
            logger.warning("[event-bus.lock] %s", error)
            raise error
        renew_task = asyncio.create_task(self._renew_lock(token), name=f"event-lane-lock-renew:{self.lock_key}")
        try:
            yield token
        finally:
            owner_task = asyncio.current_task()
            cancellation_count = owner_task.cancelling() if owner_task is not None else 0
            shutdown_cancelled: BaseException | None = None
            renew_task.cancel()
            try:
                await renew_task
            except asyncio.CancelledError as exc:
                if owner_task is not None and owner_task.cancelling() > cancellation_count:
                    shutdown_cancelled = exc
            release_task = asyncio.create_task(
                self._release_lock(token),
                name=f"event-lane-lock-release:{operation}:{self.lock_key}",
            )
            released, cancelled, error = await _complete_task_despite_cancellation(release_task)
            if error is not None:
                logger.error(
                    "[event-bus.lock] release failed operation=%s state_key=%s lock_key=%s",
                    operation,
                    self.state_key,
                    self.lock_key,
                    exc_info=(type(error), error, error.__traceback__),
                )
            elif not released:
                logger.warning(
                    "[event-bus.lock] release lost ownership operation=%s state_key=%s lock_key=%s",
                    operation,
                    self.state_key,
                    self.lock_key,
                )
            if cancelled is not None:
                if error is not None:
                    raise cancelled from error
                raise cancelled
            if shutdown_cancelled is not None:
                if error is not None:
                    raise shutdown_cancelled from error
                raise shutdown_cancelled
            if error is not None:
                raise error

    async def update(
        self,
        mutator: Callable[[EventLaneState], EventLaneState | None],
        *,
        operation: str = "update",
    ) -> EventLaneState:
        async with self.lock(operation=operation):
            state = await self.get()
            new_state = mutator(state) or state
            await self.put(new_state)
            return new_state

    async def _acquire_lock_once(self, token: str) -> bool:
        setter = getattr(self.redis, "set", None)
        if not callable(setter):
            return False
        try:
            return bool(await setter(self.lock_key, token, nx=True, ex=self.lock_ttl_seconds))
        except TypeError:
            if await self.redis.get(self.lock_key) is not None:
                return False
            await setter(self.lock_key, token)
            return True

    async def _cleanup_uncertain_acquire(self, *, token: str, operation: str) -> None:
        release_task = asyncio.create_task(
            self._release_lock(token),
            name=f"event-lane-lock-uncertain-release:{operation}:{self.lock_key}",
        )
        released, _cancelled, error = await _complete_task_despite_cancellation(release_task)
        if error is not None:
            logger.error(
                "[event-bus.lock] uncertain acquisition cleanup failed operation=%s state_key=%s lock_key=%s",
                operation,
                self.state_key,
                self.lock_key,
                exc_info=(type(error), error, error.__traceback__),
            )
        elif released:
            logger.info(
                "[event-bus.lock] cleaned lock after uncertain acquisition operation=%s state_key=%s lock_key=%s",
                operation,
                self.state_key,
                self.lock_key,
            )

    async def _lock_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        try:
            diagnostics.update(_parse_json(await self.redis.get(self.lock_key)))
        except Exception:
            pass
        try:
            pttl = getattr(self.redis, "pttl", None)
            if callable(pttl):
                diagnostics["pttl_ms"] = int(await pttl(self.lock_key))
            else:
                ttl = getattr(self.redis, "ttl", None)
                if callable(ttl):
                    diagnostics["pttl_ms"] = int(await ttl(self.lock_key)) * 1000
        except Exception:
            diagnostics["pttl_ms"] = None
        return diagnostics

    async def _renew_lock(self, token: str) -> None:
        interval = max(0.1, min(1.0, float(self.lock_ttl_seconds) / 3.0))
        while True:
            await asyncio.sleep(interval)
            renewed = await self._renew_lock_once(token)
            if not renewed:
                return

    async def _renew_lock_once(self, token: str) -> bool:
        evaluator = getattr(self.redis, "eval", None)
        if callable(evaluator):
            result = await evaluator(
                """
                if redis.call('GET', KEYS[1]) == ARGV[1] then
                    return redis.call('EXPIRE', KEYS[1], ARGV[2])
                end
                return 0
                """,
                1,
                self.lock_key,
                token,
                str(self.lock_ttl_seconds),
            )
            return bool(result)
        current = _decode(await self.redis.get(self.lock_key))
        if str(current or "") != token:
            return False
        expirer = getattr(self.redis, "expire", None)
        if callable(expirer):
            return bool(await expirer(self.lock_key, self.lock_ttl_seconds))
        return True

    async def _release_lock(self, token: str) -> bool:
        evaluator = getattr(self.redis, "eval", None)
        if callable(evaluator):
            result = await evaluator(
                """
                if redis.call('GET', KEYS[1]) == ARGV[1] then
                    return redis.call('DEL', KEYS[1])
                end
                return 0
                """,
                1,
                self.lock_key,
                token,
            )
            return bool(result)
        current = _decode(await self.redis.get(self.lock_key))
        if str(current or "") == token:
            await self.redis.delete(self.lock_key)
            return True
        return False
