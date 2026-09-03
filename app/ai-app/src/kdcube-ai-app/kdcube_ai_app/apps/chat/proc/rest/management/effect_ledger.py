# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Shared idempotency ledger for effects admitted by Connection Hub."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from kdcube_ai_app.apps.chat.proc.rest.management.contracts import EFFECT_SCHEMA

ACTION_EXECUTE = "execute"
ACTION_REPLAY = "replay"
ACTION_CONFLICT = "conflict"
ACTION_PENDING = "pending"
ACTION_UNKNOWN = "unknown"

STATE_STARTED = "effect_started"
STATE_COMPLETED = "effect_completed"
STATE_FAILED = "effect_failed"


@dataclass(frozen=True)
class EffectReservation:
    action: str
    owner: str = ""
    record: Mapping[str, Any] | None = None


class EffectLedger(Protocol):
    async def reserve(
        self,
        *,
        access_id: str,
        resource: str,
        operation: str,
        invocation_id: str,
        request_digest: str,
        audit: Mapping[str, Any],
    ) -> EffectReservation: ...

    async def finish(
        self,
        *,
        access_id: str,
        resource: str,
        operation: str,
        invocation_id: str,
        request_digest: str,
        owner: str,
        status_code: int,
        response: Mapping[str, Any],
        failed: bool = False,
    ) -> None: ...


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class RedisEffectLedger:
    _FINISH_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local current = cjson.decode(raw)
if current['state'] ~= ARGV[1] then return -1 end
if current['owner'] ~= ARGV[2] then return -2 end
if current['request_digest'] ~= ARGV[3] then return -3 end
redis.call('SET', KEYS[1], ARGV[4])
return 1
"""

    def __init__(
        self,
        redis: Any,
        *,
        tenant: str,
        project: str,
        pending_seconds: float = 120.0,
        clock: Any = time.time,
    ) -> None:
        self._redis = redis
        self._tenant = str(tenant).strip()
        self._project = str(project).strip()
        self._pending_seconds = max(1.0, float(pending_seconds))
        self._clock = clock

    def _key(
        self,
        *,
        access_id: str,
        resource: str,
        operation: str,
        invocation_id: str,
    ) -> str:
        identity = hashlib.sha256(
            f"{access_id}\n{resource}\n{operation}\n{invocation_id}".encode("utf-8")
        ).hexdigest()
        return (
            "kdcube:management:effects:"
            f"{self._tenant}:{self._project}:{identity}"
        )

    @staticmethod
    def _decode(value: Any) -> dict[str, Any]:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping) or parsed.get("schema") != EFFECT_SCHEMA:
            raise RuntimeError("management effect record is invalid")
        return dict(parsed)

    async def reserve(
        self,
        *,
        access_id: str,
        resource: str,
        operation: str,
        invocation_id: str,
        request_digest: str,
        audit: Mapping[str, Any],
    ) -> EffectReservation:
        key = self._key(
            access_id=access_id,
            resource=resource,
            operation=operation,
            invocation_id=invocation_id,
        )
        moment = float(self._clock())
        owner = uuid.uuid4().hex
        record = {
            "schema": EFFECT_SCHEMA,
            "access_id": access_id,
            "resource": resource,
            "operation": operation,
            "invocation_id": invocation_id,
            "request_digest": request_digest,
            "state": STATE_STARTED,
            "owner": owner,
            "started_at": moment,
            "audit": dict(audit),
        }
        created = await self._redis.set(key, _json(record), nx=True)
        if created:
            return EffectReservation(ACTION_EXECUTE, owner=owner, record=record)

        raw = await self._redis.get(key)
        if raw is None:
            return await self.reserve(
                access_id=access_id,
                resource=resource,
                operation=operation,
                invocation_id=invocation_id,
                request_digest=request_digest,
                audit=audit,
            )
        current = self._decode(raw)
        if current.get("request_digest") != request_digest:
            return EffectReservation(ACTION_CONFLICT, record=current)
        if current.get("state") in {STATE_COMPLETED, STATE_FAILED}:
            return EffectReservation(ACTION_REPLAY, record=current)
        age = max(0.0, moment - float(current.get("started_at") or 0.0))
        action = ACTION_PENDING if age < self._pending_seconds else ACTION_UNKNOWN
        return EffectReservation(action, record=current)

    async def finish(
        self,
        *,
        access_id: str,
        resource: str,
        operation: str,
        invocation_id: str,
        request_digest: str,
        owner: str,
        status_code: int,
        response: Mapping[str, Any],
        failed: bool = False,
    ) -> None:
        key = self._key(
            access_id=access_id,
            resource=resource,
            operation=operation,
            invocation_id=invocation_id,
        )
        raw = await self._redis.get(key)
        if raw is None:
            raise RuntimeError("management effect reservation is missing")
        current = self._decode(raw)
        completed = {
            **current,
            "state": STATE_FAILED if failed else STATE_COMPLETED,
            "status_code": int(status_code),
            "response": dict(response),
            "completed_at": float(self._clock()),
        }
        result = await self._redis.eval(
            self._FINISH_SCRIPT,
            1,
            key,
            STATE_STARTED,
            owner,
            request_digest,
            _json(completed),
        )
        if int(result or 0) != 1:
            raise RuntimeError("management effect reservation moved before settlement")


__all__ = [
    "ACTION_CONFLICT",
    "ACTION_EXECUTE",
    "ACTION_PENDING",
    "ACTION_REPLAY",
    "ACTION_UNKNOWN",
    "EffectLedger",
    "EffectReservation",
    "RedisEffectLedger",
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_STARTED",
]
