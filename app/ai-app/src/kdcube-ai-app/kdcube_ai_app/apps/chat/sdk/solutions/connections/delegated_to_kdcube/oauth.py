# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""OAuth helpers for Connection Hub delegated to KDCube."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Mapping


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_json(data: Mapping[str, Any]) -> str:
    return _b64url(json.dumps(dict(data), sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _unb64url_json(data: str) -> dict[str, Any]:
    padded = data + ("=" * (-len(data) % 4))
    raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("OAuth state payload is invalid")
    return parsed


def state_digest(state: str) -> str:
    return hashlib.sha256(str(state or "").encode("utf-8")).hexdigest()


def sign_state(payload: Mapping[str, Any], secret: str) -> str:
    if not str(secret or "").strip():
        raise ValueError("OAuth state secret is not configured")
    encoded = _b64url_json(payload)
    signature = hmac.new(str(secret).encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def verify_state(state: str, secret: str) -> dict[str, Any]:
    raw = str(state or "").strip()
    if "." not in raw:
        raise ValueError("OAuth state is invalid")
    encoded, received_signature = raw.rsplit(".", 1)
    expected = hmac.new(str(secret).encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(received_signature, expected):
        raise ValueError("OAuth state signature is invalid")
    payload = _unb64url_json(encoded)
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("OAuth state expired")
    return payload


def peek_state_payload(state: str) -> dict[str, Any]:
    raw = str(state or "").strip()
    if "." not in raw:
        raise ValueError("OAuth state is invalid")
    encoded, _signature = raw.rsplit(".", 1)
    return _unb64url_json(encoded)


class OAuthStateStore:
    async def put(self, state: str, payload: Mapping[str, Any], *, ttl_seconds: int) -> None:
        raise NotImplementedError

    async def pop(self, state: str) -> dict[str, Any] | None:
        raise NotImplementedError


class MemoryOAuthStateStore(OAuthStateStore):
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    async def put(self, state: str, payload: Mapping[str, Any], *, ttl_seconds: int) -> None:
        del ttl_seconds
        self.items[state_digest(state)] = dict(payload)

    async def pop(self, state: str) -> dict[str, Any] | None:
        return self.items.pop(state_digest(state), None)


class RedisOAuthStateStore(OAuthStateStore):
    def __init__(self, redis: Any, *, prefix: str) -> None:
        self.redis = redis
        self.prefix = str(prefix or "kdcube:connection-hub:delegated-to-kdcube:oauth-state").strip(":")

    def key(self, state: str) -> str:
        return f"{self.prefix}:{state_digest(state)}"

    async def put(self, state: str, payload: Mapping[str, Any], *, ttl_seconds: int) -> None:
        await self.redis.set(self.key(state), json.dumps(dict(payload), sort_keys=True, ensure_ascii=True), ex=int(ttl_seconds or 900))

    async def pop(self, state: str) -> dict[str, Any] | None:
        key = self.key(state)
        raw = await self.redis.get(key)
        await self.redis.delete(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            parsed = json.loads(str(raw))
        except Exception:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None


async def create_oauth_state(
    state_store: OAuthStateStore,
    *,
    secret: str,
    user_id: str,
    provider_id: str,
    connector_app_id: str,
    claims: tuple[str, ...],
    account_id: str = "",
    claims_mode: str = "add",
    return_hint: str = "",
    source: str = "connection_hub_widget",
    ttl_seconds: int = 900,
) -> dict[str, Any]:
    now = int(time.time())
    payload = {
        "v": 1,
        "user_id": str(user_id or "").strip(),
        "provider_id": str(provider_id or "").strip(),
        "connector_app_id": str(connector_app_id or "").strip(),
        "claims": [str(item) for item in claims],
        "account_id": str(account_id or "").strip(),
        "claims_mode": str(claims_mode or "add").strip().lower() or "add",
        "nonce": uuid.uuid4().hex,
        "source": str(source or "").strip() or "connection_hub_widget",
        "return_hint": str(return_hint or "").strip(),
        "iat": now,
        "exp": now + int(ttl_seconds or 900),
    }
    if not payload["user_id"]:
        raise ValueError("OAuth state requires user_id")
    if not payload["provider_id"]:
        raise ValueError("OAuth state requires provider_id")
    if not payload["connector_app_id"]:
        raise ValueError("OAuth state requires connector_app_id")
    state = sign_state(payload, secret)
    await state_store.put(state, payload, ttl_seconds=ttl_seconds)
    return {"state": state, "payload": payload, "state_id": state_digest(state)}


async def consume_oauth_state(
    state_store: OAuthStateStore,
    *,
    state: str,
    secret: str,
) -> dict[str, Any]:
    payload = verify_state(state, secret)
    stored = await state_store.pop(state)
    if stored is None:
        raise ValueError("OAuth state was not found or already used")
    if str(stored.get("nonce") or "") != str(payload.get("nonce") or ""):
        raise ValueError("OAuth state storage mismatch")
    return payload


__all__ = [
    "MemoryOAuthStateStore",
    "OAuthStateStore",
    "RedisOAuthStateStore",
    "consume_oauth_state",
    "create_oauth_state",
    "peek_state_payload",
    "sign_state",
    "state_digest",
    "verify_state",
]
