# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/runtime/scratchpad.py

from __future__ import annotations
import asyncio, copy
from typing import Any, Dict, List, Iterable

class SharedScratchpad:
    """
    Per-turn, in-memory shared pad:
      - "turn": coordinator-initialized context (summary_ctx, policy_summary, topics, etc.)
      - "workers": { <section>: { <key>: <value>, ... } }
    Workers only write to their own section.
    No persistence here — the coordinator persists aggregated facts/exceptions/artifacts via TurnScratchpad.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._data: Dict[str, Any] = {"turn": {}, "workers": {}}

    async def init_turn(self, **turn_fields):
        async with self._lock:
            self._data["turn"] = {**(self._data.get("turn") or {}), **turn_fields}

    def _ensure_section_unlocked(self, section: str) -> Dict[str, Any]:
        w = self._data.setdefault("workers", {})
        return w.setdefault(section, {})

    async def write(self, section: str, **kv) -> None:
        async with self._lock:
            sect = self._ensure_section_unlocked(section)
            sect.update(kv)

    async def append_list(self, section: str, key: str, items: Iterable[Any]) -> None:
        async with self._lock:
            sect = self._ensure_section_unlocked(section)
            arr = sect.setdefault(key, [])
            arr.extend(list(items or []))

    async def read(self, section: str, *keys: str) -> Dict[str, Any]:
        async with self._lock:
            if section == "turn":
                src = self._data.get("turn") or {}
            else:
                src = (self._data.get("workers") or {}).get(section) or {}
            if not keys:
                return copy.deepcopy(src)
            return {k: copy.deepcopy(src[k]) for k in keys if k in src}

    async def have_keys(self, section: str, *keys: str) -> bool:
        async with self._lock:
            if section == "turn":
                src = self._data.get("turn") or {}
            else:
                src = (self._data.get("workers") or {}).get(section) or {}
            return all(k in src for k in keys)

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return copy.deepcopy(self._data)


class TurnScratchpad:
    def __init__(self):
        self.proposed_facts: List[Dict[str, Any]] = []
        self.exceptions: List[Dict[str, Any]] = []
        self.short_artifacts: List[Dict[str, Any]] = []
        self.clarification_questions: List[str] = []
        self.user_shortcuts: List[str] = []

    def propose_fact(
            self,
            *,
            key: str,
            value: Any,
            desired: bool = True,
            scope: str = "conversation",
            confidence: float = 0.6,
            ttl_days: int = 365,
            reason: str = "turn-proposed",
    ):
        self.proposed_facts.append({
            "key": key,
            "value": value,
            "desired": bool(desired),
            "scope": scope,
            "confidence": float(confidence),
            "ttl_days": int(ttl_days),
            "reason": reason,
        })

    def add_exception(self, *, rule_key: str, value: Any, scope: str = "conversation", reason: str = "turn-exception"):
        self.exceptions.append({"rule_key": rule_key, "value": value, "scope": scope, "reason": reason})

    def add_artifact(self, *, kind: str, title: str, content: str):
        self.short_artifacts.append({"kind": kind, "title": title, "content": content})
