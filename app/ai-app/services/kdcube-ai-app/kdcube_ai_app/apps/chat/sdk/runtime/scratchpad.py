# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/runtime/scratchpad.py

from __future__ import annotations
import asyncio, copy
from typing import List, Literal, Optional, Dict, Any, Iterable
from pydantic import BaseModel, Field
from datetime import datetime
import json

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
        self.timings = []
        self.user_message: Optional[str] = None
        self.proposed_facts: List[Dict[str, Any]] = []
        self.exceptions: List[Dict[str, Any]] = []
        self.short_artifacts: List[Dict[str, Any]] = []
        self.clarification_questions: List[str] = []
        self.user_shortcuts: List[str] = []
        self.extracted_prefs: Dict[str, Any] = {"assertions": [], "exceptions": []}

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

LogArea = Literal["objective", "user", "attachments", "solver", "answer", "note", "summary"]
LogLevel = Literal["info", "warn", "error"]

class TurnLogEntry(BaseModel):
    t: str = Field(..., description="time-part like HH:MM:SS")
    area: LogArea
    msg: str
    level: LogLevel = "info"
    data: Optional[Dict[str, Any]] = None

    def to_line(self) -> str:
        base = f"{self.t} [{self.area}] {self.msg}"
        if self.level != "info": base += f"  !{self.level}"
        return base

class TurnLog(BaseModel):
    user_id: str
    conversation_id: str
    turn_id: str
    started_at_iso: str
    ended_at_iso: Optional[str] = None
    entries: List[TurnLogEntry] = Field(default_factory=list)

    def _nowt(self) -> str:
        return datetime.utcnow().strftime("%H:%M:%S")

    def add(self, area: LogArea, msg: str, *, level: LogLevel="info", data: Dict[str, Any] | None=None):
        self.entries.append(TurnLogEntry(t=self._nowt(), area=area, msg=msg, level=level, data=data or {}))

    # convenience shorthands
    def objective(self, msg: str, **kw): self.add("objective", msg, **kw)
    def user(self, msg: str, **kw): self.add("user", msg, **kw)
    def attachments(self, msg: str, **kw): self.add("attachments", msg, **kw)
    def solver(self, msg: str, **kw): self.add("solver", msg, **kw)
    def answer(self, msg: str, **kw): self.add("answer", msg, **kw)
    def turn_summary(self, turn_summary: dict, **kw):
        try:
            if isinstance(turn_summary, dict):
                if turn_summary["objective"]:
                    self.objective(turn_summary["objective"])
                if turn_summary.get("done"):
                    self.solver("done: " + "; ".join(map(str, turn_summary["done"][:6])))
                if turn_summary.get("not_done"):
                    self.solver("open: " + "; ".join(map(str, turn_summary["not_done"][:6])))
                if turn_summary.get("assumptions"):
                    self.note("assumptions: " + "; ".join(map(str, turn_summary["assumptions"][:6])))
                if turn_summary.get("risks"):
                    self.note("risks: " + "; ".join(map(str, turn_summary["risks"][:6])))
                if turn_summary.get("notes"):
                    self.answer(turn_summary["notes"])
                if turn_summary.get("prefs"):
                    turn_prefs = turn_summary["prefs"]
                    try:
                        compact_prefs = []
                        for a in (turn_prefs.get("assertions") or [])[:6]:
                            compact_prefs.append(f"{a.get('key')}={a.get('value')} {'(avoid)' if not a.get('desired', True) else ''}")
                        for e in (turn_prefs.get("exceptions") or [])[:3]:
                            compact_prefs.append(f"EXC[{e.get('rule_key')}]: {e.get('value')}")
                        if compact_prefs:
                            self.note("prefs: " + "; ".join(compact_prefs))
                    except Exception:
                        pass
        except Exception:
            pass

    def note(self, msg: str, **kw): self.add("note", msg, **kw)

    @property
    def user_entry(self):
        return next((d.model_dump() for d in self.entries if d.area == "user"), None)

    @property
    def objective_entry(self):
        return next((d.model_dump() for d in self.entries if d.area == "objective"), None)

    def to_markdown(self, header: str="[turn_log]") -> str:
        lines = [header]
        lines += [e.to_line() for e in self.entries]
        return "\n".join(lines)

    def to_payload(self) -> Dict[str, Any]:
        return json.loads(self.model_dump_json())

def new_turn_log(user_id: str, conversation_id: str, turn_id: str) -> TurnLog:
    return TurnLog(user_id=user_id, conversation_id=conversation_id, turn_id=turn_id, started_at_iso=datetime.utcnow().isoformat()+"Z")

