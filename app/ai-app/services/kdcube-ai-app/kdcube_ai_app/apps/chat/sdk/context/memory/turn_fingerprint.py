# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/context/memory/fingerprint.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import datetime
import json


@dataclass
class FPAssertion:
    key: str
    value: Any = None
    desired: bool = True
    confidence: float = 0.7
    scope: str = "conversation"
    since_ts: str = ""
    source: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FPException:
    key: str
    value: Any = None
    scope: str = "conversation"
    since_ts: str = ""
    source: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FPFact:
    key: str
    value: Any = None
    confidence: float = 0.6
    scope: str = "conversation"
    since_ts: str = ""
    source: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnFingerprintV1:
    version: str
    turn_id: str
    objective: str
    topics: List[str]
    assertions: List[Dict[str, Any]]
    exceptions: List[Dict[str, Any]]
    facts: List[Dict[str, Any]]
    assistant_signals: List[Dict[str, Any]]
    ctx_retrieval_queries: List[Dict[str, Any]]
    made_at: str
    conversation_title: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "turn_id": self.turn_id,
            "objective": self.objective,
            "topics": list(self.topics or []),
            "assertions": list(self.assertions or []),
            "exceptions": list(self.exceptions or []),
            "facts": list(self.facts or []),
            "assistant_signals": list(self.assistant_signals or []),
            "ctx_retrieval_queries": list(self.ctx_retrieval_queries or []),
            "made_at": self.made_at,
            "conversation_title": self.conversation_title,
        }


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def make_early_guess_fingerprint(
        *,
        turn_id: str,
        objective: str,
        topics: List[str],
        guess_prefs: Dict[str, List[Dict[str, Any]]] | None = None,
        guess_facts: List[Dict[str, Any]] | None = None,
        ctx_retrieval_queries: List[Dict[str, Any]] | None = None,
        assistant_signals: List[Dict[str, Any]] | None = None,
) -> TurnFingerprintV1:
    gp = guess_prefs or {}
    return TurnFingerprintV1(
        version="v1",
        turn_id=turn_id,
        objective=objective or "",
        topics=list(topics or []),
        assertions=list(gp.get("assertions") or []),
        exceptions=list(gp.get("exceptions") or []),
        facts=list(guess_facts or []),
        assistant_signals=list(assistant_signals or []),
        ctx_retrieval_queries=list(ctx_retrieval_queries or []),
        made_at=_now_iso(),
        conversation_title="",
    )


def _short(v: Any, max_len: int = 80) -> str:
    try:
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    except Exception:
        s = str(v)
    s = s.strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def render_fingerprint_one_liner(fp: TurnFingerprintV1) -> str:
    obj = (fp.objective or "").strip()
    def _kv(items):
        out = []
        for it in items or []:
            key = it.get("key")
            if not key:
                continue
            val = _short(it.get("value"))
            sev = (it.get("severity") or "").strip()
            scope = (it.get("scope") or "").strip()
            applies = (it.get("applies_to") or "").strip()
            meta = []
            if sev:
                meta.append(sev)
            if scope:
                meta.append(f"scope={scope}")
            if applies:
                meta.append(f"applies_to={applies}")
            meta_txt = f" ({', '.join(meta)})" if meta else ""
            out.append(f"{key}={val}{meta_txt}")
        return out
    a_vals = _kv(fp.assertions or [])
    e_vals = _kv(fp.exceptions or [])
    f_vals = _kv(fp.facts or [])
    parts = [
        f"objective={obj}",
        f"A={a_vals[:6]}",
        f"E={e_vals[:6]}",
        f"F={f_vals[:6]}",
    ]
    if fp.topics:
        parts.append(f"topics={list(fp.topics)[:4]}")
    if fp.assistant_signals:
        keys = []
        for s in fp.assistant_signals or []:
            k = (s.get("key") or "").strip()
            if k and k not in keys:
                keys.append(k)
        if keys:
            parts.append(f"assistant_signals={keys[:3]}")
    return "; ".join(parts)
