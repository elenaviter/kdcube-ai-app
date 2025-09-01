# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/policy/policy.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, TypedDict, Optional
import time
import json
from pydantic import BaseModel, Field

from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass
class KeyPolicy:
    # promotion thresholds
    min_support: int = 2              # distinct conversations needed
    avg_decayed: float = 0.7          # average decayed confidence threshold
    distinct_days: int = 2            # minimum unique days among supports
    conflict_horizon_days: int = 45   # recent opposing evidence blocks promotion
    ttl_days_user: int = 365          # TTL when promoted to user scope
    half_life_days: float = 45.0      # decay half-life for evidence

    # data semantics
    numeric_tolerance: float = 0.05   # 5% relative tolerance for equivalence
    canonicalizer: Optional[Callable[[Any], Any]] = None  # optional value normalizer

    # privacy/visibility (controls what goes into LLM prompts)
    send_to_llm: bool = True


class PrefAssertion(BaseModel):
    key: str
    value: Any = None
    desired: bool = True
    scope: str = "conversation"
    confidence: float = Field(0.6, ge=0.0, le=1.0)
    reason: str = "nl-extracted"

class PrefException(BaseModel):
    rule_key: str
    value: Any = True
    scope: str = "conversation"
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    reason: str = "nl-extracted"

class PreferenceExtractionOut(BaseModel):
    assertions: List[PrefAssertion] = Field(default_factory=list)
    exceptions: List[PrefException] = Field(default_factory=list)

# exception/neg/pos precedence across scopes
PRECEDENCE = [
    ("exception", "conversation"),
    ("exception", "user"),
    ("neg", "conversation"),
    ("pos", "conversation"),
    ("neg", "user"),
    ("pos", "user"),
    ("neg", "global"),
    ("pos", "global"),
]

@dataclass
class Item:
    key: str
    value: Any
    scope: str
    desired: Optional[bool]  # None for exception
    confidence: float
    created_at: int
    ttl_days: int
    reason: str

def _decay_score(conf: float, created_at: int, hl_days: float = 30.0) -> float:
    age_days = max(0.0, (time.time() - created_at) / 86400.0)
    return float(conf) * (0.5 ** (age_days / hl_days))

class PolicyResult(TypedDict):
    do: Dict[str, Any]
    avoid: Dict[str, Any]
    allow_if: Dict[str, Any]
    superseded: List[Dict[str, Any]]
    kept: int
    dropped: int
    reasons: List[str]

def _kind(item: Item) -> Tuple[str, str]:
    if item.desired is None:
        return ("exception", item.scope)
    return ("pos" if item.desired else "neg", item.scope)

def _norm_scope(s: Optional[str]) -> str:
    s = (s or "user").lower().strip()
    if s in ("conversation", "session", "conv", "thread"):
        return "conversation"
    return s

def _unpack_val(rec: Dict[str, Any]) -> Any:
    v = rec.get("value")
    if v is None and rec.get("value_json"):
        try:
            return json.loads(rec["value_json"])
        except Exception:
            return None
    return v

def evaluate_policy(raw: Dict[str, Any], *, half_life_days: float = 30.0) -> PolicyResult:
    """
    Apply precedence + conflict resolution.
    - exception > conversation-neg > conversation-pos > user-neg > user-pos > global-neg > global-pos
    - for the same key, choose by (decayed_score, confidence, recency)
    """
    assertions = [
        Item(
            key=a.get("key"),
            value=_unpack_val(a),
            scope=_norm_scope(a.get("scope")),
            desired=bool(a.get("desired")),
            confidence=float(a.get("confidence") or 0.5),
            created_at=int(a.get("created_at") or 0),
            ttl_days=int(a.get("ttl_days") or 365),
            reason=a.get("reason") or "unknown",
        )
        for a in (raw.get("assertions") or [])
    ]
    exceptions = [
        Item(
            key=e.get("rule_key"),
            value=_unpack_val(e),
            scope=_norm_scope(e.get("scope")),
            desired=None,
            confidence=1.0,
            created_at=int(e.get("created_at") or 0),
            ttl_days=365,
            reason=e.get("reason") or "exception",
        )
        for e in (raw.get("exceptions") or [])
    ]

    bucket: Dict[str, List[Item]] = {}
    for it in assertions + exceptions:
        bucket.setdefault(it.key, []).append(it)

    do: Dict[str, Any] = {}
    avoid: Dict[str, Any] = {}
    allow_if: Dict[str, Any] = {}
    superseded: List[Dict[str, Any]] = []
    kept = dropped = 0
    reasons: List[str] = []

    def prec_rank(it: Item) -> int:
        kind = _kind(it)
        for i, (k, sc) in enumerate(PRECEDENCE):
            if kind == (k, it.scope):
                return i
        return len(PRECEDENCE)

    def score(it: Item) -> tuple:
        dec = _decay_score(it.confidence, it.created_at, half_life_days)
        return (-prec_rank(it), dec, it.confidence, it.created_at)

    for key, items in bucket.items():
        items_sorted = sorted(items, key=score, reverse=True)
        head, *tail = items_sorted

        if head.desired is None:
            allow_if[key] = head.value
            reasons.append(f"allow_if {key} from exception scope={head.scope}")
        elif head.desired:
            do[key] = head.value
            reasons.append(f"do {key} from scope={head.scope}")
        else:
            avoid[key] = head.value
            reasons.append(f"avoid {key} from scope={head.scope}")
        kept += 1

        for t in tail:
            superseded.append({
                "key": key,
                "value": t.value,
                "scope": t.scope,
                "desired": t.desired,
                "superseded_by": head.scope,
            })
            dropped += 1

    return PolicyResult(
        do=do,
        avoid=avoid,
        allow_if=allow_if,
        superseded=superseded,
        kept=kept,
        dropped=dropped,
        reasons=reasons,
    )

def _filter_llm_prefs(d: dict, policy_for_key) -> dict:
    return {k:v for k,v in (d or {}).items() if policy_for_key(k).send_to_llm}