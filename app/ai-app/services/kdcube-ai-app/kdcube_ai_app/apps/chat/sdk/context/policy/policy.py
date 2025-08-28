# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chatbot/context/policy.py
from typing import Dict, Any, List, Tuple, TypedDict, Optional
from dataclasses import dataclass
import time
from pydantic import BaseModel, Field

# ---------- Preference extraction (generic) ----------

class PrefAssertion(BaseModel):
    key: str                               # dotted, normalized (e.g., "suggest.training", "budget.max")
    value: Any = None                      # any JSON value, e.g., {"category":"pentest","max":30000}
    desired: bool = True                   # True=prefer/include, False=avoid/exclude
    scope: str = "conversation"
    confidence: float = Field(0.6, ge=0.0, le=1.0)
    reason: str = "nl-extracted"

class PrefException(BaseModel):
    rule_key: str                          # dotted, normalized (e.g., "policy.require.mfa")
    value: Any = True
    scope: str = "conversation"
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    reason: str = "nl-extracted"

class PreferenceExtractionOut(BaseModel):
    assertions: List[PrefAssertion] = Field(default_factory=list)
    exceptions: List[PrefException] = Field(default_factory=list)


PRECEDENCE = [
    ("exception", "session"),
    ("exception", "user"),
    ("neg", "session"),
    ("pos", "session"),
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

def _kind(item: Item) -> Tuple[str,str]:
    if item.desired is None:  # exception
        return ("exception", item.scope)
    return ("pos" if item.desired else "neg", item.scope)

def evaluate_policy(raw: Dict[str, Any], *, half_life_days: float = 30.0) -> PolicyResult:
    """
    Apply precedence + conflict resolution.
    - exception > session-neg > session-pos > user-neg > user-pos > global-neg > global-pos
    - for the same key, choose by (decayed_score, confidence, recency)
    """
    assertions = [
        Item(
            key=a.get("key"), value=a.get("value"),
            scope=(a.get("scope") or "user"),
            desired=bool(a.get("desired")),
            confidence=float(a.get("confidence") or 0.5),
            created_at=int(a.get("created_at") or 0),
            ttl_days=int(a.get("ttl_days") or 365),
            reason=a.get("reason") or "unknown"
        )
        for a in (raw.get("assertions") or [])
    ]
    exceptions = [
        Item(
            key=e.get("rule_key"), value=e.get("value"),
            scope=(e.get("scope") or "user"),
            desired=None,
            confidence=1.0, created_at=int(e.get("created_at") or 0),
            ttl_days=365, reason=e.get("reason") or "exception"
        )
        for e in (raw.get("exceptions") or [])
    ]

    bucket: Dict[str, List[Item]] = {}
    for it in assertions + exceptions:
        bucket.setdefault(it.key, []).append(it)

    do: Dict[str,Any] = {}
    avoid: Dict[str,Any] = {}
    allow_if: Dict[str,Any] = {}
    superseded: List[Dict[str,Any]] = []
    kept = dropped = 0
    reasons: List[str] = []

    # per key resolve
    for key, items in bucket.items():
        # sort by precedence first
        def prec_rank(it: Item) -> int:
            kind = _kind(it)
            for i,(k,sc) in enumerate(PRECEDENCE):
                if kind == (k, it.scope):
                    return i
            # unseen → low priority
            return len(PRECEDENCE)

        # compute score
        def score(it: Item) -> tuple:
            dec = _decay_score(it.confidence, it.created_at, half_life_days)
            return (-prec_rank(it), dec, it.confidence, it.created_at)

        items_sorted = sorted(items, key=score, reverse=True)
        head, *tail = items_sorted

        # apply head
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

        # record superseded
        for t in tail:
            superseded.append({
                "key": key,
                "value": t.value,
                "scope": t.scope,
                "desired": t.desired,
                "superseded_by": head.scope
            })
            dropped += 1

    return PolicyResult(do=do, avoid=avoid, allow_if=allow_if, superseded=superseded, kept=kept, dropped=dropped, reasons=reasons)
