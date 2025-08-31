# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/prefs/promoter.py
# Can run opportunistically after each turn or as a background job / cron to promote user-level assertions

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import time, json
from dataclasses import dataclass

from kdcube_ai_app.apps.chat.sdk.context.graph.graph_ctx import GraphCtx

def _now_sec() -> int:
    return int(time.time())

def _decay_score(base: float, created_at: int, half_life_days: float = 45.0) -> float:
    age_days = max(0.0, (_now_sec() - int(created_at)) / 86400.0)
    return float(base) * (0.5 ** (age_days / half_life_days))

def _unpack_value(node: Dict[str, Any]) -> Any:
    if node.get("value_json"):
        try:
            return json.loads(node["value_json"])
        except Exception:
            return None
    return node.get("value")

@dataclass
class PolicyThresholds:
    min_support: int = 2
    avg_decayed: float = 0.7
    conflict_horizon_days: int = 30
    half_life_days: float = 45.0
    ttl_days_user: int = 365

# Per-key prefixes (configure as needed)
KEY_POLICY: List[tuple[str, PolicyThresholds]] = [
    ("budget.",   PolicyThresholds(min_support=1, avg_decayed=0.6, conflict_horizon_days=60)),
    ("suggest.",  PolicyThresholds(min_support=2, avg_decayed=0.7, conflict_horizon_days=45)),
    ("focus.",    PolicyThresholds(min_support=3, avg_decayed=0.75, conflict_horizon_days=30, ttl_days_user=240)),
    ("style.",    PolicyThresholds(min_support=3, avg_decayed=0.75, conflict_horizon_days=30, ttl_days_user=180)),
]

DEFAULT_POLICY = PolicyThresholds()

def _policy_for_key(key: str) -> PolicyThresholds:
    for prefix, pol in KEY_POLICY:
        if key.startswith(prefix):
            return pol
    return DEFAULT_POLICY

async def promote_user_preferences(
        graph: GraphCtx, *, tenant: str, project: str, user: str,
        lookback_days: int = 180
) -> Dict[str, Any]:
    """Promote stable conversation-scoped assertions to user scope."""
    items = await graph.load_conversation_assertions(
        tenant=tenant, project=project, user=user, lookback_days=lookback_days
    )

    # Aggregate by (key, value_hash, desired)
    buckets: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}
    conflicts_latest: Dict[str, int] = {}

    for a in items:
        key = a.get("key")
        vhash = a.get("value_hash")
        desired = bool(a.get("desired"))
        created_at = int(a.get("created_at") or 0)
        conf = float(a.get("confidence") or 0.6)
        k = (key, vhash, desired)
        rec = buckets.setdefault(k, {
            "key": key, "value_hash": vhash, "desired": desired,
            "value": _unpack_value(a), "support": 0, "conv_ids": set(),
            "score_sum": 0.0, "last_seen": 0
        })
        rec["support"] += 1
        rec["conv_ids"].add(a.get("conversation"))
        rec["score_sum"] += _decay_score(conf, created_at, _policy_for_key(key).half_life_days)
        rec["last_seen"] = max(rec["last_seen"], created_at)

        # Track latest opposite sighting by key
        opp_key = (key, vhash, not desired)
        if opp_key in buckets:
            conflicts_latest[key] = max(conflicts_latest.get(key, 0), buckets[opp_key]["last_seen"])

    promoted, skipped = [], []
    now = _now_sec()

    for (key, vhash, desired), rec in buckets.items():
        pol = _policy_for_key(key)
        distinct_convs = len(rec["conv_ids"])
        avg_decayed = rec["score_sum"] / max(1, rec["support"])
        recent_conflict = conflicts_latest.get(key, 0) >= (now - pol.conflict_horizon_days * 86400)

        if distinct_convs >= pol.min_support and avg_decayed >= pol.avg_decayed and not recent_conflict:
            await graph.upsert_assertion(
                tenant=tenant, project=project, user=user, conversation=None,
                key=key, value=rec["value"], desired=desired, scope="user",
                confidence=min(0.95, avg_decayed + 0.05), ttl_days=pol.ttl_days_user,
                reason="auto-promoted", turn_id=None, bump_time=True
            )
            promoted.append({
                "key": key, "desired": desired, "support": distinct_convs,
                "avg_decayed": round(avg_decayed, 3)
            })
        else:
            skipped.append({
                "key": key, "desired": desired, "support": distinct_convs,
                "avg_decayed": round(avg_decayed, 3), "recent_conflict": bool(recent_conflict)
            })

    return {"promoted": promoted, "skipped": skipped}

"""
try:
    from kdcube_ai_app.apps.chat.sdk.context.prefs.promoter import promote_user_preferences
    promo = await promote_user_preferences(
        self.graph, tenant=tenant, project=project, user=user, lookback_days=180
    )
    if (promo.get("promoted") or promo.get("skipped")) is not None:
        await self._emit({
            "type": "preferences.promotion",
            "agent": "policy",
            "step": "promote",
            "status": "completed",
            "title": "User Preference Promotion",
            "data": promo
        }, rid)
except Exception:
    pass
"""