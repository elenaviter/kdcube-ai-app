# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/prefs/promoter.py
# Can run opportunistically after each turn or as a background job / cron to promote user-level assertions

from __future__ import annotations
from typing import Any, Dict, Callable, List, Tuple
import time
import json
from kdcube_ai_app.apps.chat.sdk.context.graph.graph_ctx import GraphCtx
from kdcube_ai_app.apps.chat.sdk.context.policy.policy import KeyPolicy
from kdcube_ai_app.apps.chat.sdk.context.prefs.value_eq import canonicalize_value, values_equivalent

REASON_WEIGHTS = {
    "user-explicit": 1.00,
    "agent": 0.85,
    "nl-extracted": 0.80,
    "heuristic-budget": 0.80,
    "heuristic-negation": 0.70,
}

def _reason_w(reason: str) -> float:
    return REASON_WEIGHTS.get((reason or "").lower(), 0.75)

def _decay_score(conf: float, created_at: int, half_life_days: float) -> float:
    age_days = max(0.0, (time.time() - created_at) / 86400.0)
    return float(conf) * (0.5 ** (age_days / half_life_days))

def _unpack_value(a: Dict[str, Any]) -> Any:
    # GraphCtx stores 'value' or 'value_json' depending on type
    v = a.get("value", None)
    if v is None and a.get("value_json"):
        try:
            v = json.loads(a["value_json"])
        except Exception:
            v = a["value_json"]
    return v

async def promote_user_preferences(
        graph: GraphCtx,
        *,
        tenant: str,
        project: str,
        user: str,
        policy_for_key: Callable[[str], KeyPolicy],
        reason_weights: Dict[str, float] | None = None
) -> Dict[str, Any]:
    items = await graph.load_user_assertions_with_support(
        tenant=tenant, project=project, user=user
    )
    if not items:
        return {"promoted": [], "skipped": [], "blocked": []}

    now = int(time.time())
    REASON_W = reason_weights or REASON_WEIGHTS

    last_pos: Dict[str, int] = {}
    last_neg: Dict[str, int] = {}
    seen_by_key: Dict[str, List[Dict[str, Any]]] = {}
    challenged_at_by_key: Dict[str, int] = {}

    for a in items:
        if a.get("scope") == "user":
            continue # don't count user-scope records toward promotion evidence
        key = a.get("key")
        if not key:
            continue
        seen_by_key.setdefault(key, []).append(a)
        if a.get("scope") == "user":
            challenged_at = int(a.get("challenged_at") or 0)
            if challenged_at:
                challenged_at_by_key[key] = max(challenged_at_by_key.get(key, 0), challenged_at)

        # polarity buckets for "opposing evidence" checks
        seen_ts = int(a.get("last_seen_at") or a.get("created_at") or 0)
        if bool(a.get("desired")):
            last_pos[key] = max(last_pos.get(key, 0), seen_ts)
        else:
            last_neg[key] = max(last_neg.get(key, 0), seen_ts)

    # buckets per (key, desired, semantic value)
    buckets: Dict[Tuple[str, bool, int], Dict[str, Any]] = {}
    index: Dict[str, List[Tuple[int, Any]]] = {}
    next_id = 1

    for a in items:
        key = a.get("key")
        desired = bool(a.get("desired"))
        # use freshness
        seen_ts = int(a.get("last_seen_at") or a.get("created_at") or 0)
        conf = float(a.get("confidence") or 0.6)
        reason = (a.get("reason") or "").lower()
        if not key:
            continue

        val = canonicalize_value(key, _unpack_value(a), get_policy=policy_for_key)

        # find equivalent bucket
        bid = None
        for (bid_i, v0) in index.get(key, []):
            if buckets[(key, desired, bid_i)]["desired"] != desired:
                continue
            if values_equivalent(key, v0, val, get_policy=policy_for_key):
                bid = bid_i
                break
        if bid is None:
            bid = next_id
            next_id += 1
            index.setdefault(key, []).append((bid, val))
            buckets[(key, desired, bid)] = {
                "key": key, "desired": desired, "value": val,
                "support": 0,
                "conv_ids": set(),     # filled from edges
                "days": set(),         # distinct active days from edges
                "score_sum": 0.0,
                "last_seen": 0,
                "reasons": set(),
                "edge_hits": 0,
            }

        pol = policy_for_key(key)
        decay = _decay_score(conf, seen_ts, pol.half_life_days)
        w = (REASON_W or REASON_WEIGHTS).get(reason, 0.75)
        # tiny freshness bump for assertions seen in the last ~24h
        if (time.time() - seen_ts) < 86400:
            decay = min(1.2, decay * 1.05)

        # per-conversation edges from GraphCtx loader
        conv_edges = a.get("_conversations") or []
        # distinct conversations & days that actually touched this assertion
        conv_ids = {e.get("conv") for e in conv_edges if e.get("conv")}
        day_bins = {
            int((e.get("last_seen") or e.get("first_seen") or seen_ts) // 86400)
            for e in conv_edges
        }
        total_edge_hits = sum(int(e.get("hits") or 0) for e in conv_edges)

        rec = buckets[(key, desired, bid)]
        rec["support"] += 1
        rec["conv_ids"].update(conv_ids)
        rec["days"].update(day_bins)
        rec["score_sum"] += w * decay
        rec["last_seen"] = max(rec["last_seen"], seen_ts)
        rec["reasons"].add(reason or "unknown")
        rec["edge_hits"] += total_edge_hits

    promoted, skipped, blocked = [], [], []

    for (key, desired, bid), rec in buckets.items():
        pol = policy_for_key(key)
        distinct_convs = len(rec["conv_ids"])
        distinct_days = len(rec["days"])
        avg_decayed = rec["score_sum"] / max(1, rec["support"])

        # opposing = evidence of the opposite polarity within horizon
        opposing_ts = (last_neg if rec["desired"] else last_pos).get(key, 0)
        recent_conflict = opposing_ts >= (now - pol.conflict_horizon_days * 86400)
        challenged_recent = challenged_at_by_key.get(key, 0) >= (now - pol.conflict_horizon_days * 86400)

        if "[" in key and key.endswith("]"):
            skipped.append({"key": key, "desired": desired, "value": rec["value"],
                            "support": rec["support"], "distinct_convs": distinct_convs,
                            "distinct_days": distinct_days, "avg_decayed": round(avg_decayed,3),
                            "reason": "object_scoped"})
            continue
        if recent_conflict or challenged_recent:
            blocked.append({"key": key, "desired": desired,
                            "reason": ("recent_conflict" if recent_conflict else "challenged_recent")})
            continue

        if distinct_convs >= pol.min_support and distinct_days >= pol.distinct_days and avg_decayed >= pol.avg_decayed:
            await graph.upsert_assertion(
                tenant=tenant, project=project, user=user,
                conversation=None,                  # user-level
                key=key, value=rec["value"], desired=desired, scope="user",
                confidence=min(0.99, max(0.6, avg_decayed)),
                ttl_days=pol.ttl_days_user, reason="promoted",
                bump_time=True
            )
            promoted.append({
                "key": key, "desired": desired, "value": rec["value"],
                "support": rec["support"], "distinct_convs": distinct_convs,
                "distinct_days": distinct_days, "avg_decayed": round(avg_decayed, 3)
            })
        else:
            skipped.append({
                "key": key, "desired": desired, "value": rec["value"],
                "support": rec["support"], "distinct_convs": distinct_convs,
                "distinct_days": distinct_days, "avg_decayed": round(avg_decayed, 3)
            })

    return {"promoted": promoted, "skipped": skipped, "blocked": blocked}

"""
    from <your app>.registry import policy_for_key as gardening_policy
    
    promote_res = await promote_user_preferences(
        self.graph,
        tenant=tenant, project=project, user=user,
        policy_for_key=gardening_policy
    )
    await self._emit({"type":"preferences.promotion","agent":"policy","step":"promote","status":"completed",
                      "title":"User-level Preference Promotion","data": promote_res}, rid)
"""