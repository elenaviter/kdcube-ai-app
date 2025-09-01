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

async def upsert_user_assertion(
    graph: GraphCtx, *, tenant: str, project: str, user: str,
    key: str, value: Any, desired: bool, confidence: float,
    ttl_days: int, reason: str
):
    v = value
    v_prim = v if isinstance(v, (str, int, float, bool, list)) else None
    v_json = None if v_prim is not None else json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    import hashlib
    async with graph._driver.session() as s:
        await s.run(
            """MATCH (u:User {key:$uk})-[:HAS_ASSERTION]->(old:Assertion)
               WHERE old.tenant=$tenant AND old.project=$project AND old.scope='user' AND old.key=$key
               DETACH DELETE old""",
            uk=f"{tenant}:{project}:{user}", tenant=tenant, project=project, key=key
        )
        await s.run(
            """MERGE (u:User {key:$uk})
               CREATE (a:Assertion {
                 id: randomUUID(), tenant:$tenant, project:$project, user:$user, conversation:$conv,
                 key:$key, value:$value_primitive, value_json:$value_json, value_type:$value_type,
                 value_hash:$value_hash, desired:$desired, scope:'user', confidence:$confidence,
                 created_at:$now, ttl_days:$ttl_days, reason:$reason
               })
               MERGE (u)-[:HAS_ASSERTION]->(a)""",
            uk=f"{tenant}:{project}:{user}", tenant=tenant, project=project, user=user,
            conv=f"{tenant}:{project}:<user>", key=key,
            value_primitive=v_prim,
            value_json=v_json,
            value_type=("primitive" if v_prim is not None else "object"),
            value_hash=hashlib.sha1((v_json if v_prim is None else json.dumps(v_prim, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")).hexdigest(),
            desired=bool(desired), confidence=float(confidence),
            now=int(time.time()), ttl_days=int(ttl_days), reason=reason
        )

async def promote_user_preferences(
        graph: GraphCtx,
        *,
        tenant: str,
        project: str,
        user: str,
        policy_for_key: Callable[[str], KeyPolicy],
        reason_weights: Dict[str, float] | None = None
) -> Dict[str, Any]:
    items = await graph.load_user_assertions(tenant=tenant, project=project, user=user)
    if not items:
        return {"promoted": [], "skipped": [], "blocked": []}

    now = int(time.time())
    REASON_W = reason_weights or REASON_WEIGHTS

    # latest opposing timestamps per key
    latest_opposing: Dict[str, int] = {}
    seen_by_key: Dict[str, List[Dict[str, Any]]] = {}
    challenged_at_by_key: Dict[str, int] = {}

    for a in items:
        key = a.get("key")
        seen_by_key.setdefault(key, []).append(a)
        if a.get("scope") == "user":
            challenged_at = int(a.get("challenged_at") or 0)
            if challenged_at:
                challenged_at_by_key[key] = max(challenged_at_by_key.get(key, 0), challenged_at)

    for key, arr in seen_by_key.items():
        arr_sorted = sorted(arr, key=lambda r: int(r.get("created_at") or 0), reverse=True)
        pos_ts = next((int(r.get("created_at") or 0) for r in arr_sorted if r.get("desired") is True), 0)
        neg_ts = next((int(r.get("created_at") or 0) for r in arr_sorted if r.get("desired") is False), 0)
        # if most recent item is opposite of candidate, we block within horizon
        latest_opposing[key] = max(pos_ts, neg_ts)

    # buckets per (key, desired, semantic value)
    buckets: Dict[Tuple[str, bool, int], Dict[str, Any]] = {}
    index: Dict[str, List[Tuple[int, Any]]] = {}
    next_id = 1

    for a in items:
        key = a.get("key")
        desired = bool(a.get("desired"))
        created_at = int(a.get("created_at") or 0)
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
                "support": 0, "conv_ids": set(), "days": set(),
                "score_sum": 0.0, "last_seen": 0, "reasons": set()
            }

        pol = policy_for_key(key)
        decay = _decay_score(conf, created_at, pol.half_life_days)
        w = REASON_W.get(reason, 0.75)

        rec = buckets[(key, desired, bid)]
        rec["support"] += 1
        rec["conv_ids"].add(a.get("conversation"))
        rec["days"].add(int(created_at // 86400))
        rec["score_sum"] += w * decay
        rec["last_seen"] = max(rec["last_seen"], created_at)
        rec["reasons"].add(reason or "unknown")

    promoted, skipped, blocked = [], [], []

    for (key, desired, bid), rec in buckets.items():
        pol = policy_for_key(key)
        distinct_convs = len(rec["conv_ids"])
        distinct_days = len(rec["days"])
        avg_decayed = rec["score_sum"] / max(1, rec["support"])

        recent_conflict = latest_opposing.get(key, 0) >= (now - pol.conflict_horizon_days * 86400)
        challenged_recent = challenged_at_by_key.get(key, 0) >= (now - pol.conflict_horizon_days * 86400)

        if recent_conflict or challenged_recent:
            blocked.append({
                "key": key,
                "desired": desired,
                "reason": ("recent_conflict" if recent_conflict else "challenged_recent")
            })
            continue

        if distinct_convs >= pol.min_support and distinct_days >= pol.distinct_days and avg_decayed >= pol.avg_decayed:
            await upsert_user_assertion(
                graph,
                tenant=tenant, project=project, user=user,
                key=key, value=rec["value"], desired=desired,
                confidence=min(0.99, max(0.6, avg_decayed)),
                ttl_days=pol.ttl_days_user, reason="promoted"
            )
            promoted.append({
                "key": key, "desired": desired, "value": rec["value"],
                "support": rec["support"], "avg_decayed": round(avg_decayed, 3)
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