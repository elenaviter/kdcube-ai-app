# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/context/memory/memories_reconciler.py
from __future__ import annotations

import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import json, re, datetime

from kdcube_ai_app.apps.chat.sdk.streaming.streaming import _stream_simple_structured_json, _get_2section_protocol
from kdcube_ai_app.infra.service_hub.inventory import create_cached_system_message

from ..memory.buckets import MemoryBucket, TimelineSlice, Signal, now_iso

# ----------------- Strict I/O schemas for the LLM -----------------
def _slug(s: str, n: int = 24) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:n] or "bucket"

def gen_new_bucket_id(objective_text: str) -> str:
    """
    Stable-enough bucket id: slug + short hash.
    Different objectives → different ids; same objective → same id unless we later split/merge.
    """
    base = _slug(objective_text, 32)
    h = hashlib.sha1((objective_text or "").encode("utf-8")).hexdigest()[:8]
    return f"obj-{base}-{h}"


class ThinSliceIn(BaseModel):
    # trimmed FSP aggregate for the window (already deduped by host)
    # oldest->newest or newest-first? We'll pass NEWEST-FIRST to the LLM (explicit in prompt)
    made_at: str
    objective: str = ""
    topics: List[str] = Field(default_factory=list)
    assertions: List[Dict[str, Any]] = Field(default_factory=list)   # [{key,value,severity?,scope?,applies_to?}]
    exceptions: List[Dict[str, Any]] = Field(default_factory=list)   # [{key,value,severity?,scope?,applies_to?}]
    facts: List[Dict[str, Any]] = Field(default_factory=list)        # [{key,value}]
    support_turn_id: Optional[str] = None

class ThinBucketIn(BaseModel):
    bucket_id: str
    status: str = "enabled"               # enabled|disabled
    name: str
    short_desc: str = ""
    topic_centroid: List[str] = Field(default_factory=list)
    objective_text: str = ""
    # Last N compressed slices, oldest->newest (thin)
    timeline_sample: List[Dict[str, Any]] = Field(default_factory=list)

class ReconcileBucketOut(BaseModel):
    # FULL bucket after reconciliation — for changed/added buckets
    bucket_id: str = ""                   # empty => host assigns
    status: str = "enabled"               # enabled|disabled
    name: str
    short_desc: str = ""
    topic_centroid: List[str] = Field(default_factory=list)
    objective_text: str = ""
    # Full, compressed, ordered oldest->newest
    timeline: List[Dict[str, Any]] = Field(default_factory=list)  # List[TimelineSlice as dict]

class ReconcilerOut(BaseModel):
    changed_buckets: List[ReconcileBucketOut] = Field(default_factory=list)   # existing ids with new contents
    added_buckets: List[ReconcileBucketOut]   = Field(default_factory=list)   # new buckets, no id or empty id
    disabled_bucket_ids: List[str] = Field(default_factory=list)              # ids to mark disabled
    last_covered_ts: str = ""
    reason: str = ""

DEFAULT_EXAMPLE_INPUT = {
    "all_buckets_thin": [
        {"bucket_id":"mem_rev","status":"enabled","name":"Revenue Forecast FY25","short_desc":"Quarterly targets & pipeline health","topic_centroid":["revenue","forecast","pipeline"],"objective_text":"Build FY25 revenue forecast","timeline_sample":[
            {"ts_from":"2025-06-01T00:00:00Z","ts_to":"2025-08-01T00:00:00Z","objective_hint":"Base case","facts":[{"key":"forecast.revenue.total","value":1250000,"weight":0.6}],"assertions":[{"key":"pipeline.coverage","value":"3.2x","weight":0.4}],"exceptions":[]}
        ]}
    ],
    "window_newest_first": [
        {"made_at":"2025-09-23T11:59:00Z","objective":"Revision: raise forecast; offset churn via renewals+upsell","topics":["revenue","forecast"],"facts":[{"key":"forecast.revenue.total","value":1320000}],"assertions":[{"key":"churn.offset.plan","value":{"choice":"renewals+upsell","severity":"prefer"}}]}
    ]
}
DEFAULT_EXAMPLE_OUTPUT = {
    "changed_buckets":[
        {
            "bucket_id":"mem_rev",
            "status":"enabled",
            "name":"Revenue Forecast FY25",
            "short_desc":"Quarterly targets, pipeline health, and churn offsets",
            "topic_centroid":["revenue","forecast","pipeline"],
            "objective_text":"FY25 revenue forecast with pipeline growth and churn containment",
            "timeline":[
                {"ts_from":"2025-06-01T00:00:00Z","ts_to":"2025-08-01T00:00:00Z","objective_hint":"Base case",
                 "facts":[{"key":"forecast.revenue.total","value":1250000,"weight":0.6,"last_seen_ts":"2025-08-01T00:00:00Z"}],
                 "assertions":[{"key":"pipeline.coverage","value":"3.2x","weight":0.4,"last_seen_ts":"2025-08-01T00:00:00Z"}]},
                {"ts_from":"2025-09-23T11:59:00Z","ts_to":"2025-09-23T11:59:00Z","objective_hint":"Revision",
                 "facts":[{"key":"forecast.revenue.total","value":1320000,"weight":0.9,"last_seen_ts":"2025-09-23T11:59:00Z"}],
                 "assertions":[{"key":"churn.offset.plan","value":"renewals+upsell","weight":0.6,"last_seen_ts":"2025-09-23T11:59:00Z"}],
                 "exceptions":[]}
            ]
        }
    ],
    "added_buckets":[],
    "disabled_bucket_ids":[],
    "last_covered_ts":"2025-09-23T11:59:00Z",
    "reason":"Window aligns with revenue forecasting; raised total forecast and noted churn offset strategy."
}

# ----------------- LLM call -----------------

async def objective_reconciler_stream(
        *,
        model_service,
        all_buckets_thin: List[ThinBucketIn],      # enabled+disabled; the reconciler can re-enable by returning status="enabled" inside changed_buckets
        window_newest_first: List[ThinSliceIn],    # newest-first FSP aggregates since last reconcile
        max_changed: int = 4,
        max_added: int = 2,
        example_input: Optional[dict] = None,
        example_output: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    The reconciler sees ALL buckets (enabled+disabled) and the newest window of
    trimmed FSP aggregates. It outputs the *full contents* of changed/added buckets
    (thin, compressed timelines) and a list of bucket ids to disable.
    """
    today = datetime.datetime.utcnow().date().isoformat()

    if not example_input:
        example_input = DEFAULT_EXAMPLE_INPUT
    if not example_output:
        example_output = DEFAULT_EXAMPLE_OUTPUT

    sys_1 = (
        "You are a MEMORY RECONCILER.\n"
        "You maintain THEMATIC MEMORY BUCKETS as THIN, COMPRESSED TIMELINES.\n"
        "Inputs:\n"
        "• all_buckets_thin: every existing bucket (enabled+disabled) with a small timeline sample.\n"
        "• window_newest_first: newest-first per turn local memories (fingerprints) aggregates since last reconcile.\n\n"

        "Tasks:\n"
        "1) THEME & MATCH:\n"
        "   - Identify 1–3 themes in the window (by objective+topics+signals).\n"
        "   - For each theme, match the best existing bucket by name/desc/topic_centroid/objective_text similarity.\n"
        "   - If nothing fits, propose a new bucket (add to added_buckets).\n"
        "   - You MAY revive a previously disabled bucket if the theme clearly matches it; just return it in changed_buckets with status=\"enabled\" and update if needed.\n"
        "2) COMPRESS:\n"
        "   - Build a THIN, ORDERED (oldest->newest) timeline per changed/added bucket.\n"
        "   - A timeline is a list of SLICES. Each slice summarizes a time window (can be a single turn window) with:\n"
        "     {ts_from, ts_to, objective_hint, assertions[], exceptions[], facts[]}, where each list contains unique keys with consolidated values and weights.\n"
        "   - Each assertion/exception item should preserve optional fields when present: severity, scope, applies_to.\n"
        "   - Merge rule: treat (key,value,scope,applies_to) as the identity. If duplicates appear, keep the newest and increase weight; if conflicting values exist for the same key+scope+applies_to, keep both but reduce weights.\n"
        "   - Weights are relative importance [0.05..1.0]. Use 0.10 weak, 0.20 moderate, 0.30 strong signal increments; decay older/conflicting cues by decreasing weights.\n"
        "   - Deduplicate keys, squash synonyms, avoid ephemeral one-offs. Keep only durable items.\n"
        "   - Keep the bucket thin. Prefer ≤3 signals per kind per newest slice, and truncate older slices if needed.\n"
        "3) BUCKET MGMT:\n"
        "   - If a bucket is obsolete, list its id in disabled_bucket_ids.\n"
        "   - You NEVER delete in place; disabling hides it from future pickers. Reviving is allowed by returning it as changed with status=\"enabled\".\n"
        "4) OUTPUT ONLY JSON with these contents (no extra text):\n"
        "   - changed_buckets: full buckets that existed and now changed (including possibly re-enabled).\n"
        "   - added_buckets: full new buckets (bucket_id should be empty; host will assign ids).\n"
        "   - disabled_bucket_ids: ids to disable.\n"
        "   - last_covered_ts: the freshest made_at from the window.\n"
        "STRICTNESS:\n"
        f"• Touch at most {max_changed} changed buckets and {max_added} added buckets.\n"
        "• Keep timelines compact and ordered oldest->newest; slice fields are required.\n"
        "• Do NOT output any text outside the JSON response.\n"
        "Return a single JSON object matching this shape:\n"
        "```json\n"
        "{\n"
        "  \"changed_buckets\": [ {\"bucket_id\": str, \"status\": \"enabled|disabled\", \"name\": str, \"short_desc\": str,\n"
        "                          \"topic_centroid\": [str], \"objective_text\": str,\n"
        "                          \"timeline\": [ {\"ts_from\": str, \"ts_to\": str, \"objective_hint\": str,\n"
        "                                         \"facts\": [ {\"key\": str, \"value\": any, \"weight\": number, \"last_seen_ts\": str?, \"support_turn_ids\": [str]? } ],\n"
        "                                         \"assertions\": [ {\"key\": str, \"value\": any, \"severity\"?: str, \"scope\"?: str, \"applies_to\"?: str, \"weight\": number, \"last_seen_ts\": str?, \"support_turn_ids\": [str]? } ],\n"
        "                                         \"exceptions\": [ {\"key\": str, \"value\": any, \"severity\"?: str, \"scope\"?: str, \"applies_to\"?: str, \"weight\": number, \"last_seen_ts\": str?, \"support_turn_ids\": [str]? } ] } ] } ],\n"
        "  \"added_buckets\":   [ {\"bucket_id\": \"\"|str, ... same shape as above ... } ],\n"
        "  \"disabled_bucket_ids\": [str],\n"
        "  \"last_covered_ts\": str,\n"
        "  \"reason\": str\n"
        "}\n"
        "```\n"
    )
    time_evidence =  f"Assume today={today} (UTC)."

    # two_section_proto = _get_2section_protocol(json.dumps(example_output, indent=2))
    # sys = _add_2section_protocol(sys, schema)
    sys_2 = (f"EXAMPLE_INPUT:\n{json.dumps(example_input, indent=2, ensure_ascii=False)}\n\n")
    sys_3 = (f"EXAMPLE_OUTPUT:\n{json.dumps(example_output, indent=2, ensure_ascii=False)}\n\n")

    # "EXAMPLE_OUTPUT": example_output
    system_msg = create_cached_system_message([
        {"text": sys_1, "cache": True},
        {"text": time_evidence, "cache": True},
        {"text": sys_2, "cache": True},
        {"text": sys_3, "cache": True},
    ])

    user_payload = {
        "ALL_BUCKETS_THIN": [b.model_dump() if isinstance(b, ThinBucketIn) else b for b in all_buckets_thin],
        "WINDOW_NEWEST_FIRST": [w.model_dump() if isinstance(w, ThinSliceIn) else w for w in window_newest_first],
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False)

    out = await _stream_simple_structured_json(
        model_service,
        client_name="memories.reconciler",
        client_role="memories.reconciler",
        sys_prompt=system_msg,
        user_msg=user_msg,
        schema_model=ReconcilerOut,
        ctx="memories.reconciler",
        max_tokens=5000,
    )

    if not out:
        fresh = max((w.made_at for w in window_newest_first), default="")
        empty = ReconcilerOut(last_covered_ts=fresh, reason="fallback_noop").model_dump()
        return {"json": empty, "agent_response": empty}

    resp = out.get("agent_response") or {}
    # Pydantic validation / size clamps (unchanged)
    try:
        model = ReconcilerOut.model_validate(resp)
    except Exception:
        model = ReconcilerOut(**{
            "changed_buckets": resp.get("changed_buckets", []),
            "added_buckets": resp.get("added_buckets", []),
            "disabled_bucket_ids": resp.get("disabled_bucket_ids", []),
            "last_covered_ts": resp.get("last_covered_ts", ""),
            "reason": resp.get("reason", "")
        })

    model.changed_buckets = model.changed_buckets[:max_changed]
    model.added_buckets = model.added_buckets[:max_added]

    # Assign IDs to added buckets if missing
    for b in model.added_buckets:
        if not b.bucket_id:
            b.bucket_id = gen_new_bucket_id(b.name or "bucket")

    # Ensure required slice fields & ordering
    def _sanitize_bucket(b: ReconcileBucketOut) -> ReconcileBucketOut:
        if b.status not in ("enabled", "disabled"):
            b.status = "enabled"
        # order timeline oldest->newest by ts_to/ts_from
        try:
            b.timeline.sort(key=lambda s: (s.get("ts_to") or s.get("ts_from") or ""))
        except Exception:
            pass
        return b

    model.changed_buckets = [_sanitize_bucket(b) for b in model.changed_buckets]
    model.added_buckets = [_sanitize_bucket(b) for b in model.added_buckets]

    if not model.last_covered_ts:
        model.last_covered_ts = max((w.made_at for w in window_newest_first), default=now_iso())

    clean = model.model_dump()
    return {"json": clean, "agent_response": clean}


# ----------------- Engineering wrappers -----------------

def thin_sample_from_bucket(b: MemoryBucket, *, max_slices: int = 2, max_per_kind: int = 4) -> ThinBucketIn:
    # Take last N slices, oldest->newest
    slices = b.timeline[-max_slices:] if b.timeline else []
    thin = []
    for sl in slices:
        def _top(arr: List[Signal]):
            arr2 = sorted(arr, key=lambda s: float(s.weight), reverse=True)[:max_per_kind]
            return [s.model_dump() for s in arr2]
        thin.append({
            "ts_from": sl.ts_from, "ts_to": sl.ts_to, "objective_hint": sl.objective_hint,
            "facts": _top(sl.facts), "assertions": _top(sl.assertions), "exceptions": _top(sl.exceptions)
        })
    return ThinBucketIn(
        bucket_id=b.bucket_id,
        status=b.status,
        name=b.name,
        short_desc=b.short_desc,
        topic_centroid=b.topic_centroid,
        objective_text=b.objective_text,
        timeline_sample=thin
    )

def materialize_bucket(rec: ReconcileBucketOut|dict|None) -> MemoryBucket|None:
    # Convert LLM output back to our storage model
    if not rec:
        return None
    if isinstance(rec, dict):
        try:
            rec = ReconcileBucketOut(**rec)  # validate
        except Exception:
            return None

    tl: List[TimelineSlice] = []
    for s in rec.timeline or []:
        def _mk_signals(items: List[Dict[str, Any]]) -> List[Signal]:
            out = []
            for it in items or []:
                out.append(Signal(
                    key=it.get("key",""),
                    value=it.get("value"),
                    weight=float(it.get("weight", 0.5)),
                    last_seen_ts=it.get("last_seen_ts"),
                    support_turn_ids=[tid for tid in (it.get("support_turn_ids") or []) if tid]
                ))
            return out
        tl.append(TimelineSlice(
            ts_from=s.get("ts_from",""), ts_to=s.get("ts_to",""),
            objective_hint=s.get("objective_hint",""),
            facts=_mk_signals(s.get("facts",[])),
            assertions=_mk_signals(s.get("assertions",[])),
            exceptions=_mk_signals(s.get("exceptions",[])),
        ))
    return MemoryBucket(
        bucket_id=rec.bucket_id,
        status=rec.status,
        name=rec.name,
        short_desc=rec.short_desc,
        topic_centroid=list(rec.topic_centroid or []),
        objective_text=rec.objective_text or rec.name,
        updated_at=now_iso(),
        timeline=tl
    )
