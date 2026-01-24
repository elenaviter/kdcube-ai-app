# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/context/memory/active_set_management.py
from typing import Any, Dict, List, Optional, Tuple

from kdcube_ai_app.apps.chat.sdk.context.retrieval.ctx_rag import ContextRAGClient
import kdcube_ai_app.apps.chat.sdk.viz.logging_helpers as logging_helpers
from kdcube_ai_app.apps.chat.sdk.context.memory.conv_memories import ConvMemoriesStore
from kdcube_ai_app.apps.chat.sdk.context.memory.memories_reconciler import ThinBucketIn, ThinSliceIn, objective_reconciler_stream, materialize_bucket, \
    ReconcileBucketOut
from .buckets import BucketStore, MemoryBucket, to_store_dict
from kdcube_ai_app.apps.chat.sdk.util import _tstart, _tend, _to_iso_minute
from ...runtime.scratchpad import TurnScratchpad

FINGERPRINT_KIND = "artifact:turn.fingerprint.v1"

def _sanitize_timeline_sample(sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in (sample or []):
        s2 = dict(s)
        s2["ts_from"] = _to_iso_minute(s2.get("ts_from", ""))
        s2["ts_to"]   = _to_iso_minute(s2.get("ts_to", ""))
        # If present on signals, normalize last_seen_ts as well
        for fld in ("facts", "assertions", "exceptions"):
            arr = []
            for it in (s2.get(fld) or []):
                it2 = dict(it)
                if it2.get("last_seen_ts"):
                    it2["last_seen_ts"] = _to_iso_minute(it2["last_seen_ts"])
                arr.append(it2)
            s2[fld] = arr
        out.append(s2)
    return out


async def _fetch_one_fp_by_tid(*, ctx_client, user: str, conversation_id: str, turn_id: str) -> dict | None:

    if not turn_id:
        return None
    try:
        res = await ctx_client.search(
            query=None,
            roles=("artifact",),
            kinds=(FINGERPRINT_KIND,),
            scope="track",
            user_id=user,
            conversation_id=conversation_id,
            turn_id=turn_id,              # <-- single turn filter
            days=365,
            top_k=1,
            include_deps=False,
            sort="recency",
            with_payload=True
        )
        for it in (res.get("items") or []):
            entry = (it.get("payload") or {}).get("payload")
            if entry:
                return entry
    except Exception:
        pass
    return None

def _merge_fps_by_tid(newest_first_a: list[dict], newest_first_b: list[dict]) -> list[dict]:
    by_tid = {}
    def _put(d):
        if not isinstance(d, dict): return
        tid = d.get("turn_id") or (d.get("source") or {}).get("turn_id")
        if tid and tid not in by_tid:
            by_tid[tid] = d
    for d in (newest_first_a or []): _put(d)   # give A precedence
    for d in (newest_first_b or []): _put(d)

    def _ts(d):
        ts = d.get("made_at") or d.get("ts") or ""
        # fallbacks are fine; we just need a stable sort
        return ts

    out = list(by_tid.values())
    out.sort(key=_ts, reverse=True)  # newest -> oldest
    return out

async def _preload_active_anchor_and_delta_fps(_ctx,
                                               scratchpad: TurnScratchpad,
                                               active_store: ConvMemoriesStore,
                                               ctx_client: ContextRAGClient,
                                               bundle_id: Optional[str] = None):
    """
    Loads current ACTIVE SET (once), extracts its anchor timestamp & turn id,
    then fetches ALL turn fingerprints strictly after that ts.
    Caches both on scratchpad: .active_anchor, .delta_fps
    """
    tenant, project, user = _ctx["service"]["tenant"], _ctx["service"]["project"], _ctx["service"]["user"]
    conversation_id = _ctx["conversation"]["conversation_id"]

    # 1) active set (single fetch)
    active_set = await active_store.get_active_set(
        tenant=tenant, project=project, user=user, conversation=conversation_id
    )

    scratchpad.is_new_conversation = active_set is None or (active_set.get("new") or False)
    if "new" in active_set:
        del active_set["new"]

    scratchpad.active_set = active_set or {"version":"v1", "assertions":[], "exceptions":[], "facts":[],
                                           "objective_hint":"", "topics":[], "active_bucket_id": None,
                                           "last_reconciled_ts": ""}
    if not scratchpad.is_new_conversation:
        scratchpad.conversation_title = scratchpad.active_set.get("conversation_title")

    # Load objective memory logs per bucket (freshest artifacts)
    logs_hit = await ctx_client.search(
        query=None,
        kinds=("artifact:conversation.objective.memory.log.v1",),
        roles=("artifact",),
        scope="conversation",
        user_id=user, conversation_id=conversation_id,
        days=365, top_k=64, include_deps=False, sort="recency",
        with_payload=True
    )
    # Keep headers (what you already had) AND a separate map of log entries
    scratchpad.objective_memories = {}       # bucket_id -> {header fields}
    scratchpad.objective_memory_logs = {}    # bucket_id -> [entries newest->oldest]

    for it in (logs_hit.get("items") or []):
        doc = (it.get("doc") or {})
        j = doc.get("json") or {}
        bid = (j.get("bucket_id") or "").strip()
        if not bid:
            continue
        # Header (idempotent)
        scratchpad.objective_memories.setdefault(bid, {
            "bucket_id": bid,
            "objective_text": j.get("objective_text", ""),
            "objective_embedding": j.get("objective_embedding") or [],
            "topics_centroid": j.get("topics_centroid") or [],
        })
        # Entries (freshest-first if writer saved them that way; otherwise we'll trust order given)
        entries = list(j.get("entries") or j.get("log_entries") or [])
        if entries:
            scratchpad.objective_memory_logs.setdefault(bid, [])
            # append preserving newest->oldest invariant (no heavy dedupe here)
            scratchpad.objective_memory_logs[bid].extend(entries)

    # 3) Hydrate previously picked buckets (conversation_state → cards + timelines)
    try:
        # Merge with any already-active picks (ADD recommended)
        picked_ids_raw = (scratchpad.active_set.get("picked_bucket_ids") or []) + (scratchpad.selected_memory_bucket_ids or [])
        picked_ids = list(dict.fromkeys([bid for bid in picked_ids_raw if bid]))  # preserve order, no dups

        # picked_ids = scratchpad.selected_memory_bucket_ids or list(scratchpad.active_set.get("picked_bucket_ids") or [])
        scratchpad.selected_memory_bucket_cards = []
        scratchpad.objective_memory_timelines = {}

        if picked_ids:
            bstore = BucketStore(ctx_client)
            all_buckets = await bstore.list_buckets(user=user, conversation_id=conversation_id, include_disabled=True)
            by_id = {b.get("bucket_id"): b for b in all_buckets}

            def _mk_card_from_store(b: dict) -> dict:
                last = (b.get("timeline") or [])[-1] if b.get("timeline") else {}
                def _top(items):
                    arr = sorted(items or [], key=lambda s: float(s.get("weight", 0.0)), reverse=True)[:4]
                    return [{"key": x.get("key",""), "value": x.get("value"), "weight": float(x.get("weight", 0.5))} for x in arr]
                return {
                    "bucket_id": b.get("bucket_id",""),
                    "status": "enabled" if b.get("enabled", True) else "disabled",
                    "name": b.get("name",""),
                    "short_desc": b.get("short_desc",""),
                    "topic_centroid": list(b.get("topic_centroid") or []),
                    "objective_text": b.get("objective_text") or b.get("name",""),
                    "top_signals": {
                        "facts": _top(last.get("facts")),
                        "assertions": _top(last.get("assertions")),
                        "exceptions": _top(last.get("exceptions")),
                    },
                    "updated_at": b.get("updated_at",""),
                }

            selected_cards = []
            objective_memory_timelines = {}

            for bid in picked_ids:
                b = by_id.get(bid)
                if not b or not b.get("enabled", True):
                    continue
                selected_cards.append(_mk_card_from_store(b))
                # full timeline for objective memory block (context display only)
                tl = []
                for s in (b.get("timeline") or []):
                    tl.append({
                        "ts_from": s.get("ts_from",""),
                        "ts_to": s.get("ts_to",""),
                        "objective_hint": s.get("objective_hint",""),
                        "assertions": list(s.get("assertions") or []),
                        "exceptions": list(s.get("exceptions") or []),
                        "facts":      list(s.get("facts") or []),
                    })
                objective_memory_timelines[b.get("bucket_id")] = tl

            scratchpad.selected_memory_bucket_cards = selected_cards
            scratchpad.objective_memory_timelines = objective_memory_timelines

    except Exception:
        # never fail the turn because of hydration
        pass

    # 4) Delta FPS since last_reconciled_ts (server-side filter)
    last_ts = scratchpad.active_set.get("last_reconciled_ts") or ""
    delta = await ctx_client.search(
        roles=("artifact",),
        kinds=(FINGERPRINT_KIND,),
        days=365, top_k=256, scope="track",
        user_id=user, conversation_id=conversation_id,
        include_deps=False, sort="recency",
        timestamp_filters=([{"op": ">", "value": last_ts}] if last_ts else None),
        with_payload=True
    )
    window_docs: list[dict] = []
    for it in (delta.get("items") or []):
        entry = (it.get("payload") or {}).get("payload")
        if entry:
            window_docs.append(entry)  # newest->oldest

    # Fetch explicit selections by TID and union with the window
    selected_docs: list[dict] = []
    seen_tid: set[str] = set()
    for tid in list(dict.fromkeys(scratchpad.selected_local_memories_turn_ids or [])):
        doc = await _fetch_one_fp_by_tid(
            ctx_client=ctx_client, user=user, conversation_id=conversation_id, turn_id=tid
        )
        if doc:
            t = doc.get("turn_id") or (doc.get("source") or {}).get("turn_id")
            if t and t not in seen_tid:
                selected_docs.append(doc)
                seen_tid.add(t)

    # Merge: explicit picks win on collisions; newest->oldest
    scratchpad.delta_turns_local_mem_entries = _merge_fps_by_tid(
        selected_docs,   # A: explicit picks
        window_docs      # B: since-anchor
    )
    # if scratchpad.selected_local_memories_turn_ids:
    #     # fetch allso these and then merge with below!
    #     pass
    # scratchpad.delta_turns_local_mem_entries = []
    # for it in (delta.get("items") or []):
    #     entry = (it.get("payload") or {}).get("payload")
    #     if entry:
    #         scratchpad.delta_turns_local_mem_entries.append(entry)  # keep newest->oldest order from search

async def _reconcile_objectives_if_due(
        *,
        window: List[dict],
        tenant: str, project: str, user: str, user_type: str,
        track_id: str, conversation_id: str, turn_id: str,
        bundle_id: str,
        model_service,
        ctx_client,
        active_store: ConvMemoriesStore,
        _emit_fn: Optional[Any] = None,
        example_input: Optional[dict] = None,
        example_output: Optional[dict] = None,
        persist_active_set: bool = True,
) -> Dict[str, Any]:
    """
    Execute LLM reconciler over the recent window of FSP aggregates.
    - Load ALL existing buckets (enabled + disabled)
    - Call LLM
    - Upsert changed/added IN PLACE; disable by in-place update
    - Update last_reconciled_ts and reset cadence counter
    """
    bstore = BucketStore(ctx_client)

    # 1) load all buckets (thin)
    existing = await bstore.list_buckets(user=user, conversation_id=conversation_id, include_disabled=True)
    thin_buckets: List[ThinBucketIn] = []
    for b in existing:
        # convert dict → MemoryBucket (just the fields we use)
        mb = MemoryBucket(
            bucket_id=b["bucket_id"],
            status=("enabled" if b.get("enabled", True) else "disabled"),
            name=b.get("name",""),
            short_desc=b.get("short_desc",""),
            topic_centroid=list(b.get("topic_centroid") or []),
            objective_text=b.get("objective_text","") or b.get("name",""),
            timeline=[]  # not needed in thin; we pass samples below
        )
        # Build a tiny sample from last slices (up to 2)
        sample = list(b.get("timeline") or [])[-2:]
        sample = _sanitize_timeline_sample(sample)
        thin_buckets.append(ThinBucketIn(
            bucket_id=mb.bucket_id,
            status=mb.status,
            name=mb.name,
            short_desc=mb.short_desc,
            topic_centroid=mb.topic_centroid,
            objective_text=mb.objective_text,
            timeline_sample=sample
        ))

    # 2) window → ThinSliceIn (keep newest-first order)
    thin_window: List[ThinSliceIn] = []
    for w in window or []:
        thin_window.append(ThinSliceIn(
            made_at=_to_iso_minute(w.get("made_at","")),
            objective=w.get("objective","") or "",
            topics=list(w.get("topics") or []),
            assertions=[{"key":a.get("key"), "value":a.get("value"), "desired":bool(a.get("desired",True))} for a in (w.get("assertions") or []) if a.get("key")],
            exceptions=[{"rule_key":e.get("rule_key"), "value":e.get("value")} for e in (w.get("exceptions") or []) if e.get("rule_key")],
            facts=[{"key":f.get("key"), "value":f.get("value")} for f in (w.get("facts") or []) if f.get("key")],
            support_turn_id=(w.get("source") or {}).get("turn_id")
        ))

    # 3) call reconciler LLM
    t04, ms04 = _tstart()
    out = await objective_reconciler_stream(
        model_service=model_service,
        all_buckets_thin=thin_buckets,
        window_newest_first=thin_window,
        max_changed=4, max_added=2,
        example_output=example_output,
        example_input=example_input
    )
    timing_reconcile = _tend(t04, ms04)
    res = out.get("agent_response") or {}
    logging_helpers.log_agent_packet("memory_buckets_reconciler", "reconcile conversation memories", res)
    if _emit_fn:
        await _emit_fn({"type": "conversation.long_mem_reconciler",
                        "agent": "long_memories_reconciler",
                        "step": "long_memories_reconciling",
                        "status": "completed",
                        "title": "Conversation Long Memories Reconciler",
                        "data": res,
                        "timing": timing_reconcile})

    changed = res.get("changed_buckets") or []
    added = res.get("added_buckets") or []
    disabled_ids = [bid for bid in (res.get("disabled_bucket_ids") or []) if bid]
    last_ts = _to_iso_minute(res.get("last_covered_ts") or "")

    # 4) persist (UPDATES IN PLACE)
    # changed → upsert
    for cb in changed:
        mb = materialize_bucket(cb)
        await bstore.save_bucket_upsert(
            tenant=tenant, project=project, user=user, user_type=user_type,
            conversation_id=conversation_id, turn_id=turn_id, track_id=track_id,
            bucket=to_store_dict(mb),
            bundle_id=bundle_id
        )
    # added → upsert (will create if not present)
    for ab in added:
        mb = materialize_bucket(ab)
        await bstore.save_bucket_upsert(
            tenant=tenant, project=project, user=user, user_type=user_type,
            conversation_id=conversation_id, turn_id=turn_id, track_id=track_id,
            bucket=to_store_dict(mb),
            bundle_id=bundle_id
        )
    # disabled → update in place
    for bid in disabled_ids:
        await bstore.disable_bucket_upsert(
            tenant=tenant, project=project, user=user, user_type=user_type,
            conversation_id=conversation_id, turn_id=turn_id, track_id=track_id,
            bucket_id=bid,
            bundle_id=bundle_id
        )

    # 5) update active-set pointer (cadence + last ts)
    as_ptr = await active_store.get_active_set(
        tenant=tenant, project=project, user=user, conversation=conversation_id
    ) or {}
    as_ptr["last_reconciled_ts"] = last_ts
    as_ptr["since_last_reconcile"] = 0
    if persist_active_set:
        await active_store.put_active_set(
            tenant=tenant, project=project, user=user, conversation=conversation_id,
            turn_id=turn_id,
            active_set=as_ptr, user_type=user_type, track_id=track_id, bundle_id=bundle_id
        )
        return res

    res = dict(res)
    res["active_set"] = as_ptr
    return res
