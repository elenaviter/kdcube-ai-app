# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter
import datetime
# sdk/context/memory/active_set_management.py
from typing import Any, Dict, List, Optional, Tuple
import json
import ast
import datetime

from kdcube_ai_app.apps.chat.sdk.context.retrieval.ctx_rag import ContextRAGClient, FINGERPRINT_KIND
import kdcube_ai_app.apps.chat.sdk.viz.logging_helpers as logging_helpers
from kdcube_ai_app.apps.chat.sdk.context.memory.conv_memories import ConvMemoriesStore
from kdcube_ai_app.apps.chat.sdk.context.memory.memories_reconciler import ThinBucketIn, ThinSliceIn, objective_reconciler_stream, materialize_bucket, \
    ReconcileBucketOut
from .buckets import BucketStore, MemoryBucket, to_store_dict
from kdcube_ai_app.apps.chat.sdk.util import _tstart, _tend, _to_iso_minute
from ...runtime.scratchpad import TurnScratchpad

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

async def _preload_conversation_memory_state(_ctx,
                                             scratchpad: TurnScratchpad,
                                             active_store: ConvMemoriesStore,
                                             ctx_client: ContextRAGClient,
                                             bundle_id: Optional[str] = None,
                                             assistant_signal_scope: str = "user"):
    """
    Loads current ACTIVE SET (once), extracts its anchor timestamp & turn id,
    then fetches ALL turn fingerprints strictly after that ts.
    Also loads assistant-originated promo signals (scope configurable).
    Caches results on scratchpad: active_set, delta_turns_local_mem_entries,
    selected memory buckets, assistant_signals_user_level, feedback_conversation_level.
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

    scratchpad.active_set = active_set or {
        "version": "v1",
        "objective_hint": "",
        "topics": [],
        "active_bucket_id": None,
        "picked_bucket_ids": [],
        "selected_local_memories_turn_ids": [],
        "last_reconciled_ts": "",
    }
    if not scratchpad.is_new_conversation:
        scratchpad.conversation_title = scratchpad.active_set.get("conversation_title")

    scratchpad.objective_memories = {}
    scratchpad.assistant_signals_user_level = []

    def _parse_index_text(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        raw = text.strip()
        if raw.startswith("[turn.log.reaction]"):
            raw = raw.split("]", 1)[-1].strip()
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None

    # Assistant-originated signals (server-side tag filter)
    try:
        hit = await ctx_client.search(
            query=None,
            roles=("artifact",),
            kinds=(FINGERPRINT_KIND,),
            scope=assistant_signal_scope or "conversation",
            user_id=user,
            conversation_id=conversation_id,
            all_tags=["assistant_signal"],
            days=365,
            top_k=512,
            include_deps=False,
            sort="recency",
            with_payload=False,
        )
        items = []
        for it in (hit.get("items") or []):
            entry = _parse_index_text((it.get("text") or "").strip())
            if isinstance(entry, dict):
                items.append(entry)
        def _as_ts(fp: Dict[str, Any]) -> float:
            ts = (fp.get("made_at") or fp.get("ts") or "").strip()
            if not ts:
                return float("-inf")
            try:
                s = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
                return datetime.datetime.fromisoformat(s).timestamp()
            except Exception:
                return float("-inf")
        scratchpad.assistant_signals_user_level = sorted(items, key=_as_ts, reverse=True)
    except Exception:
        scratchpad.assistant_signals_user_level = []

    # Feedback reactions (server-side tag filter; index-only)
    scratchpad.feedback_conversation_level = []
    try:
        feedback_hit = await ctx_client.search(
            query=None,
            roles=("artifact",),
            kinds=("artifact:turn.log.reaction",),
            scope="conversation",
            user_id=user,
            conversation_id=conversation_id,
            days=365,
            top_k=128,
            include_deps=False,
            sort="recency",
            with_payload=False,
        )
        feedback_items = []
        for it in (feedback_hit.get("items") or []):
            entry = _parse_index_text((it.get("text") or "").strip())
            if not isinstance(entry, dict):
                continue
            if not entry.get("turn_id") and it.get("turn_id"):
                entry["turn_id"] = it.get("turn_id")
            feedback_items.append(entry)

        turn_ids = list({(f.get("turn_id") or "").strip() for f in feedback_items if f.get("turn_id")})
        objective_by_turn: Dict[str, str] = {}
        if turn_ids:
            fp_hit = await ctx_client.search(
                query=None,
                roles=("artifact",),
                kinds=(FINGERPRINT_KIND,),
                scope="conversation",
                user_id=user,
                conversation_id=conversation_id,
                days=365,
                top_k=min(256, max(32, len(turn_ids) * 2)),
                include_deps=False,
                sort="recency",
                any_tags=[f"turn:{tid}" for tid in turn_ids],
                with_payload=False,
            )
            for it in (fp_hit.get("items") or []):
                fp = _parse_index_text((it.get("text") or "").strip())
                if isinstance(fp, dict) and fp.get("turn_id"):
                    objective_by_turn[fp["turn_id"]] = (fp.get("objective") or "")

        def _fb_ts(fb: Dict[str, Any]) -> float:
            ts = (fb.get("ts") or "").strip()
            if not ts:
                return float("-inf")
            try:
                s = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
                return datetime.datetime.fromisoformat(s).timestamp()
            except Exception:
                return float("-inf")

        for fb in feedback_items:
            tid = (fb.get("turn_id") or "").strip()
            if tid and tid in objective_by_turn:
                fb["objective"] = objective_by_turn.get(tid) or ""
        scratchpad.feedback_conversation_level = sorted(feedback_items, key=_fb_ts, reverse=True)
    except Exception:
        scratchpad.feedback_conversation_level = []

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

    # 4) Delta FPS since last_reconciled_ts (server-side filter, index-only)
    last_ts = scratchpad.active_set.get("last_reconciled_ts") or ""
    delta = await ctx_client.search(
        roles=("artifact",),
        kinds=(FINGERPRINT_KIND,),
        days=365, top_k=256, scope="track",
        user_id=user, conversation_id=conversation_id,
        include_deps=False, sort="recency",
        timestamp_filters=([{"op": ">", "value": last_ts}] if last_ts else None),
        with_payload=False
    )
    window_docs: list[dict] = []
    for it in (delta.get("items") or []):
        entry = _parse_index_text((it.get("text") or "").strip())
        if isinstance(entry, dict):
            window_docs.append(entry)  # newest->oldest

    # Fetch explicit selections by TID (batch) and union with the window
    selected_docs: list[dict] = []
    selected_tids = list(dict.fromkeys(scratchpad.selected_local_memories_turn_ids or []))
    if selected_tids:
        selected_hit = await ctx_client.search(
            query=None,
            roles=("artifact",),
            kinds=(FINGERPRINT_KIND,),
            scope="conversation",
            user_id=user,
            conversation_id=conversation_id,
            days=365,
            top_k=min(256, max(32, len(selected_tids) * 2)),
            include_deps=False,
            sort="recency",
            any_tags=[f"turn:{tid}" for tid in selected_tids],
            with_payload=False,
        )
        for it in (selected_hit.get("items") or []):
            entry = _parse_index_text((it.get("text") or "").strip())
            if isinstance(entry, dict):
                selected_docs.append(entry)

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
            assertions=[{
                "key": a.get("key"),
                "value": a.get("value"),
                **({"severity": a.get("severity")} if a.get("severity") else {}),
                **({"scope": a.get("scope")} if a.get("scope") else {}),
                **({"applies_to": a.get("applies_to")} if a.get("applies_to") else {}),
            } for a in (w.get("assertions") or []) if a.get("key")],
            exceptions=[{
                "key": e.get("key"),
                "value": e.get("value"),
                **({"severity": e.get("severity")} if e.get("severity") else {}),
                **({"scope": e.get("scope")} if e.get("scope") else {}),
                **({"applies_to": e.get("applies_to")} if e.get("applies_to") else {}),
            } for e in (w.get("exceptions") or []) if e.get("key")],
            facts=[{"key": f.get("key"), "value": f.get("value")} for f in (w.get("facts") or []) if f.get("key")],
            support_turn_id=w.get("turn_id")
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
