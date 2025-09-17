# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/retrieval/ctx_rag.py

from __future__ import annotations

import datetime
import pathlib, json
from typing import Optional, Sequence, List, Dict, Any

from kdcube_ai_app.apps.chat.sdk.inventory import ModelServiceBase
from kdcube_ai_app.apps.chat.sdk.runtime.scratchpad import TurnLog

from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore
from kdcube_ai_app.apps.chat.sdk.context.vector.conv_index import ConvIndex

TURN_LOG_TAGS_BASE = ["kind:turn.log", "artifact:turn.log"]

class ContextRAGClient:
    def __init__(self, *,
                 conv_idx: ConvIndex,
                 store: ConversationStore,
                 model_service: ModelServiceBase,
                 default_ctx_path: Optional[str] = None):
        self.idx = conv_idx
        self.store = store
        self.model_service = model_service
        self.default_ctx_path = default_ctx_path or "context.json"

    def _load_ctx(self, ctx: Optional[dict]) -> dict:
        if ctx is not None:
            return ctx
        p = pathlib.Path(self.default_ctx_path)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        from kdcube_ai_app.infra.accounting import _get_context
        context = _get_context()
        context_snapshot = context.to_dict()
        return context_snapshot

    def _scope_from_ctx(self, ctx: dict, *, user_id=None, conversation_id=None, track_id=None) -> tuple[str,str,str]:
        user = user_id or ctx.get("user_id")
        conv = conversation_id or ctx.get("conversation_id")
        track = track_id or ctx.get("track_id")
        return user, conv, track

    # ---------- public API ----------

    async def search(
            self,
            *,
            query: Optional[str],
            embedding: Optional[Sequence[float]] = None,
            kinds: Optional[Sequence[str]] = None,
            scope: str = "track",
            days: int = 90,
            top_k: int = 12,
            include_deps: bool = True,
            half_life_days: float = 7.0,
            ctx: Optional[dict] = None,
            user_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            track_id: Optional[str] = None,
            roles: tuple[str,...] = ("artifact","assistant","user"),
            with_payload: bool = False,
            sort: str = "hybrid",
    ) -> dict:
        """
        Semantic/Hybrid search (needs embedding unless provided).
        """
        ctx_loaded = self._load_ctx(ctx)
        user, conv, track = self._scope_from_ctx(ctx_loaded, user_id=user_id, conversation_id=conversation_id, track_id=track_id)

        qvec = list(embedding) if embedding is not None else None
        if qvec is None and query:
            [qvec] = await self.model_service.embed_texts([query])
        if qvec is None:
            # If caller truly wants recency-only, use .recent() instead of .search().
            raise ValueError("search() needs either 'embedding' or 'query' to create one. For recency, call recent().")

        rows = await self.idx.search_context(
            user_id=user,
            conversation_id=(conv or None),
            track_id=(track or None),
            query_embedding=qvec,
            top_k=top_k,
            days=days,
            scope=scope,
            roles=roles,
            kinds=kinds,
            half_life_days=half_life_days,
            include_deps=include_deps,
            sort=sort,
        )

        items = []
        for r in rows:
            item = {
                "id": r["id"],
                "message_id": r["message_id"],
                "role": r["role"],
                "text": r.get("text") or "",
                "ts": r["ts"].isoformat() if hasattr(r["ts"], "isoformat") else r["ts"],
                "tags": list(r.get("tags") or []),
                "score": float(r.get("score") or 0.0),
                "sim": float(r.get("sim") or 0.0),
                "rec": float(r.get("rec") or 0.0),
                "track_id": r.get("track_id"),
                "s3_uri": r.get("s3_uri"),
            }
            if include_deps and "deps" in r:
                item["deps"] = r["deps"]
            if with_payload:
                try:
                    doc = self.store.get_message(r["s3_uri"])
                    item["payload"] = doc
                except Exception:
                    pass
            items.append(item)
        return {"items": items}

    async def recent(
            self,
            *,
            kinds: Optional[Sequence[str]] = None,
            scope: str = "track",
            days: int = 90,
            limit: int = 12,
            ctx: Optional[dict] = None,
            user_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            track_id: Optional[str] = None,
            roles: tuple[str, ...] = ("artifact","assistant","user"),
            any_tags: Optional[Sequence[str]] = None,
            all_tags: Optional[Sequence[str]] = None,
            not_tags: Optional[Sequence[str]] = None,
            with_payload: bool = False,
    ) -> dict:
        """
        Pure-recency fetch (no embeddings). Fast path for "last N in track".
        """
        ctx_loaded = self._load_ctx(ctx)
        user, conv, track = self._scope_from_ctx(ctx_loaded, user_id=user_id, conversation_id=conversation_id, track_id=track_id)
        any_tags = list(any_tags or [])
        if kinds:
            any_tags += list(kinds)
            all_tags += list(kinds)
        rows = await self.idx.fetch_recent(
            user_id=user,
            conversation_id=(conv or None),
            track_id=(track or None),
            roles=roles,
            any_tags=any_tags or None,
            all_tags=list(all_tags or []) or None,
            not_tags=list(not_tags or []) or None,
            limit=limit,
            days=days
        )
        items = []
        for r in rows:
            item = {
                "id": r["id"],
                "message_id": r["message_id"],
                "role": r["role"],
                "text": r.get("text") or "",
                "ts": r["ts"].isoformat() if hasattr(r["ts"], "isoformat") else r["ts"],
                "tags": list(r.get("tags") or []),
                "track_id": r.get("track_id"),
                "s3_uri": r.get("s3_uri"),
            }
            if with_payload:
                try:
                    doc = self.store.get_message(r["s3_uri"])
                    item["payload"] = doc
                except Exception:
                    pass
            items.append(item)
        return {"items": items}

    async def pull_text_artifact(self, *, artifact_uri: str) -> dict:
        doc = self.store.get_message(artifact_uri)
        return doc.get("payload") or {}

    async def decide_reuse(
            self,
            *,
            goal_kind: str,
            query: str,
            threshold: float = 0.78,
            days: int = 180,
            scope: str = "track",
            ctx: Optional[dict] = None
    ) -> dict:
        kinds = (goal_kind,)
        res = await self.search(query=query, kinds=kinds, scope=scope, days=days, top_k=5, include_deps=True, ctx=ctx)
        items = res.get("items") or []
        if not items:
            return {"reuse": False, "search": res}

        top = items[0]
        sim = float(top.get("sim") or 0.0)
        rec = float(top.get("rec") or 0.0)
        ok = (sim >= threshold) or (sim >= (threshold - 0.05) and rec >= 0.60)

        reason = f"sim={sim:.2f}, rec={rec:.2f}, goal_kind={goal_kind}"
        out = {"reuse": bool(ok), "search": res}
        if ok:
            out["candidate"] = {
                "message_id": top["message_id"],
                "id": top["id"],
                "score": float(top.get("score") or 0.0),
                "sim": sim,
                "rec": rec,
                "reason": reason
            }
        else:
            out["reason"] = reason
        return out

    async def save_turn_log_as_artifact(
            self,
            *,
            tenant: str, project: str, user: str,
            conversation_id: str, user_type: str,
            turn_id: str, track_id: Optional[str],
            log: TurnLog,
            extra_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Writes markdown to store (assistant artifact) + indexes it."""
        md = log.to_markdown()
        payload = {"turn_log": log.to_payload()}

        tags = TURN_LOG_TAGS_BASE + [f"turn:{turn_id}"] + ([f"track:{track_id}"] if track_id else [])
        if extra_tags:
            tags.extend([t for t in extra_tags if isinstance(t, str) and t.strip()])
        s3_uri, message_id, rn = self.store.put_message(
            tenant=tenant, project=project, user=user, fingerprint=None,
            conversation_id=conversation_id, role="artifact", text=md,
            payload=payload,
            meta={"kind": "turn.log", "turn_id": turn_id, "track_id": track_id},
            embedding=None, user_type=user_type, turn_id=turn_id, track_id=track_id,
        )
        await self.idx.add_message(
            user_id=user, conversation_id=conversation_id, role="artifact",
            text=md, s3_uri=s3_uri, ts=log.started_at_iso,
            tags=tags,
            ttl_days=365, user_type=user_type, embedding=None, message_id=message_id, track_id=track_id
        )
        return {"s3_uri": s3_uri, "message_id": message_id, "rn": rn}

    async def materialize_turn(
            self,
            *,
            turn_id: str,
            scope: str = "track",
            days: int = 365,
            ctx: Optional[dict] = None,
            user_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            track_id: Optional[str] = None,
            with_payload: bool = True
    ) -> dict:
        """
        Returns the user msg, assistant reply, and user-visible artifacts for the turn.
        Visible artifacts include:
          - codegen.program.presentation (project canvas / draft)
          - codegen.program.out.deliverables
        """
        # 1) user
        u = await self.recent(
            scope=scope, days=days, limit=1, ctx=ctx,
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=("user",), all_tags=[f"turn:{turn_id}"], with_payload=with_payload
        )
        # 2) assistant
        a = await self.recent(
            scope=scope, days=days, limit=1, ctx=ctx,
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=("assistant",), all_tags=[f"turn:{turn_id}"], with_payload=with_payload
        )
        # 3) presentation (draft the user saw)
        prez = await self.recent(
            kinds=("artifact:codegen.program.presentation",),  # meta.kind
            scope=scope, days=days, limit=1, ctx=ctx,
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=("artifact",), all_tags=[f"turn:{turn_id}"], with_payload=with_payload
        )
        # 4) deliverables (file list user could download)
        dels = await self.recent(
            kinds=("artifact:codegen.program.out.deliverables",),
            scope=scope, days=days, limit=1, ctx=ctx,
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=("artifact",), all_tags=[f"turn:{turn_id}"], with_payload=with_payload
        )
        # 5) errors
        solver_failure = await self.recent(
            kinds=("artifact:solver:failure",),
            scope=scope, days=days, limit=1, ctx=ctx,
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=("artifact",), all_tags=[f"turn:{turn_id}"], with_payload=with_payload
        )
        # 6) citables
        citables = await self.recent(
            kinds=("artifact:codegen.program.citables",),
            scope=scope, days=days, limit=3, ctx=ctx,
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=("artifact",), all_tags=[f"turn:{turn_id}"], with_payload=with_payload
        )
        def first(results: dict) -> Optional[dict]:
            arr = next(iter(results.get("items") or []), None)
            return arr
            # return arr.get("payload") if arr else None

        return {
            "user": first(u),
            "assistant": first(a),
            "presentation": first(prez),
            "deliverables": first(dels),
            "citables": first(citables),
            "solver_failure": first(solver_failure)
        }

    async def append_reaction_to_turn_log(self, *,
                                          turn_id: str, reaction: str,
                                          tenant: str, project: str, user: str,
                                          fingerprint: Optional[str],
                                          user_type: str, conversation_id: str, track_id: str):

        payload = {"reaction": {"text": reaction, "ts": datetime.datetime.utcnow().isoformat()+"Z"}}
        # persist as a small artifact tied to the same turn (don’t overwrite)
        s3_uri, message_id, rn = self.store.put_message(
            tenant=tenant, project=project, user=user,
            conversation_id=conversation_id, role="artifact",
            text=f"[turn.log.reaction]\n{reaction}",
            payload=payload,
            meta={"kind": "turn.log.reaction", "turn_id": turn_id, "track_id": track_id},
            embedding=None, user_type=user_type, turn_id=turn_id, track_id=track_id,
            fingerprint=fingerprint
        )
        await self.idx.add_message(
            user_id=user, conversation_id=conversation_id, role="artifact",
            text=f"[turn.log.reaction] {reaction}", s3_uri=s3_uri, ts=payload["reaction"]["ts"],
            tags=["kind:turn.log.reaction", f"turn:{turn_id}", f"track:{track_id}"],
            ttl_days=365, user_type=user_type, embedding=None, message_id=message_id, track_id=track_id
        )

