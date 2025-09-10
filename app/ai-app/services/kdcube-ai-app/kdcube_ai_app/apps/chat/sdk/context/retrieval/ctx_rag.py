# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/retrieval/ctx_rag.py

from __future__ import annotations

import pathlib, json
from typing import Optional, Sequence

from kdcube_ai_app.apps.chat.sdk.inventory import ModelServiceBase

from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore
from kdcube_ai_app.apps.chat.sdk.context.vector.conv_index import ConvIndex

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
        return {}

    def _scope_from_ctx(self, ctx: dict, *, user_id=None, conversation_id=None, track_id=None) -> tuple[str,str,str]:
        user = user_id or ctx.get("user_id") or ctx.get("user") or ""
        conv = conversation_id or ctx.get("conversation_id") or ctx.get("session_id") or ""
        track = track_id or ctx.get("track_id") or ""
        return user, conv, track

    # ---------- public API (what codegen calls) ----------

    async def search(
            self,
            *,
            query: str,
            kinds: Optional[Sequence[str]] = None,
            scope: str = "track",            # 'track' | 'conversation' | 'user'
            days: int = 90,
            top_k: int = 12,
            include_deps: bool = True,
            half_life_days: float = 7.0,
            ctx: Optional[dict] = None,
            user_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            track_id: Optional[str] = None,
            roles: tuple[str,...] = ("artifact","assistant","user"),
            with_payload: bool = False       # if True, fetch payloads via store
    ) -> dict:
        """
        Context search for codegen & turn-start.
        - query: user intent or artifact description
        - kinds: artifact kinds to prefer (match by tags), e.g. ("codegen.program.presentation","codegen.out.inline")
        - scope: default 'track' → narrows noise without losing continuity
        - include_deps: return direct dependencies for artifacts (id/message_id/preview/policy)
        - with_payload: if True, we also load slow-storage payloads for hits (costly)
        Returns: { items: [ {id,message_id,role,text,ts,tags,score,sim,rec,track_id,s3_uri,deps?,payload?} ] }
        """
        ctx_loaded = self._load_ctx(ctx)
        user, conv, track = self._scope_from_ctx(ctx_loaded, user_id=user_id, conversation_id=conversation_id, track_id=track_id)

        [qvec] = await self.model_service.embed_texts([query])
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
            include_deps=include_deps
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
                # deps is a json array of lightweight items
                item["deps"] = r["deps"]
            if with_payload:
                try:
                    doc = self.store.get_message(r["s3_uri"])  # (message_id is your store key)
                    item["payload"] = doc  # contains meta, payload, etc.
                except Exception:
                    pass
            items.append(item)
        return {"items": items}

    async def pull_text_artifact(self, *, artifact_uri: str) -> dict:
        """Return the full stored record for an artifact/message (payload+meta+text)."""
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
        """
        Heuristic: find an artifact of 'goal_kind' that is similar to 'query' and recent.
        If top hit has (sim >= threshold) OR (sim >= threshold-0.05 AND rec >= 0.6), suggest reuse.
        Returns { reuse: bool, candidate?: {message_id, id, score, reason}, search?: {...} }
        """
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