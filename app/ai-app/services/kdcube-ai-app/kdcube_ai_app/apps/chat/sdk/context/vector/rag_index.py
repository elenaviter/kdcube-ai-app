# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/vector/rag_index.py

from __future__ import annotations
import asyncpg, hashlib
from typing import Optional, List, Dict, Any, Sequence
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.infra.embedding.embedding import convert_embedding_to_string

class RAGIndex:
    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._settings = get_settings()
        tenant = self._settings.TENANT.replace("-", "_").replace(" ", "_")
        project = self._settings.PROJECT.replace("-", "_").replace(" ", "_")
        schema = f"{'kdcube_' if not f'{tenant}_{project}'.startswith('kdcube_') else ''}{tenant}_{project}"
        self.schema = schema

    async def init(self):
        self._pool = await asyncpg.create_pool(
            host=self._settings.PGHOST, port=self._settings.PGPORT,
            user=self._settings.PGUSER, password=self._settings.PGPASSWORD,
            database=self._settings.PGDATABASE, ssl=self._settings.PGSSL
        )

    async def close(self):
        if self._pool: await self._pool.close()

    async def upsert_chunk(
            self,
            *,
            corpus: str,
            source_id: Optional[str],
            chunk: str,
            embedding: Optional[Sequence[float]],
            metadata: Optional[Dict[str, Any]] = None,
            expires_at: Optional[str] = None
    ) -> int:
        sha1 = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
        args = [
            corpus, source_id, chunk, convert_embedding_to_string(list(embedding)) if embedding else None,
            metadata or {}
        ]
        q = f"""
        INSERT INTO {self.schema}.rag_chunks (corpus, source_id, chunk, embedding, metadata, chunk_sha1)
        VALUES ($1,$2,$3,$4,$5,'{sha1}')
        ON CONFLICT (corpus, source_id, chunk_sha1) DO UPDATE
        SET chunk = EXCLUDED.chunk,
            embedding = COALESCE(EXCLUDED.embedding, {self.schema}.rag_chunks.embedding),
            metadata = EXCLUDED.metadata,
            created_at = now()
        RETURNING id
        """
        async with self._pool.acquire() as con:
            rec = await con.fetchrow(q, *args)
            if expires_at:
                await con.execute(f"UPDATE {self.schema}.rag_chunks SET expires_at=$1 WHERE id=$2", expires_at, int(rec["id"]))
            return int(rec["id"])

    async def delete_by_source(self, *, corpus: str, source_id: str) -> int:
        async with self._pool.acquire() as con:
            res = await con.execute(
                f"DELETE FROM {self.schema}.rag_chunks WHERE corpus=$1 AND source_id=$2",
                corpus, source_id
            )
            return int(res.split()[-1])

    async def hybrid_search(
            self,
            *,
            query_embedding: Optional[Sequence[float]],
            query_text: Optional[str],
            corpus: Optional[str] = None,
            top_k: int = 25,
            filter_meta_contains: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        where = ["(r.expires_at IS NULL OR r.expires_at > now())"]
        args: List[Any] = []
        if corpus:
            args.append(corpus); where.append(f"r.corpus = ${len(args)}")
        vecpos = None
        if query_embedding is not None:
            vecpos = len(args) + 1
            args.append(convert_embedding_to_string(list(query_embedding)))
            where.append("r.embedding IS NOT NULL")
        textpos = None
        if query_text:
            textpos = len(args) + 1
            args.append(query_text)
            where.append("to_tsvector('english', r.chunk) @@ plainto_tsquery('english', $" + str(textpos) + ")")
        if filter_meta_contains:
            args.append(filter_meta_contains)
            where.append(f"r.metadata @> ${len(args)}::jsonb")

        vscore = f"(1 - (r.embedding <=> ${vecpos}))" if vecpos else "0.0"
        kscore = f"ts_rank(to_tsvector('english', r.chunk), plainto_tsquery('english', ${textpos}))" if textpos else "0.0"

        q = f"""
        WITH base AS (
          SELECT r.*,
                 {vscore} AS vscore,
                 {kscore} AS kscore
          FROM {self.schema}.rag_chunks_active r
          WHERE {' AND '.join(where)}
        )
        SELECT id, corpus, source_id, chunk, metadata,
               (0.65*vscore + 0.35*kscore) AS score
        FROM base
        ORDER BY score DESC, created_at DESC
        LIMIT {int(top_k)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, *args)
        return [dict(r) for r in rows]
