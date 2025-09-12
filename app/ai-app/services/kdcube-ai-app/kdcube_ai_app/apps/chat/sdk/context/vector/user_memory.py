# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/vector/user_memory.py

from __future__ import annotations
import asyncpg, hashlib
from typing import Optional, List, Dict, Any, Sequence, Union
from datetime import datetime, timezone

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.infra.embedding.embedding import convert_embedding_to_string

def _aware(dt: Union[str, datetime]) -> datetime:
    if isinstance(dt, datetime):
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    s = dt.replace("Z", "+00:00") if isinstance(dt, str) and dt.endswith("Z") else dt
    return datetime.fromisoformat(s)

class UserMemoryStore:
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

    async def upsert_fact(
            self,
            *,
            user_id: str,
            fact: str,
            embedding: Optional[List[float]] = None,
            source: str = "user_said",
            strength: float = 0.90,
            expires_at: Optional[Union[str, datetime]] = None,
            tags: Optional[Sequence[str]] = None
    ) -> int:
        """Idempotent(ish) by (user_id, fact_sha1)."""
        sha1 = hashlib.sha1(fact.encode("utf-8")).hexdigest()
        args = [
            user_id, fact, source, float(max(0.0, min(1.0, strength))),
            convert_embedding_to_string(embedding) if embedding else None,
            list(tags or [])
        ]
        q = f"""
        INSERT INTO {self.schema}.user_memory (user_id, fact, source, strength, embedding, tags, fact_sha1)
        VALUES ($1,$2,$3,$4,$5,$6,'{sha1}')
        ON CONFLICT (user_id, fact_sha1) DO UPDATE
        SET fact = EXCLUDED.fact,
            source = EXCLUDED.source,
            strength = GREATEST(LEAST({self.schema}.user_memory.strength * 0.75 + EXCLUDED.strength * 0.25, 1.0), 0.0),
            last_seen_at = now(),
            embedding = COALESCE(EXCLUDED.embedding, {self.schema}.user_memory.embedding),
            tags = (SELECT ARRAY(SELECT DISTINCT UNNEST({self.schema}.user_memory.tags || EXCLUDED.tags)))
        RETURNING id
        """
        async with self._pool.acquire() as con:
            rec = await con.fetchrow(q, *args)
            if expires_at:
                await con.execute(
                    f"UPDATE {self.schema}.user_memory SET expires_at=$1 WHERE id=$2",
                    _aware(expires_at), int(rec["id"])
                )
            return int(rec["id"])

    async def search(
            self,
            *,
            user_id: str,
            query_embedding: Optional[List[float]] = None,
            text_query: Optional[str] = None,
            top_k: int = 6,
            half_life_days: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Hybrid scoring: 0.6*vector + 0.3*BM25 + 0.1*recency-decay."""
        # Build dynamic WHERE
        where = [f"m.user_id = $1"]
        args: List[Any] = [user_id]
        vecpos = None
        if query_embedding is not None:
            vecpos = len(args) + 1
            args.append(convert_embedding_to_string(query_embedding))
            where.append(f"m.embedding IS NOT NULL")
        if text_query:
            qpos = len(args) + 1
            args.append(text_query)
            where.append(f"to_tsvector('simple', m.fact) @@ plainto_tsquery('simple', ${qpos})")
        where.append("(m.expires_at IS NULL OR m.expires_at > now())")

        args.append(str(max(0.1, float(half_life_days))))
        hlpos = len(args)

        vscore = f"(1 - (m.embedding <=> ${vecpos}))" if vecpos else "0.0"
        kscore = "ts_rank(to_tsvector('simple', m.fact), plainto_tsquery('simple', $2))" if text_query else "0.0"

        q = f"""
        WITH base AS (
          SELECT m.*,
                 {vscore} AS vscore,
                 {kscore} AS kscore,
                 EXTRACT(EPOCH FROM (now() - m.last_seen_at)) AS age_sec
          FROM {self.schema}.user_memory_active m
          WHERE {' AND '.join(where)}
        )
        SELECT id, fact, source, strength, tags, last_seen_at,
               (0.60*vscore + 0.30*kscore + 0.10*exp(-ln(2) * age_sec / (${hlpos}::float*24*3600.0))) AS score
        FROM base
        ORDER BY score DESC, last_seen_at DESC
        LIMIT {int(top_k)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, *args)
        return [dict(r) for r in rows]

    async def decay_all(self, *, user_id: str, factor: float = 0.98) -> int:
        """Gentle global decay; call from a periodic job if you like."""
        async with self._pool.acquire() as con:
            res = await con.execute(
                f"UPDATE {self.schema}.user_memory SET strength = GREATEST(strength * $1, 0.0) WHERE user_id=$2",
                float(factor), user_id
            )
            return int(res.split()[-1])
