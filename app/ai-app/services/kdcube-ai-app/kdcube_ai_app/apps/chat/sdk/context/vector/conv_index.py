# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/vector/conv_index.py

import asyncpg
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Sequence, Union, Callable

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.infra.embedding.embedding import convert_embedding_to_string


def _coerce_ts(ts: Union[str, datetime]) -> datetime:
    """Ensure ts is a timezone-aware datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        # Handle ISO8601 with 'Z' or offset
        s = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        return datetime.fromisoformat(s)  # raises if invalid; that’s good
    raise TypeError("ts must be a datetime or ISO8601 string")


class ConvIndex:
    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._settings = get_settings()

        tenant = self._settings.TENANT.replace("-", "_").replace(" ", "_")
        project = self._settings.PROJECT.replace("-", "_").replace(" ", "_")

        schema_name = f"{tenant}_{project}"
        if schema_name and not schema_name.startswith("kdcube_"):
            schema_name = f"kdcube_{schema_name}"

        self.schema: str = schema_name

    async def init(self):
        self._pool = await asyncpg.create_pool(
            host=self._settings.PGHOST, port=self._settings.PGPORT,
            user=self._settings.PGUSER, password=self._settings.PGPASSWORD, database=self._settings.PGDATABASE,
            ssl=self._settings.PGSSL
        )

    async def close(self):
        if self._pool:
            await self._pool.close()

    async def ensure_schema(self):
        sql_raw = (await self._read_sql()).decode()
        sql = sql_raw.replace("<SCHEMA>", self.schema)
        # execute statements defensively
        async with self._pool.acquire() as con:
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                await con.execute(stmt)

    async def _read_sql(self) -> bytes:
        import pkgutil
        # Try both names so packaging can pick either
        data = pkgutil.get_data(__package__, "conversation_history.sql")
        if data:
            return data
        data = pkgutil.get_data(__package__, "deploy-conversation-history.sql")
        if data:
            return data
        raise FileNotFoundError("conversation_history.sql / deploy-conversation-history.sql not found in package")

    async def add_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        role: str,
        text: str,
        s3_uri: str,
        ts: Union[str, datetime],
        tags: Optional[List[str]] = None,
        ttl_days: int = 365,
        user_type: str = "anonymous",
        embedding: Optional[List[float]] = None,
        message_id: Optional[str] = None
    ) -> int:
        ts_dt = _coerce_ts(ts)
        async with self._pool.acquire() as con:
            q = f"""
                INSERT INTO {self.schema}.conv_messages
                    (user_id, conversation_id, message_id, role, text, s3_uri, ts, ttl_days, user_type, tags, embedding)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                RETURNING id
            """
            rec = await con.fetchrow(
                q,
                user_id, conversation_id, message_id, role, text, s3_uri, ts_dt,
                int(ttl_days), user_type, (tags or []),
                convert_embedding_to_string(embedding) if embedding else None
            )
            return int(rec["id"])

    async def search_recent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        query_embedding: List[float],
        top_k: int = 8,
        days: int = 90,
        roles: tuple[str, ...] = ("user", "assistant", "artifact")
    ) -> List[Dict[str, Any]]:
        """
        Single semantic search (cosine) over recent, non-expired rows with embeddings.
        Returns tags so callers can filter locally without extra SQL scans.
        """
        async with self._pool.acquire() as con:
            q = f"""
                SELECT id, message_id, role, text, s3_uri, ts, tags,
                       1 - (embedding <=> $5) AS score
                FROM {self.schema}.conv_messages
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND role = ANY($3)
                  AND ts >= now() - ($4::text || ' days')::interval
                  AND ts + (ttl_days || ' days')::interval >= now()
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $5
                LIMIT {int(top_k)}
            """
            rows = await con.fetch(
                q, user_id, conversation_id, list(roles), str(days),
                convert_embedding_to_string(query_embedding)
            )
            return [dict(r) for r in rows]

    async def search_recent_with_tags(
        self, *,
        user_id: str,
        conversation_id: str,
        query_embedding: List[float],
        top_k: int = 8,
        days: int = 90,
        roles: tuple[str, ...] = ("user", "assistant", "artifact"),
        any_tags: Optional[List[str]] = None,
        all_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Optional server-side tag filtering flavor. Most callers should prefer search_recent
        + Python-side filtering to avoid extra scans.
        """
        args: List[Any] = [
            user_id,                                          # $1
            conversation_id,                                  # $2
            list(roles),                                      # $3
            str(days),                                        # $4
            convert_embedding_to_string(query_embedding)      # $5
        ]
        clauses = [
            "user_id = $1",
            "conversation_id = $2",
            "role = ANY($3)",
            "ts >= now() - ($4::text || ' days')::interval",
            "ts + (ttl_days || ' days')::interval >= now()",   # TTL guard
            "embedding IS NOT NULL"
        ]

        if any_tags:
            args.append(any_tags)
            clauses.append(f"tags && ${len(args)}")

        if all_tags:
            args.append(all_tags)
            idx = len(args)
            # require all tags from parameter array to appear in tags column
            clauses.append(
                f"(SELECT COUNT(*) FROM unnest(${idx}) t WHERE t = ANY(tags)) = array_length(${idx}, 1)"
            )

        q = f"""
            SELECT id, message_id, role, text, s3_uri, ts, tags,
                   1 - (embedding <=> $5) AS score
            FROM {self.schema}.conv_messages
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding <=> $5
            LIMIT {int(top_k)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, *args)
        return [dict(r) for r in rows]

    async def purge_user_type(self, *, user_type: str, older_than_days: Optional[int] = None) -> int:
        """Quick purge by cohort. If older_than_days is None, delete all rows for that user_type."""
        async with self._pool.acquire() as con:
            if older_than_days is None:
                q = f"DELETE FROM {self.schema}.conv_messages WHERE user_type = $1"
                res = await con.execute(q, user_type)
            else:
                q = f"""
                    DELETE FROM {self.schema}.conv_messages
                    WHERE user_type = $1
                      AND ts < now() - ($2::text || ' days')::interval
                """
                res = await con.execute(q, user_type, str(older_than_days))
            return int(res.split()[-1])

    async def purge_expired(self) -> int:
        """TTL-based purge (same criteria as the cleanup job)."""
        async with self._pool.acquire() as con:
            q = f"""
                DELETE FROM {self.schema}.conv_messages
                WHERE ts + (ttl_days || ' days')::interval < now()
            """
            res = await con.execute(q)
            return int(res.split()[-1])

    async def backfill_from_store(
        self,
        *,
        records: List[Dict[str, Any]],
        default_ttl_days: int = 365,
        default_user_type: str = "anonymous",
        embedder: Optional[Callable[[str], List[float]]] = None
    ) -> int:
        """
        Re-index previously persisted messages. Each record is a JSON dict as written by ConversationStore.
        If 'embedding' is present in the record, it is used; otherwise, if 'embedder' is provided, it is called.
        """
        n = 0
        for rec in records:
            meta = rec.get("meta") or {}
            emb = rec.get("embedding") or meta.get("embedding")
            if emb is None and embedder:
                try:
                    emb = embedder(rec.get("text") or "")
                except Exception:
                    emb = None
            ttl_days = int(meta.get("ttl_days", default_ttl_days))
            user_type = str(meta.get("user_type", default_user_type))
            try:
                await self.add_message(
                    user_id=rec.get("user") or "anonymous",
                    conversation_id=rec.get("conversation_id"),
                    message_id=meta.get("message_id"),
                    role=rec.get("role"),
                    text=rec.get("text") or "",
                    s3_uri=meta.get("s3_uri") or "",
                    ts=rec.get("timestamp") or meta.get("timestamp") or datetime.utcnow(),
                    ttl_days=ttl_days,
                    user_type=user_type,
                    tags=meta.get("tags") or [],
                    embedding=emb
                )
                n += 1
            except Exception:
                continue
        return n
