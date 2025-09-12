# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/vector/conv_index.py

import asyncpg
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Sequence, Union, Callable, Iterable

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
        message_id: Optional[str] = None,
        track_id: Optional[str] = None                       # NEW
    ) -> int:
        ts_dt = _coerce_ts(ts)
        async with self._pool.acquire() as con:
            q = f"""
                INSERT INTO {self.schema}.conv_messages
                    (user_id, conversation_id, message_id, role, text, s3_uri, ts, ttl_days, user_type, tags, embedding, track_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                RETURNING id
            """
            rec = await con.fetchrow(
                q,
                user_id, conversation_id, message_id, role, text, s3_uri, ts_dt,
                int(ttl_days), user_type, (tags or []),
                convert_embedding_to_string(embedding) if embedding else None,
                track_id
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
        roles: tuple[str, ...] = ("user", "assistant", "artifact"),
        track_id: Optional[str] = None                         # NEW
    ) -> List[Dict[str, Any]]:
        args = [user_id, conversation_id, list(roles), str(days), convert_embedding_to_string(query_embedding)]
        where = [
            "user_id = $1",
            "conversation_id = $2",
            "role = ANY($3)",
            "ts >= now() - ($4::text || ' days')::interval",
            "ts + (ttl_days || ' days')::interval >= now()",
            "embedding IS NOT NULL"
        ]
        if track_id:
            args.append(track_id)
            where.append(f"track_id = ${len(args)}")

        q = f"""
            SELECT id, message_id, role, text, s3_uri, ts, tags, track_id,
                   1 - (embedding <=> $5) AS score
            FROM {self.schema}.conv_messages
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> $5
            LIMIT {int(top_k)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, *args)
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
        all_tags: Optional[List[str]] = None,
        track_id: Optional[str] = None                         # NEW
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
            args.append(any_tags); clauses.append(f"tags && ${len(args)}")
        if all_tags:
            args.append(all_tags); idx = len(args)
            clauses.append(f"(SELECT COUNT(*) FROM unnest(${idx}) t WHERE t = ANY(tags)) = array_length(${idx},1)")
        if track_id:
            args.append(track_id); clauses.append(f"track_id = ${len(args)}")

        q = f"""
            SELECT id, message_id, role, text, s3_uri, ts, tags, track_id,
                   1 - (embedding <=> $5) AS score
            FROM {self.schema}.conv_messages
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding <=> $5
            LIMIT {int(top_k)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, *args)
        return [dict(r) for r in rows]

    async def search_recent_with_turn_pairs(
            self,
            *,
            user_id: str,
            conversation_id: str,
            query_embedding: List[float],
            top_k: int = 8,
            days: int = 90,
            track_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns nearest neighbors, each augmented with the *closest prior user*
        and the *closest following assistant* message within the same conversation.
        Pairing is done on the DB side via LATERAL joins.
        """
        args = [
            user_id, conversation_id, str(days),
            convert_embedding_to_string(query_embedding)
        ]
        where = [
            "m.user_id = $1",
            "m.conversation_id = $2",
            "m.ts >= now() - ($3::text || ' days')::interval",
            "m.ts + (m.ttl_days || ' days')::interval >= now()",
            "m.embedding IS NOT NULL"
        ]
        if track_id:
            args.append(track_id)
            where.append(f"m.track_id = ${len(args)}")

        q = f"""
        WITH hits AS (
          SELECT
            m.*,
            1 - (m.embedding <=> $4) AS score
          FROM {self.schema}.conv_messages m
          WHERE {' AND '.join(where)}
          ORDER BY m.embedding <=> $4
          LIMIT {int(top_k)}
        )
        SELECT
          h.id AS hit_id, h.role AS hit_role, h.text AS hit_text, h.s3_uri AS hit_s3_uri,
          h.ts AS hit_ts, h.tags AS hit_tags, h.track_id AS hit_track_id, h.score AS hit_score,

          u.id AS user_id_msg, u.text AS user_text, u.s3_uri AS user_s3_uri, u.ts AS user_ts,
          a.id AS assistant_id_msg, a.text AS assistant_text, a.s3_uri AS assistant_s3_uri, a.ts AS assistant_ts

        FROM hits h
        LEFT JOIN LATERAL (
          SELECT u.*
          FROM {self.schema}.conv_messages u
          WHERE u.user_id = h.user_id
            AND u.conversation_id = h.conversation_id
            AND u.role = 'user'
            AND u.ts <= h.ts
          ORDER BY u.ts DESC
          LIMIT 1
        ) u ON TRUE
        LEFT JOIN LATERAL (
          SELECT a.*
          FROM {self.schema}.conv_messages a
          WHERE a.user_id = h.user_id
            AND a.conversation_id = h.conversation_id
            AND a.role = 'assistant'
            AND a.ts >= COALESCE(u.ts, h.ts)
          ORDER BY a.ts ASC
          LIMIT 1
        ) a ON TRUE
        ORDER BY h.score DESC, h.ts DESC
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

    async def add_edges_by_id(self, *, from_id: int, to_ids: Iterable[int], policy: str = "none") -> int:
        rows = [(int(from_id), int(t), policy) for t in to_ids if t and int(t) != int(from_id)]
        if not rows: return 0
        q = f"INSERT INTO {self.schema}.conv_artifact_edges (from_id, to_id, policy) VALUES ($1,$2,$3) ON CONFLICT DO NOTHING"
        async with self._pool.acquire() as con:
            await con.executemany(q, rows)
        return len(rows)

    async def search_context(
            self,
            *,
            user_id: str,
            conversation_id: Optional[str],
            track_id: Optional[str],
            query_embedding: list[float],
            top_k: int = 12,
            days: int = 90,
            scope: str = "track",                     # 'track' | 'conversation' | 'user'
            roles: tuple[str, ...] = ("user", "assistant", "artifact"),
            any_tags: Optional[Sequence[str]] = None, # OR
            all_tags: Optional[Sequence[str]] = None, # AND (tags @> array)
            not_tags: Optional[Sequence[str]] = None, # NOT (tags && array)
            text_query: Optional[str] = None,         # ILIKE '%q%' (trgm indexed)
            kinds: Optional[Sequence[str]] = None,    # sugar: expands to any_tags += kinds
            half_life_days: float = 7.0,
            include_deps: bool = True,
            sort: str = "hybrid"                      # 'hybrid' | 'semantic' | 'recency'
    ) -> list[dict]:
        if kinds:
            any_tags = (list(any_tags or []) + list(kinds))
        args = [user_id, list(roles), str(days), convert_embedding_to_string(query_embedding)]
        where = [
            "m.user_id = $1",
            "m.role = ANY($2)",
            "m.ts >= now() - ($3::text || ' days')::interval",
            "m.ts + (m.ttl_days || ' days')::interval >= now()",
            "m.embedding IS NOT NULL"
        ]
        # scope
        if scope == "track" and track_id:
            args.append(track_id); where.append(f"m.track_id = ${len(args)}")
            if conversation_id:
                args.append(conversation_id); where.append(f"m.conversation_id = ${len(args)}")
        elif scope == "conversation" and conversation_id:
            args.append(conversation_id); where.append(f"m.conversation_id = ${len(args)}")
        # tags
        if any_tags:
            args.append(list(any_tags)); where.append(f"m.tags && ${len(args)}")
        if all_tags:
            args.append(list(all_tags)); where.append(f"m.tags @> ${len(args)}::text[]")
        if not_tags:
            args.append(list(not_tags)); where.append(f"NOT (m.tags && ${len(args)})")
        # text match (optional)
        if text_query:
            args.append(f"%{text_query}%"); where.append(f"m.text ILIKE ${len(args)}")

        # scoring + optional deps
        args.append(str(max(0.1, float(half_life_days))))
        half_life_days_s = f"(${len(args)}::text)::float"

        order_by = {
            "semantic": "sim DESC, m.ts DESC",
            "recency":  "m.ts DESC",
            "hybrid":   f"(0.70*sim + 0.25*exp(-ln(2) * age_sec / ({half_life_days_s}*24*3600.0)) + 0.05*rboost) DESC, m.ts DESC"
        }.get(sort, "m.ts DESC")

        deps_select, deps_join = "", ""
        if include_deps:
            deps_select = ", COALESCE(d.deps, '[]'::json) AS deps"
            deps_join = f"""
              LEFT JOIN LATERAL (
                SELECT COALESCE(json_agg(
                  json_build_object(
                    'id', cm2.id, 'message_id', cm2.message_id, 'role', cm2.role,
                    'tags', cm2.tags, 's3_uri', cm2.s3_uri, 'ts', cm2.ts,
                    'policy', e.policy,
                    'text_preview', CASE WHEN cm2.text IS NULL THEN NULL ELSE left(cm2.text, 400) END
                  )
                  ORDER BY cm2.ts ASC
                ), '[]'::json) AS deps
                FROM {self.schema}.conv_artifact_edges e
                JOIN {self.schema}.conv_messages cm2 ON cm2.id = e.to_id
                WHERE e.from_id = m.id
              ) d ON TRUE
            """

        q = f"""
          WITH base AS (
            SELECT m.*,
                   1 - (m.embedding <=> $4) AS sim,
                   EXTRACT(EPOCH FROM (now() - m.ts)) AS age_sec,
                   CASE m.role WHEN 'artifact' THEN 1.10 WHEN 'assistant' THEN 1.00 ELSE 0.98 END AS rboost
            FROM {self.schema}.conv_messages m
            WHERE {' AND '.join(where)}
          )
          SELECT m.id, m.message_id, m.role, m.text, m.s3_uri, m.ts, m.tags, m.track_id,
                 m.sim, m.age_sec, m.rboost,
                 (0.70*m.sim + 0.25*exp(-ln(2) * m.age_sec / ({half_life_days_s}*24*3600.0)) + 0.05*m.rboost) AS score
                 {deps_select}
          FROM base m
          {deps_join}
          ORDER BY {order_by}
          LIMIT {int(top_k)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, *args)
        return [dict(r) for r in rows]

    async def fetch_recent(
            self,
            *,
            user_id: str,
            conversation_id: Optional[str] = None,
            track_id: Optional[str] = None,
            roles: tuple[str, ...] = ("user", "assistant", "artifact"),
            any_tags: Optional[Sequence[str]] = None,
            all_tags: Optional[Sequence[str]] = None,
            not_tags: Optional[Sequence[str]] = None,
            limit: int = 30,
            days: int = 30
    ) -> list[dict]:
        args = [user_id, list(roles), str(days)]
        where = [
            "user_id = $1",
            "role = ANY($2)",
            "ts >= now() - ($3::text || ' days')::interval",
            "ts + (ttl_days || ' days')::interval >= now()"
        ]
        if track_id:
            args.append(track_id); where.append(f"track_id = ${len(args)}")
            if conversation_id:
                args.append(conversation_id); where.append(f"conversation_id = ${len(args)}")
        elif conversation_id:
            args.append(conversation_id); where.append(f"conversation_id = ${len(args)}")
        if any_tags:
            args.append(list(any_tags)); where.append(f"tags && ${len(args)}")
        if all_tags:
            args.append(list(all_tags)); where.append(f"tags @> ${len(args)}::text[]")
        if not_tags:
            args.append(list(not_tags)); where.append(f"NOT (tags && ${len(args)})")

        q = f"""
          SELECT id, message_id, role, text, s3_uri, ts, tags, track_id
          FROM {self.schema}.conv_messages
          WHERE {' AND '.join(where)}
          ORDER BY ts DESC
          LIMIT {int(limit)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, *args)
        return [dict(r) for r in rows]

    async def hybrid_context(
            self,
            *,
            user_id: str,
            conversation_id: str,
            query_embedding: list[float],
            track_id: Optional[str],
            recent_limit: int = 30,     # deterministic
            recent_days: int = 30,
            semantic_top_k: int = 12,   # add older-but-relevant
            semantic_days: int = 365,
            roles: tuple[str, ...] = ("user","assistant","artifact"),
            topic_tags: Optional[Sequence[str]] = None
    ) -> list[dict]:
        # A) recent window (recency only)
        recent = await self.fetch_recent(
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=roles, any_tags=topic_tags, limit=recent_limit, days=recent_days
        )
        seen_ids = {r["id"] for r in recent}

        # B) semantic extras (exclude those already present)
        sem = await self.search_context(
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            query_embedding=query_embedding, top_k=semantic_top_k, days=semantic_days,
            scope=("track" if track_id else "conversation"),
            roles=roles, any_tags=topic_tags, include_deps=True, sort="hybrid"
        )
        sem = [r for r in sem if r["id"] not in seen_ids]

        # Merge: newest-first within recent, then sem by score
        return recent + sem

    async def find_last_deliverable_for_mention(
            self,
            *,
            user_id: str,
            conversation_id: str,
            track_id: Optional[str],
            mention: str,
            mention_emb: Optional[list[float]],
            prefer_kinds: Sequence[str] = ("codegen.out.inline","codegen.program.presentation","codegen.out.file"),
            window_limit: int = 40
    ) -> Optional[dict]:
        # 1) Look in recent window for deliverables
        recent = await self.fetch_recent(
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            roles=("artifact",), any_tags=list(prefer_kinds), limit=window_limit, days=90
        )
        # 2) If mention text looks exact-ish, try ILIKE in-window first
        if mention and len(mention) >= 3:
            txt = mention.strip()
            for r in recent:
                if txt.lower() in (r.get("text") or "").lower():
                    return r

        # 3) Otherwise do semantic over deliverable kinds
        qvec = mention_emb
        sem = await self.search_context(
            user_id=user_id, conversation_id=conversation_id, track_id=track_id,
            query_embedding=qvec, top_k=8, days=365, scope=("track" if track_id else "conversation"),
            roles=("artifact",), any_tags=list(prefer_kinds), include_deps=True, sort="hybrid"
        )
        return sem[0] if sem else (recent[0] if recent else None)


    async def fetch_latest_summary(self, *, user_id: str, conversation_id: str, kind: str = "conversation.summary") -> Optional[Dict[str, Any]]:
        q = f"""
          SELECT id, message_id, role, text, s3_uri, ts, tags, track_id
          FROM {self.schema}.conv_messages
          WHERE user_id=$1 AND conversation_id=$2
            AND role='artifact' AND tags @> ARRAY[$3]::text[]
          ORDER BY ts DESC
          LIMIT 1
        """
        async with self._pool.acquire() as con:
            row = await con.fetchrow(q, user_id, conversation_id, f"kind:{kind}")
        return dict(row) if row else None

    async def fetch_last_turn_logs(self, *, user_id: str, conversation_id: str, max_turns: int = 3) -> list[dict]:
        """
        Returns newest-first up to one log per turn (tagged 'kind:turn.log').
        """
        q = f"""
        WITH cand AS (
          SELECT id, message_id, role, text, s3_uri, ts, tags, track_id
          FROM {self.schema}.conv_messages
          WHERE user_id=$1 AND conversation_id=$2
            AND role='artifact' AND tags @> ARRAY['kind:turn.log']::text[]
          ORDER BY ts DESC
        )
        SELECT DISTINCT ON ( (SELECT t FROM regexp_matches(unnest(tags)::text, '^turn:(.+)$') AS r(t) LIMIT 1) )
               id, message_id, role, text, s3_uri, ts, tags, track_id
        FROM cand
        ORDER BY (SELECT t FROM regexp_matches(unnest(tags)::text, '^turn:(.+)$') AS r(t) LIMIT 1), ts DESC
        LIMIT {int(max_turns)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, user_id, conversation_id)
        # newest-first
        return [dict(r) for r in sorted(rows, key=lambda r: r["ts"], reverse=True)]

    async def load_turn_prefs(self, *, user_id: str, conversation_id: str, turn_id: str) -> dict:
        q1 = f"""SELECT key, value_json, desired, confidence FROM {self.schema}.conv_prefs
                 WHERE user_id=$1 AND conversation_id=$2 AND turn_id=$3 ORDER BY ts ASC"""
        q2 = f"""SELECT rule_key, value_json, confidence FROM {self.schema}.conv_pref_exceptions
                 WHERE user_id=$1 AND conversation_id=$2 AND turn_id=$3 ORDER BY ts ASC"""
        async with self._pool.acquire() as con:
            prefs = await con.fetch(q1, user_id, conversation_id, turn_id)
            excs  = await con.fetch(q2, user_id, conversation_id, turn_id)
        return {
            "assertions": [dict(p) for p in prefs],
            "exceptions": [dict(e) for e in excs]
        }

    async def fetch_latest_summary_text(self, *, user_id: str, conversation_id: str) -> str:
        q = f"""
        SELECT text FROM {self.schema}.conv_messages
        WHERE user_id=$1 AND conversation_id=$2
          AND role='artifact' AND tags @> ARRAY['kind:conversation.summary']::text[]
        ORDER BY ts DESC LIMIT 1
        """
        async with self._pool.acquire() as con:
            row = await con.fetchrow(q, user_id, conversation_id)
        return (row["text"] if row else "") or ""

    async def fetch_last_turn_summaries(self, *, user_id: str, conversation_id: str, limit: int = 3) -> list[str]:
        q = f"""
        SELECT text FROM {self.schema}.conv_messages
        WHERE user_id=$1 AND conversation_id=$2
          AND role='artifact' AND tags @> ARRAY['kind:turn.summary']::text[]
        ORDER BY ts DESC LIMIT {int(limit)}
        """
        async with self._pool.acquire() as con:
            rows = await con.fetch(q, user_id, conversation_id)
        return [r["text"] for r in rows]

