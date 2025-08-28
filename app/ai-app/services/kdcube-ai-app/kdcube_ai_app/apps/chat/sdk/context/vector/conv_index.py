# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/vector/conv_index.py
import asyncpg
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Sequence, Union

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

def _embed_to_pg(vec: Sequence[float]) -> list[float]:
    return [float(x) for x in vec]

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
        # packaged as conversation_history.sql
        return pkgutil.get_data(__package__, "conversation_history.sql")

    # NOTE: tenant/project are outside table; they are encoded in schema choice
    async def add_message(
            self, *, user_id:str, conversation_id:str,
            role:str, text:str, s3_uri:str, ts:Union[str, datetime],
            tags:Optional[list[str]]=None, ttl_days:int=365,
            embedding:Optional[List[float]]=None, message_id: Optional[str]=None
    ) -> int:
        ts_dt = _coerce_ts(ts)
        # if embedding:
        #     embedding = convert_embedding_to_string(embedding)
        async with self._pool.acquire() as con:
            q = f"""INSERT INTO {self.schema}.conv_messages
                    (user_id,conversation_id,message_id,role,text,s3_uri,ts,ttl_days,tags,embedding)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    RETURNING id"""
            rec = await con.fetchrow(
                q, user_id, conversation_id, message_id, role, text, s3_uri, ts_dt, ttl_days, tags or [],
                convert_embedding_to_string(embedding) if embedding else None
            )
            return int(rec["id"])

    async def search_recent(
            self, *, user_id:str, conversation_id:str,
            query_embedding:List[float], top_k:int=8, days:int=90,
            roles:tuple[str,...]=("user","assistant","artifact")
    ) -> List[Dict[str,Any]]:
        async with self._pool.acquire() as con:
            q = f"""SELECT id, message_id, role, text, s3_uri, ts,
                           1 - (embedding <=> $5) AS score
                    FROM {self.schema}.conv_messages
                    WHERE user_id=$1 AND conversation_id=$2
                      AND role = ANY($3)
                      AND ts >= now() - ($4::text || ' days')::interval
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> $5
                    LIMIT {int(top_k)}"""
            rows = await con.fetch(
                q, user_id, conversation_id, list(roles), str(days),
                convert_embedding_to_string(query_embedding)
            )
            return [dict(r) for r in rows]
