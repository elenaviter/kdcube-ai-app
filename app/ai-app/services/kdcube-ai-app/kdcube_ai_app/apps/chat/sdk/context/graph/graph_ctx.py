# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chatbot/graph/graph_ctx.py

from neo4j import AsyncGraphDatabase
from typing import Optional, Dict, Any, List, Tuple
import time, math, uuid

from kdcube_ai_app.apps.chat.sdk.config import get_settings

Scope = str  # "global"|"user"|"session"|"topic"

def _now_sec() -> int:
    return int(time.time())

def _decay_score(base: float, created_at: int, half_life_days: float = 30.0) -> float:
    # Exponential decay by age
    age_days = max(0.0, (_now_sec() - created_at) / 86400.0)
    return float(base) * (0.5 ** (age_days / half_life_days))

class GraphCtx:
    def __init__(self):
        self._settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(self._settings.NEO4J_URI, auth=(self._settings.NEO4J_USER, self._settings.NEO4J_PASSWORD))

    async def close(self):
        await self._driver.close()

    async def ensure_schema(self):
        async with self._driver.session() as s:
            for stmt in (await self._read_cypher()).decode().split(";"):
                if stmt.strip():
                    await s.run(stmt)

    async def _read_cypher(self) -> bytes:
        import pkgutil
        return pkgutil.get_data(__package__, "graph_ctx.cypher")

    async def add_assertion(self, *, tenant:str, project:str, user:str, conversation:str,
                            key:str, value:Any, desired:bool, scope:Scope,
                            confidence:float=0.9, ttl_days:int=365, reason:str="agent"):
        now = _now_sec()
        aid = str(uuid.uuid4())
        async with self._driver.session() as s:
            await s.run(
                """MERGE (u:User {key:$uk})
                   MERGE (conv:Conversation {key:$ck})
                   CREATE (a:Assertion {
                     id:$aid, tenant:$tenant, project:$project, user:$user, conversation:$conversation,
                     key:$key, value:$value, desired:$desired, scope:$scope,
                     confidence:$confidence, created_at:$now, ttl_days:$ttl, reason:$reason
                   })
                   MERGE (u)-[:HAS_ASSERTION]->(a)
                   MERGE (conv)-[:INCLUDES]->(a)""",
                uk=f"{tenant}:{project}:{user}",
                ck=f"{tenant}:{project}:{conversation}",
                aid=aid, tenant=tenant, project=project, user=user, conversation=conversation,
                key=key, value=value, desired=bool(desired), scope=scope,
                confidence=float(confidence), now=now, ttl=int(ttl_days), reason=reason
            )
        return aid

    async def add_exception(self, *, tenant:str, project:str, user:str, conversation:str,
                            rule_key:str, scope:Scope, value:Any, reason:str="agent"):
        now = _now_sec()
        eid = str(uuid.uuid4())
        async with self._driver.session() as s:
            await s.run(
                """MERGE (u:User {key:$uk})
                   MERGE (conv:Conversation {key:$ck})
                   CREATE (e:Exception {
                     id:$eid, tenant:$tenant, project:$project, user:$user, conversation:$conversation,
                     rule_key:$rk, value:$value, scope:$scope, created_at:$now, reason:$reason
                   })
                   MERGE (u)-[:HAS_EXCEPTION]->(e)
                   MERGE (conv)-[:INCLUDES]->(e)""",
                uk=f"{tenant}:{project}:{user}",
                ck=f"{tenant}:{project}:{conversation}",
                eid=eid, tenant=tenant, project=project, user=user, conversation=conversation,
                rk=rule_key, value=value, scope=scope, now=now, reason=reason
            )
        return eid

    async def snapshot(self, *, tenant:str, project:str, user:str, conversation:str) -> Dict[str,Any]:
        async with self._driver.session() as s:
            res = await s.run(
                """MATCH (u:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                   WHERE a.tenant=$tenant AND a.project=$project
                   OPTIONAL MATCH (u)-[:HAS_EXCEPTION]->(e:Exception)
                   WHERE e.tenant=$tenant AND e.project=$project
                   RETURN collect(a) as assertions, collect(e) as exceptions""",
                uk=f"{tenant}:{project}:{user}", tenant=tenant, project=project
            )
            rec = await res.single()
            assertions = [dict(r) for r in (rec["assertions"] or [])]
            exceptions = [dict(r) for r in (rec["exceptions"] or [])]

            # conversation-local:
            res2 = await s.run(
                """MATCH (c:Conversation {key:$ck})-[:INCLUDES]->(a:Assertion)
                   RETURN collect(a) as c_assertions""",
                ck=f"{tenant}:{project}:{conversation}"
            )
            s_assertions = [dict(r) for r in (await res2.single())["c_assertions"] or []]

            res3 = await s.run(
                """MATCH (c:Conversation {key:$ck})-[:INCLUDES]->(e:Exception)
                   RETURN collect(e) as c_exceptions""",
                ck=f"{tenant}:{project}:{conversation}"
            )
            s_exceptions = [dict(r) for r in (await res3.single())["c_exceptions"] or []]

        return {"assertions": assertions + s_assertions, "exceptions": exceptions + s_exceptions}


    async def cleanup_expired(self) -> dict:
        """Remove assertions whose ttl expired."""
        now = _now_sec()
        async with self._driver.session() as s:
            res = await s.run(
                """MATCH (a:Assertion)
                   WHERE (a.created_at + (a.ttl_days * 86400)) < $now
                   DETACH DELETE a
                   RETURN count(*) as deleted"""
                , now=now
            )
            rec = await res.single()
            return {"deleted": rec["deleted"]}
