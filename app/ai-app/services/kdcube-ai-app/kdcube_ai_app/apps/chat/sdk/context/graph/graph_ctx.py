# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chatbot/graph/graph_ctx.py

from neo4j import AsyncGraphDatabase
from typing import Optional, Dict, Any, List, Tuple
import time, math, uuid, json, hashlib

from kdcube_ai_app.apps.chat.sdk.config import get_settings

Scope = str  # "global"|"user"|"conversation"|"topic"

def _now_sec() -> int:
    return int(time.time())

def _decay_score(base: float, created_at: int, half_life_days: float = 30.0) -> float:
    age_days = max(0.0, (_now_sec() - created_at) / 86400.0)
    return float(base) * (0.5 ** (age_days / half_life_days))

def _is_primitive(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def _stable_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _value_hash(v_prim: Any, v_json: Optional[str]) -> str:
    payload = _stable_json(v_prim) if v_json is None else v_json
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _pack_for_neo4j_value(v: Any) -> Tuple[Any, Optional[str], str, str]:
    """
    Returns tuple:
      (value_primitive, value_json, value_type, value_hash)
    - value_primitive: None | str | int | float | bool | list[primitive]
    - value_json: JSON string when non-primitive (dict / mixed arrays), else None
    - value_type: 'primitive' | 'array' | 'object'
    - value_hash: sha1 for quick dedupe/search
    """
    if _is_primitive(v):
        v_prim = v
        v_json = None
        v_type = "primitive"
        return v_prim, v_json, v_type, _value_hash(v_prim, v_json)

    if isinstance(v, (list, tuple)) and all(_is_primitive(x) for x in v):
        v_prim = list(v)
        v_json = None
        v_type = "array"
        return v_prim, v_json, v_type, _value_hash(v_prim, v_json)

    # fallback: store full object as JSON
    v_json = _stable_json(v)
    v_prim = None
    v_type = "object"
    return v_prim, v_json, v_type, _value_hash(v_prim, v_json)

# --------------------------------------------------

class GraphCtx:
    def __init__(self):
        self._settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            self._settings.NEO4J_URI,
            auth=(self._settings.NEO4J_USER, self._settings.NEO4J_PASSWORD)
        )

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

    async def add_assertion(
        self, *,
        tenant: str, project: str, user: str, conversation: str,
        key: str, value: Any, desired: bool, scope: Scope,
        confidence: float = 0.9, ttl_days: int = 365, reason: str = "agent",
        turn_id: Optional[str] = None
    ) -> str:
        """
        Stores 'value' safely:
          - 'value'      : primitive or array (Neo4j-safe)
          - 'value_json' : JSON string for objects / mixed arrays
          - 'value_type' : 'primitive'|'array'|'object'
          - 'value_hash' : sha1 for quick dedupe/search
        """
        now = _now_sec()
        aid = str(uuid.uuid4())
        v_prim, v_json, v_type, v_hash = _pack_for_neo4j_value(value)

        async with self._driver.session() as s:
            await s.run(
                """MERGE (u:User {key:$uk})
                   MERGE (conv:Conversation {key:$ck})
                   CREATE (a:Assertion {
                     id:$aid, tenant:$tenant, project:$project, user:$user, conversation:$conversation,
                     key:$key, value:$value_primitive, value_json:$value_json, value_type:$value_type, value_hash:$value_hash,
                     desired:$desired, scope:$scope,
                     confidence:$confidence, created_at:$now, ttl_days:$ttl, reason:$reason,
                     turn_id:$turn_id
                   })
                   MERGE (u)-[:HAS_ASSERTION]->(a)
                   MERGE (conv)-[:INCLUDES]->(a)""",
                uk=f"{tenant}:{project}:{user}",
                ck=f"{tenant}:{project}:{conversation}",
                aid=aid, tenant=tenant, project=project, user=user, conversation=conversation,
                key=key,
                value_primitive=v_prim, value_json=v_json, value_type=v_type, value_hash=v_hash,
                desired=bool(desired), scope=scope,
                confidence=float(confidence), now=now, ttl=int(ttl_days), reason=reason,
                turn_id=turn_id
            )
        return aid

    async def add_exception(
        self, *,
        tenant: str, project: str, user: str, conversation: str,
        rule_key: str, scope: Scope, value: Any, reason: str = "agent",
        turn_id: Optional[str] = None
    ) -> str:
        """
        Same safe value handling for exceptions.
        """
        now = _now_sec()
        eid = str(uuid.uuid4())
        v_prim, v_json, v_type, v_hash = _pack_for_neo4j_value(value)

        async with self._driver.session() as s:
            await s.run(
                """MERGE (u:User {key:$uk})
                   MERGE (conv:Conversation {key:$ck})
                   CREATE (e:Exception {
                     id:$eid, tenant:$tenant, project:$project, user:$user, conversation:$conversation,
                     rule_key:$rk, value:$value_primitive, value_json:$value_json, value_type:$value_type, value_hash:$value_hash,
                     scope:$scope, created_at:$now, reason:$reason,
                     turn_id:$turn_id
                   })
                   MERGE (u)-[:HAS_EXCEPTION]->(e)
                   MERGE (conv)-[:INCLUDES]->(e)""",
                uk=f"{tenant}:{project}:{user}",
                ck=f"{tenant}:{project}:{conversation}",
                eid=eid, tenant=tenant, project=project, user=user, conversation=conversation,
                rk=rule_key,
                value_primitive=v_prim, value_json=v_json, value_type=v_type, value_hash=v_hash,
                scope=scope, now=now, reason=reason, turn_id=turn_id
            )
        return eid

    async def snapshot(self, *, tenant: str, project: str, user: str, conversation: str) -> Dict[str, Any]:
        now = _now_sec()
        async with self._driver.session() as s:
            # User-level (and global) items only, filtered by TTL
            res = await s.run(
                """MATCH (u:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                   WHERE a.tenant=$tenant AND a.project=$project
                     AND a.scope IN ['user','global']
                     AND (a.created_at + (a.ttl_days * 86400)) >= $now
                   OPTIONAL MATCH (u)-[:HAS_EXCEPTION]->(e:Exception)
                   WHERE e.tenant=$tenant AND e.project=$project
                     AND e.scope IN ['user','global']
                     AND (e.created_at + (365 * 86400)) >= $now
                   RETURN collect(a) as assertions, collect(e) as exceptions""",
                uk=f"{tenant}:{project}:{user}",
                tenant=tenant,
                project=project,
                now=now,
            )
            rec = await res.single()
            assertions = [dict(r) for r in (rec["assertions"] or [])]
            exceptions = [dict(r) for r in (rec["exceptions"] or [])]

            # Conversation-local items (this conversation only), filtered by TTL
            res2 = await s.run(
                """MATCH (c:Conversation {key:$ck})-[:INCLUDES]->(a:Assertion)
                   WHERE (a.created_at + (a.ttl_days * 86400)) >= $now
                   RETURN collect(a) as c_assertions""",
                ck=f"{tenant}:{project}:{conversation}",
                now=now,
            )
            s_assertions = [dict(r) for r in (await res2.single())["c_assertions"] or []]

            res3 = await s.run(
                """MATCH (c:Conversation {key:$ck})-[:INCLUDES]->(e:Exception)
                   WHERE (e.created_at + (365 * 86400)) >= $now
                   RETURN collect(e) as c_exceptions""",
                ck=f"{tenant}:{project}:{conversation}",
                now=now,
            )
            s_exceptions = [dict(r) for r in (await res3.single())["c_exceptions"] or []]

        def _dedup(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen = set()
            out = []
            for n in nodes:
                nid = n.get("id")
                if nid in seen:
                    continue
                seen.add(nid)
                out.append(n)
            return out

        return {
            "assertions": _dedup(assertions + s_assertions),
            "exceptions": _dedup(exceptions + s_exceptions),
        }

    async def cleanup_expired(self) -> dict:
        now = _now_sec()
        async with self._driver.session() as s:
            rec = await (await s.run(
                """MATCH (a:Assertion)
                   WHERE (a.created_at + (a.ttl_days * 86400)) < $now
                   DETACH DELETE a
                   RETURN count(*) as deleted""",
                now=now
            )).single()
            return {"deleted": rec["deleted"]}

    # --------- NEW: helpers for promotion & hygiene ---------

    async def load_conversation_assertions(self, *, tenant: str, project: str, user: str, lookback_days: int = 180) -> list[dict]:
        since = _now_sec() - int(lookback_days * 86400)
        async with self._driver.session() as s:
            res = await s.run(
                """MATCH (u:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                   WHERE a.tenant=$tenant AND a.project=$project
                     AND a.scope='conversation'
                     AND a.created_at >= $since
                   RETURN collect(a) AS items""",
                uk=f"{tenant}:{project}:{user}", tenant=tenant, project=project, since=since
            )
            rec = await res.single()
            return [dict(x) for x in (rec["items"] or [])]

    async def upsert_assertion(
        self, *, tenant: str, project: str, user: str, conversation: str | None,
        key: str, value: Any, desired: bool, scope: Scope,
        confidence: float, ttl_days: int, reason: str, turn_id: str | None = None, bump_time: bool = True
    ) -> str:
        now = _now_sec()
        v_prim, v_json, v_type, v_hash = _pack_for_neo4j_value(value)
        aid = str(uuid.uuid4())
        async with self._driver.session() as s:
            params = dict(
                uk=f"{tenant}:{project}:{user}",
                ck=f"{tenant}:{project}:{conversation}" if conversation else None,
                tenant=tenant, project=project, user=user, key=key,
                value_primitive=v_prim, value_json=v_json, value_type=v_type, value_hash=v_hash,
                desired=bool(desired), scope=scope, confidence=float(confidence),
                ttl=int(ttl_days), reason=reason, now=now, aid=aid, turn_id=turn_id, bump_time=bool(bump_time)
            )
            cypher = """
                MERGE (u:User {key:$uk})
                FOREACH (_ IN CASE WHEN $ck IS NULL THEN [] ELSE [1] END |
                    MERGE (conv:Conversation {key:$ck})
                )
                MERGE (a:Assertion {
                    tenant:$tenant, project:$project, user:$user,
                    key:$key, value_hash:$value_hash, scope:$scope, desired:$desired
                })
                ON CREATE SET
                    a.id=$aid, a.value=$value_primitive, a.value_json=$value_json, a.value_type=$value_type,
                    a.confidence=$confidence, a.created_at=$now, a.ttl_days=$ttl, a.reason=$reason, a.turn_id=$turn_id,
                    a.conversation=COALESCE($ck,'(promoted)')
                ON MATCH SET
                    a.value=$value_primitive, a.value_json=$value_json, a.value_type=$value_type,
                    a.confidence = 0.6*a.confidence + 0.4*$confidence,
                    a.ttl_days = GREATEST(a.ttl_days, $ttl),
                    a.reason = $reason,
                    a.conversation = COALESCE(a.conversation, COALESCE($ck,'(promoted)')),
                    a.turn_id = COALESCE($turn_id, a.turn_id),
                    a.created_at = CASE WHEN $bump_time THEN $now ELSE a.created_at END
                MERGE (u)-[:HAS_ASSERTION]->(a)
                FOREACH (_ IN CASE WHEN $ck IS NULL THEN [] ELSE [1] END |
                    MERGE (conv)-[:INCLUDES]->(a)
                )
                RETURN a.id AS id
            """
            rec = await (await s.run(cypher, **params)).single()
            return rec["id"] or aid

    async def forget_user_key(self, *, tenant: str, project: str, user: str, key: str) -> int:
        async with self._driver.session() as s:
            rec = await (await s.run(
                """MATCH (:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                   WHERE a.tenant=$tenant AND a.project=$project AND a.scope='user' AND a.key=$key
                   DETACH DELETE a
                   RETURN count(*) as deleted""",
                uk=f"{tenant}:{project}:{user}", tenant=tenant, project=project, key=key
            )).single()
            return int(rec["deleted"] or 0)

    async def forget_user_all(self, *, tenant: str, project: str, user: str) -> int:
        async with self._driver.session() as s:
            rec = await (await s.run(
                """MATCH (:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                   WHERE a.tenant=$tenant AND a.project=$project AND a.scope='user'
                   DETACH DELETE a
                   RETURN count(*) as deleted""",
                uk=f"{tenant}:{project}:{user}", tenant=tenant, project=project
            )).single()
            return int(rec["deleted"] or 0)
