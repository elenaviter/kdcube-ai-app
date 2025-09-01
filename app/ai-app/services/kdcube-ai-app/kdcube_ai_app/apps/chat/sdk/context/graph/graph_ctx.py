# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chatbot/graph/graph_ctx.py
from __future__ import annotations

import json, time, uuid, hashlib
from typing import Optional, Dict, Any, List, Tuple

from neo4j import AsyncGraphDatabase

from kdcube_ai_app.apps.chat.sdk.config import get_settings

Scope = str  # "global" | "user" | "conversation"

# ---------- helpers ----------

def _now_sec() -> int:
    return int(time.time())

def _is_primitive(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def _stable_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _value_hash(v_prim: Any, v_json: Optional[str]) -> str:
    payload = _stable_json(v_prim) if v_json is None else v_json
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _pack_for_neo4j_value(v: Any) -> Tuple[Any, Optional[str], str, str]:
    """
    Returns (value_primitive, value_json, value_type, value_hash)
    - primitives & list-of-primitives go into value (Neo4j-safe)
    - objects/mixed arrays go into value_json
    """
    if _is_primitive(v):
        v_prim, v_json, v_type = v, None, "primitive"
        return v_prim, v_json, v_type, _value_hash(v_prim, v_json)

    if isinstance(v, (list, tuple)) and all(_is_primitive(x) for x in v):
        v_prim, v_json, v_type = list(v), None, "array"
        return v_prim, v_json, v_type, _value_hash(v_prim, v_json)

    v_json, v_prim, v_type = _stable_json(v), None, "object"
    return v_prim, v_json, v_type, _value_hash(v_prim, v_json)

# ---------- GraphCtx ----------

class GraphCtx:
    """
    Minimal surface:
      - ensure_schema()
      - ensure_user_and_conversation()
      - set_conversation_meta(), get_conversation_meta()
      - add_assertion(), add_exception(), upsert_assertion()
      - snapshot()
      - load_user_assertions(), mark_user_key_challenged()
      - forget_user_key(), forget_user_all(), purge_anonymous()
    Nodes use a single 'key' property for MERGEs:
      u.key = f"{tenant}:{project}:{user}"
      c.key = f"{tenant}:{project}:{conversation}"
    """

    def __init__(self):
        self._settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            self._settings.NEO4J_URI,
            auth=(self._settings.NEO4J_USER, self._settings.NEO4J_PASSWORD),
            max_connection_lifetime=300,
        )

    async def close(self):
        await self._driver.close()

    # ---------- schema ----------

    async def ensure_schema(self):
        cypher = (await self._read_cypher()).decode()
        stmts = [s.strip() for s in cypher.split(";") if s.strip()]
        async with self._driver.session() as s:
            for stmt in stmts:
                await s.run(stmt)

    async def _read_cypher(self) -> bytes:
        import pkgutil
        # package data file name should match: graph_ctx.cypher (above)
        return pkgutil.get_data(__package__, "graph_ctx.cypher")

    # ---------- conversation ownership ----------

    async def ensure_user_and_conversation(
        self, *, tenant: str, project: str, user: str, conversation: str, user_type: Optional[str] = None
    ):
        """
        Idempotently ensure:
          (:User {key})-[:HAS_CONVERSATION]->(:Conversation {key})
        and bump conversation.last_seen_at.
        """
        uk = f"{tenant}:{project}:{user}"
        ck = f"{tenant}:{project}:{conversation}"
        now = _now_sec()
        async with self._driver.session() as s:
            await s.run(
                """
                MERGE (u:User {key:$uk})
                  ON CREATE SET u.created_at=$now, u.user_type=coalesce($user_type,'anonymous')
                  ON MATCH  SET u.user_type = coalesce(u.user_type, $user_type)
                MERGE (c:Conversation {key:$ck})
                  ON CREATE SET c.started_at=$now, c.user_id=$user, c.user_type=coalesce($user_type,'anonymous')
                SET c.last_seen_at=$now
                MERGE (u)-[r:HAS_CONVERSATION]->(c)
                  ON CREATE SET r.created_at=$now
                """,
                uk=uk, ck=ck, user=user, user_type=user_type, now=now
            )

    # ---------- conversation meta ----------

    async def set_conversation_meta(
        self, *, tenant: str, project: str, conversation: str, fields: Dict[str, Any]
    ) -> None:
        """
        Stores meta props on Conversation. You can pass primitives, arrays, and JSON strings (e.g., topics_json).
        Also stamps meta_updated_at.
        """
        ck = f"{tenant}:{project}:{conversation}"
        async with self._driver.session() as s:
            await s.run(
                """
                MERGE (c:Conversation {key:$ck})
                SET c += $fields,
                    c.meta_updated_at = $now
                """,
                ck=ck, fields=fields or {}, now=_now_sec()
            )

    async def get_conversation_meta(self, *, tenant: str, project: str, conversation: str) -> Dict[str, Any]:
        ck = f"{tenant}:{project}:{conversation}"
        async with self._driver.session() as s:
            rec = await (await s.run(
                "MATCH (c:Conversation {key:$ck}) RETURN c LIMIT 1", ck=ck
            )).single()
        if not rec:
            return {}
        node = dict(rec["c"])
        return {k: node[k] for k in node.keys()}

    # ---------- assertions & exceptions ----------

    async def add_assertion(
        self, *,
        tenant: str, project: str, user: str, conversation: str,
        key: str, value: Any, desired: bool, scope: Scope,
        confidence: float = 0.9, ttl_days: int = 365, reason: str = "agent",
        turn_id: Optional[str] = None, user_type: str = "anonymous"
    ) -> str:
        """
        Create a *new* assertion for this turn and link it to both User and Conversation.
        """
        uk = f"{tenant}:{project}:{user}"
        ck = f"{tenant}:{project}:{conversation}"
        now = _now_sec()
        aid = str(uuid.uuid4())
        v_prim, v_json, v_type, v_hash = _pack_for_neo4j_value(value)

        async with self._driver.session() as s:
            await s.run(
                """
                MERGE (u:User {key:$uk})
                  ON CREATE SET u.created_at=$now, u.user_type=$user_type
                  ON MATCH  SET u.user_type = coalesce(u.user_type, $user_type)
                MERGE (c:Conversation {key:$ck})
                CREATE (a:Assertion {
                  id:$aid, tenant:$tenant, project:$project, user:$user, conversation:$conversation,
                  key:$key, value:$value_primitive, value_json:$value_json, value_type:$value_type, value_hash:$value_hash,
                  desired:$desired, scope:$scope,
                  confidence:$confidence, created_at:$now, ttl_days:$ttl_days, reason:$reason, turn_id:$turn_id
                })
                MERGE (u)-[:HAS_ASSERTION]->(a)
                MERGE (c)-[:INCLUDES]->(a)
                """,
                uk=uk, ck=ck, aid=aid, tenant=tenant, project=project, user=user, conversation=conversation,
                key=key,
                value_primitive=v_prim, value_json=v_json, value_type=v_type, value_hash=v_hash,
                desired=bool(desired), scope=scope,
                confidence=float(confidence), now=now, ttl_days=int(ttl_days), reason=reason,
                turn_id=turn_id, user_type=user_type
            )
        return aid

    async def add_exception(
        self, *,
        tenant: str, project: str, user: str, conversation: str,
        rule_key: str, scope: Scope, value: Any, reason: str = "agent",
        turn_id: Optional[str] = None, user_type: str = "anonymous"
    ) -> str:
        uk = f"{tenant}:{project}:{user}"
        ck = f"{tenant}:{project}:{conversation}"
        now = _now_sec()
        eid = str(uuid.uuid4())
        v_prim, v_json, v_type, v_hash = _pack_for_neo4j_value(value)

        async with self._driver.session() as s:
            await s.run(
                """
                MERGE (u:User {key:$uk})
                  ON CREATE SET u.created_at=$now, u.user_type=$user_type
                  ON MATCH  SET u.user_type = coalesce(u.user_type, $user_type)
                MERGE (c:Conversation {key:$ck})
                CREATE (e:Exception {
                  id:$eid, tenant:$tenant, project:$project, user:$user, conversation:$conversation,
                  rule_key:$rule_key, value:$value_primitive, value_json:$value_json, value_type:$value_type, value_hash:$value_hash,
                  scope:$scope, created_at:$now, reason:$reason, turn_id:$turn_id
                })
                MERGE (u)-[:HAS_EXCEPTION]->(e)
                MERGE (c)-[:INCLUDES]->(e)
                """,
                uk=uk, ck=ck, eid=eid, tenant=tenant, project=project, user=user, conversation=conversation,
                rule_key=rule_key,
                value_primitive=v_prim, value_json=v_json, value_type=v_type, value_hash=v_hash,
                scope=scope, now=now, reason=reason, turn_id=turn_id, user_type=user_type
            )
        return eid

    async def upsert_assertion(
        self, *,
        tenant: str, project: str, user: str, conversation: str | None,
        key: str, value: Any, desired: bool, scope: Scope,
        confidence: float, ttl_days: int, reason: str, turn_id: str | None = None, bump_time: bool = True
    ) -> str:
        """
        Idempotent promotion/update (usually scope='user'). Keeps a single record per semantic identity.
        """
        uk = f"{tenant}:{project}:{user}"
        ck = f"{tenant}:{project}:{conversation}" if conversation else None
        now = _now_sec()
        v_prim, v_json, v_type, v_hash = _pack_for_neo4j_value(value)
        aid = str(uuid.uuid4())
        async with self._driver.session() as s:
            rec = await (await s.run(
                """
                MERGE (u:User {key:$uk})
                FOREACH(_ IN CASE WHEN $ck IS NULL THEN [] ELSE [1] END |
                    MERGE (conv:Conversation {key:$ck})
                )
                MERGE (a:Assertion {
                  tenant:$tenant, project:$project, user:$user,
                  key:$key, value_hash:$value_hash, scope:$scope, desired:$desired
                })
                ON CREATE SET
                  a.id=$aid, a.value=$value_primitive, a.value_json=$value_json, a.value_type=$value_type,
                  a.confidence=$confidence, a.created_at=$now, a.ttl_days=$ttl_days, a.reason=$reason,
                  a.turn_id=$turn_id, a.conversation=coalesce($ck,'(promoted)')
                ON MATCH SET
                  a.value=$value_primitive, a.value_json=$value_json, a.value_type=$value_type,
                  a.confidence = 0.6*a.confidence + 0.4*$confidence,
                  a.ttl_days = GREATEST(a.ttl_days, $ttl_days),
                  a.reason = $reason,
                  a.turn_id = coalesce($turn_id, a.turn_id),
                  a.conversation = coalesce(a.conversation, coalesce($ck,'(promoted)')),
                  a.created_at = CASE WHEN $bump_time THEN $now ELSE a.created_at END
                MERGE (u)-[:HAS_ASSERTION]->(a)
                FOREACH(_ IN CASE WHEN $ck IS NULL THEN [] ELSE [1] END |
                    MERGE (conv)-[:INCLUDES]->(a)
                )
                RETURN a.id AS id
                """,
                uk=uk, ck=ck, tenant=tenant, project=project, user=user,
                key=key, value_primitive=v_prim, value_json=v_json, value_type=v_type, value_hash=v_hash,
                scope=scope, desired=bool(desired), confidence=float(confidence), ttl_days=int(ttl_days),
                reason=reason, turn_id=turn_id, now=now, bump_time=bool(bump_time), aid=aid
            )).single()
            return rec["id"] or aid

    # ---------- reads & hygiene ----------

    async def snapshot(self, *, tenant: str, project: str, user: str, conversation: str) -> Dict[str, Any]:
        """
        Returns {assertions, exceptions} = (user/global, TTL-valid) ∪ (this conversation, TTL-valid)
        """
        uk = f"{tenant}:{project}:{user}"
        ck = f"{tenant}:{project}:{conversation}"
        now = _now_sec()

        async with self._driver.session() as s:
            # user/global
            rec = await (await s.run(
                """
                MATCH (u:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                WHERE a.tenant=$tenant AND a.project=$project
                  AND a.scope IN ['user','global']
                  AND (a.created_at + (a.ttl_days * 86400)) >= $now
                OPTIONAL MATCH (u)-[:HAS_EXCEPTION]->(e:Exception)
                WHERE e.tenant=$tenant AND e.project=$project
                  AND e.scope IN ['user','global']
                  AND (e.created_at + (coalesce(e.ttl_days,365) * 86400)) >= $now
                RETURN collect(a) AS asrts, collect(e) AS excs
                """,
                uk=uk, tenant=tenant, project=project, now=now
            )).single()
            u_assertions = [dict(x) for x in (rec["asrts"] or [])]
            u_exceptions = [dict(x) for x in (rec["excs"] or [])]

            # conversation
            rec2 = await (await s.run(
                """
                MATCH (c:Conversation {key:$ck})-[:INCLUDES]->(a:Assertion)
                WHERE (a.created_at + (a.ttl_days * 86400)) >= $now
                OPTIONAL MATCH (c)-[:INCLUDES]->(e:Exception)
                WHERE (e.created_at + (coalesce(e.ttl_days,365) * 86400)) >= $now
                RETURN collect(a) AS casrts, collect(e) AS cexcs
                """,
                ck=ck, now=now
            )).single()
            c_assertions = [dict(x) for x in (rec2["casrts"] or [])]
            c_exceptions = [dict(x) for x in (rec2["cexcs"] or [])]

        def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen, out = set(), []
            for n in items:
                nid = n.get("id")
                if nid in seen:
                    continue
                seen.add(nid)
                out.append(n)
            return out

        return {
            "assertions": _dedup(u_assertions + c_assertions),
            "exceptions": _dedup(u_exceptions + c_exceptions),
        }

    async def load_user_assertions(self, *, tenant: str, project: str, user: str) -> List[dict]:
        uk = f"{tenant}:{project}:{user}"
        async with self._driver.session() as s:
            rows = await (await s.run(
                """
                MATCH (u:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                WHERE a.tenant=$tenant AND a.project=$project
                RETURN a ORDER BY a.created_at DESC
                """,
                uk=uk, tenant=tenant, project=project
            )).values()
        return [dict(r[0]) for r in (rows or [])]

    async def mark_user_key_challenged(self, *, tenant: str, project: str, user: str, key: str) -> int:
        uk = f"{tenant}:{project}:{user}"
        async with self._driver.session() as s:
            rec = await (await s.run(
                """
                MATCH (:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                WHERE a.tenant=$tenant AND a.project=$project AND a.scope='user' AND a.key=$key
                SET a.challenged_at=$now
                RETURN count(a) AS n
                """,
                uk=uk, tenant=tenant, project=project, key=key, now=_now_sec()
            )).single()
            return int(rec["n"] or 0)

    async def forget_user_key(self, *, tenant: str, project: str, user: str, key: str) -> int:
        uk = f"{tenant}:{project}:{user}"
        async with self._driver.session() as s:
            rec = await (await s.run(
                """
                MATCH (:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                WHERE a.tenant=$tenant AND a.project=$project AND a.scope='user' AND a.key=$key
                DETACH DELETE a
                RETURN count(*) AS deleted
                """,
                uk=uk, tenant=tenant, project=project
            )).single()
            return int(rec["deleted"] or 0)

    async def forget_user_all(self, *, tenant: str, project: str, user: str) -> int:
        uk = f"{tenant}:{project}:{user}"
        async with self._driver.session() as s:
            rec = await (await s.run(
                """
                MATCH (:User {key:$uk})-[:HAS_ASSERTION]->(a:Assertion)
                WHERE a.tenant=$tenant AND a.project=$project AND a.scope='user'
                DETACH DELETE a
                RETURN count(*) AS deleted
                """,
                uk=uk, tenant=tenant, project=project
            )).single()
            return int(rec["deleted"] or 0)

    async def purge_anonymous(self, *, tenant: str, project: str, older_than_days: int = 1) -> Dict[str, int]:
        cutoff = _now_sec() - older_than_days * 86400
        async with self._driver.session() as s:
            r1 = await (await s.run(
                """
                MATCH (u:User {user_type:'anonymous'})-[:HAS_ASSERTION]->(a:Assertion)
                WHERE a.tenant=$tenant AND a.project=$project AND a.created_at <= $cutoff
                DETACH DELETE a
                RETURN count(*) AS n
                """,
                tenant=tenant, project=project, cutoff=cutoff
            )).single()

            r2 = await (await s.run(
                """
                MATCH (u:User {user_type:'anonymous'})-[:HAS_EXCEPTION]->(e:Exception)
                WHERE e.tenant=$tenant AND e.project=$project AND e.created_at <= $cutoff
                DETACH DELETE e
                RETURN count(*) AS n
                """,
                tenant=tenant, project=project, cutoff=cutoff
            )).single()

            # prune orphan nodes (cheap hygiene)
            await s.run("MATCH (u:User {user_type:'anonymous'}) WHERE NOT (u)--() DELETE u")
            await s.run("MATCH (c:Conversation) WHERE NOT (c)-[:INCLUDES]->() AND NOT ()-[:HAS_CONVERSATION]->(c) DELETE c")

        return {"assertions_deleted": int(r1["n"] or 0), "exceptions_deleted": int(r2["n"] or 0)}
