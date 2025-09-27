# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/retrieval/kb_client.py
import traceback, os
import asyncpg, json, re, unicodedata, logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone, timedelta
import kdcube_ai_app.apps.utils.sql_dt_utils as dt_utils

from kdcube_ai_app.apps.knowledge_base.db.data_models import HybridSearchParams
from kdcube_ai_app.infra.embedding.embedding import convert_embedding_to_string
from kdcube_ai_app.apps.chat.sdk.config import get_settings

logger = logging.getLogger(__name__)

def _day_bounds_utc(day_str: str) -> tuple[str, str]:
    """
    For YYYY-MM-DD, return [start, next_day) bounds as ISO UTC text.
    """
    d = datetime.fromisoformat(day_str).date()
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )

def _patch_placeholders(clauses: list[str], start_index: int) -> tuple[list[str], int]:
    """
    Replace each '$%s' with sequential $N starting at start_index+1.
    Returns (patched_clauses, last_index).
    """
    idx = start_index
    patched: list[str] = []
    for clause in clauses:
        # replace one placeholder at a time so each gets a unique index
        while "$%s" in clause:
            idx += 1
            clause = clause.replace("$%s", f"${idx}", 1)
        patched.append(clause)
    return patched, idx


class KBClient:
    """
    Queries your KB schema:
      - <SCHEMA>.retrieval_segment with (search_vector TSVECTOR, embedding VECTOR(1536))
      - <SCHEMA>.datasource for expiration
    """
    def __init__(self,
                 pool: Optional[asyncpg.Pool] = None):

        self._pool: Optional[asyncpg.Pool] = pool
        self._settings = get_settings()

        tenant = self._settings.TENANT.replace("-", "_").replace(" ", "_")
        project = self._settings.PROJECT.replace("-", "_").replace(" ", "_")

        schema_name = f"{tenant}_{project}"
        if schema_name and not schema_name.startswith("kdcube_"):
            schema_name = f"kdcube_{schema_name}"

        self.schema = schema_name

    async def init(self):
        # import ssl
        # ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # ctx.check_hostname = False
        # ctx.verify_mode = ssl.CERT_NONE
        async def _init_conn(conn: asyncpg.Connection):
            # Encode/decode json & jsonb as Python dicts automatically
            await conn.set_type_codec('json',  encoder=json.dumps, decoder=json.loads, schema='pg_catalog')
            await conn.set_type_codec('jsonb', encoder=json.dumps, decoder=json.loads, schema='pg_catalog')
            await conn.execute("SET TIME ZONE 'UTC'; SET datestyle = ISO, YMD;")

        if not self._pool:
            self._pool = await asyncpg.create_pool(
                host=self._settings.PGHOST,
                port=self._settings.PGPORT,
                user=self._settings.PGUSER,
                password=self._settings.PGPASSWORD,
                database=self._settings.PGDATABASE,
                ssl=self._settings.PGSSL,
                max_inactive_connection_lifetime=300.0,
                max_size=int(os.getenv("PGPOOL_MAX_SIZE", "10")),
                min_size=int(os.getenv("PGPOOL_MIN_SIZE", "1")),
                init=_init_conn,
                server_settings={"application_name": "kdcube-kb-client"},
            )

    async def close(self):
        if self._pool: await self._pool.close()

    async def hybrid_search(
            self, *, query:str, embedding:list[float] | None,
            top_n:int=8, include_expired:bool=False,
            providers: list[str] | None = None,         # NEW
    ) -> List[Dict[str,Any]]:
        # --- use websearch_to_tsquery ---
        fts_query = query.strip()
        use_fts = bool(fts_query)

        facet = ""
        if not include_expired:
            facet += f""" AND EXISTS (SELECT 1 FROM {self.schema}.datasource ds
                           WHERE ds.id=resource_id AND ds.version=rs.version
                           AND (ds.expiration IS NULL OR ds.expiration > now()))"""

        provider_filter = ""
        params = []
        if providers:
            provider_filter = " AND provider = ANY($2)"
            params.append(providers)

        async with self._pool.acquire() as con:
            bm25_rows = []
            if use_fts:
                q = (
                    f"""SELECT id, resource_id, version, provider, content, title, entities, tags, created_at
                        FROM {self.schema}.retrieval_segment rs
                        WHERE search_vector @@ websearch_to_tsquery('english', $1) {facet} {provider_filter}
                        ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', $1), 32) DESC
                        LIMIT {int(top_n*4)}"""
                )
                args = [fts_query] + params
                bm25_rows = await con.fetch(q, *args)

            ann_rows = []
            if embedding is not None:
                embedding = convert_embedding_to_string(embedding)
                q = (
                    f"""SELECT id, resource_id, version, provider, content, title, entities, tags, created_at,
                               (1.0 - (embedding <=> $1)) AS semantic_score
                        FROM {self.schema}.retrieval_segment rs
                        WHERE embedding IS NOT NULL {facet} {provider_filter}
                        ORDER BY embedding <=> $1
                        LIMIT {int(top_n*4)}"""
                )
                args = [embedding] + params
                ann_rows = await con.fetch(q, *args)

            out: Dict[str,Dict[str,Any]] = {}
            for r in bm25_rows:
                out[str(r["id"])] = dict(r) | {"bm25": 1.0, "semantic_score": 0.0}
            for r in ann_rows:
                cur = out.get(str(r["id"]))
                if cur:
                    cur["semantic_score"] = max(cur.get("semantic_score",0.0), float(r["semantic_score"]))
                else:
                    out[str(r["id"])] = dict(r) | {"bm25": 0.0}
            merged = list(out.values())
            merged.sort(key=lambda x: (x.get("semantic_score", 0.0), x.get("bm25", 0.0), x.get("created_at")), reverse=True)
            return merged[:top_n]

    async def hybrid_pipeline_search(
            self,
            params: Union[HybridSearchParams, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval (BM25 + ANN), then semantic scoring and optional rerank.
        Stage 1 & 2: retrieval_segment only (no join).
        Stage 3: JOIN datasource for provider/expiry/publication/modified filters on the *candidate set*.
        """

        # -------- normalize incoming params (allow dict) --------
        if isinstance(params, dict):
            params = HybridSearchParams(**params)

        rs = f"{self.schema}.retrieval_segment"
        ds = f"{self.schema}.datasource"

        # --- define TEXT JSON paths once (index-friendly; no ::timestamptz casts) ---
        pub_text = "(ds.metadata->'metadata'->>'published_time_iso')"
        mod_text = "(ds.metadata->'metadata'->>'modified_time_iso')"

        def normalize_query(q: str) -> str:
            nfkd = unicodedata.normalize('NFKD', q or "")
            cleaned = re.sub(r'[^0-9A-Za-z\s]', ' ', nfkd)
            return cleaned.lower().strip()

        def _build_entity_group(entity_filters, use_and: bool, sink: list) -> str:
            if not entity_filters:
                return ""
            parts = []
            for ent in entity_filters:
                parts.append("entities @> $%s::jsonb")
                sink.append(json.dumps([{"key": ent.key, "value": ent.value}]))
            joiner = " AND " if use_and else " OR "
            # We don’t know the $N numbers yet; we’ll backfill below.
            return "(" + joiner.join(parts) + ")"

        # -------- tokens for BM25 prefix query --------
        raw_query = params.query or ""
        q_norm = normalize_query(raw_query)
        terms = [t for t in q_norm.split() if len(t) > 3]
        prefix_tsquery = ' & '.join(f"{t}:*" for t in terms) if terms else ""

        # -------- FACETS that live on retrieval_segment (no join) --------
        rs_clauses: List[str] = []
        rs_params: List[Any] = []

        # resource_ids facet (fast with =ANY($n))
        if params.resource_ids:
            rs_clauses.append(f"resource_id = ANY($%s::text[])")
            rs_params.append(params.resource_ids)

        # entities (JSONB)
        entities_match_all = getattr(params, "entities_match_all", params.match_all)
        ent_params: List[Any] = []
        ent_clause_template = _build_entity_group(getattr(params, "entity_filters", None),
                                                  bool(entities_match_all), ent_params)
        if ent_clause_template:
            # patch $%s with correct indices after we know current param count
            base = len(rs_params) + 1
            patched = ent_clause_template
            for i in range(len(ent_params)):
                patched = patched.replace("$%s", f"${base+i}", 1)
            rs_clauses.append(patched)
            rs_params.extend(ent_params)

        rs_facets_sql = " AND ".join(rs_clauses) if rs_clauses else ""

        # -------- RECALL (query + tags) with match_all --------
        recall_clauses: List[str] = []
        recall_params: List[Any] = []

        if prefix_tsquery:
            recall_clauses.append("search_vector @@ to_tsquery('english', $%s)")
            recall_params.append(prefix_tsquery)

        if params.tags:
            if params.match_all:
                recall_clauses.append("tags @> $%s::text[]")
            else:
                recall_clauses.append("tags && $%s::text[]")
            recall_params.append(params.tags)

        if recall_clauses:
            # backfill $N in recall template
            base = len(rs_params) + 1
            patched_parts = []
            for i, tmpl in enumerate(recall_clauses):
                patched_parts.append(tmpl.replace("$%s", f"${base+i}"))
            recall_sql = "(" + (" AND " if params.match_all else " OR ").join(patched_parts) + ")"
            rs_params.extend(recall_params)
        else:
            recall_sql = ""

        # -------- Stage 1: BM25 over rs only --------
        # bind WHERE first
        where_parts = []
        if rs_facets_sql: where_parts.append(f"({rs_facets_sql})")
        if recall_sql:     where_parts.append(recall_sql)
        bm25_where = " AND ".join(where_parts) if where_parts else "TRUE"

        order_params: List[Any] = []  # <<< keep ORDER BY params separate
        if prefix_tsquery:
            # reference a NEW placeholder and push the param
            bm25_order = f"ts_rank_cd(search_vector, to_tsquery('english', ${len(rs_params)+1}), 32) DESC"
            order_params.append(prefix_tsquery)
        else:
            bm25_order = "created_at DESC"

        bm25_k = int(getattr(params, "bm25_k", 100))
        bm25_sql = f"""
            SELECT id AS segment_id
            FROM {rs}
            WHERE {bm25_where}
            ORDER BY {bm25_order}
            LIMIT {bm25_k}
        """

        async with self._pool.acquire() as con:
            # <<< pass rs_params + order_params
            bm25_rows = await con.fetch(bm25_sql, *(rs_params + order_params))
            bm25_ids = [r["segment_id"] for r in bm25_rows]

            # -------- Stage 2: ANN fallback (rs only) --------
            if params.embedding is None:
                return []

            # We pass vector literal as text; server casts during <=> comparison.
            from kdcube_ai_app.infra.embedding.embedding import convert_embedding_to_string
            emb_lit = convert_embedding_to_string(params.embedding)  # e.g., "[0.1,0.2,...]"
            fallback_k = int(getattr(params, "fallback_k", params.top_n))

            ann_clauses = ["embedding IS NOT NULL"]
            ann_params: List[Any] = []

            if rs_facets_sql:
                # We must reproduce the same rs facets with new $N numbering
                # Re-generate the facet SQL with fresh numbering
                # (simple approach: rebuild like above)
                ann_rs_clauses: List[str] = []
                ann_rs_params: List[Any] = []

                if params.resource_ids:
                    ann_rs_clauses.append(f"resource_id = ANY($%s::text[])")
                    ann_rs_params.append(params.resource_ids)

                ent_params2: List[Any] = []
                ent_clause2 = _build_entity_group(getattr(params, "entity_filters", None),
                                                  bool(entities_match_all), ent_params2)
                if ent_clause2:
                    base = len(ann_rs_params) + 1
                    patched = ent_clause2
                    for i in range(len(ent_params2)):
                        patched = patched.replace("$%s", f"${base+i}", 1)
                    ann_rs_clauses.append(patched)
                    ann_rs_params.extend(ent_params2)

                if ann_rs_clauses:
                    ann_clauses.append("(" + " AND ".join(ann_rs_clauses) + ")")
                    ann_params.extend(ann_rs_params)

            # <<< cast embedding param to ::vector for asyncpg
            ann_sql = f"""
                SELECT id AS segment_id
                FROM {rs}
                WHERE {" AND ".join(ann_clauses)}
                ORDER BY embedding <=> ${len(ann_params)+1}::vector
                LIMIT {fallback_k}
            """
            ann_rows = await con.fetch(ann_sql, *(ann_params + [emb_lit]))
            ann_ids = [r["segment_id"] for r in ann_rows]

            # -------- Stage 3: semantic scoring on union + JOIN ds (post-filter) --------
            candidate_ids = list(dict.fromkeys(bm25_ids + ann_ids))
            if not candidate_ids:
                return []

            # Build ds filters (providers/expiry/publication/modified) AFTER candidates known
            ds_filters: List[str] = []
            ds_params: List[Any] = []

            # ---------- provider facet
            if params.providers:
                ds_filters.append(f"ds.provider = ANY($%s::text[])")
                ds_params.append(params.providers)

            # ---------- expiry facet
            if not getattr(params, "include_expired", True):
                ds_filters.append("(ds.expiration IS NULL OR ds.expiration > now())")

            # ---------- publication time (TEXT, index-friendly)
            pub_clauses, pub_vals = dt_utils.build_temporal_filters(
                col_expr=pub_text, mode="text",
                on=params.published_on, after=params.published_after, before=params.published_before
            )
            # --- modified time filters ---
            mod_clauses, mod_vals = dt_utils.build_temporal_filters(
                col_expr=mod_text, mode="text",
                on=params.modified_on,  after=params.modified_after,  before=params.modified_before
            )
            ds_filters.extend(pub_clauses + mod_clauses)
            ds_params.extend(pub_vals + mod_vals)

            # Patch $%s placeholders with actual indexes (after the first two params)
            # Order of params: 1) emb_lit, 2) candidate_ids array, then ds_params...
            base = 2  # $1=embedding vector, $2=candidate_ids
            patched_ds_filters, _ = _patch_placeholders(ds_filters, start_index=base)
            ds_filter_sql = (" AND " + " AND ".join(patched_ds_filters)) if patched_ds_filters else ""

            # <<< cast $1 to ::vector; cast $2 to ::text[]
            semantic_sql = f"""
                SELECT
                  rs.id            AS segment_id,
                  rs.resource_id,
                  rs.version,
                  rs.provider,
                  rs.content,
                  rs.title,
                  rs.lineage,
                  rs.entities,
                  rs.extensions,
                  rs.tags,
                  rs.created_at,
                  (1.0 - (rs.embedding <=> $1::vector)) AS semantic_score,
                  CASE WHEN rs.embedding IS NOT NULL THEN 1.0 ELSE 0.0 END AS has_embedding,
                  jsonb_build_object(
                    'id', rs.resource_id,
                    'version', rs.version,
                    'rn', ds.rn,
                    'provider', ds.provider,
                    'title', ds.title,
                    'uri', ds.uri,
                    'system_uri', ds.system_uri,
                    'metadata', ds.metadata,
                    'active', (ds.expiration IS NULL OR ds.expiration > now()),
                    'published_time_iso', ds.metadata->'metadata'->>'published_time_iso',
                    'modified_time_iso',  ds.metadata->'metadata'->>'modified_time_iso'
                  ) AS datasource
                FROM {rs} rs
                JOIN {ds} ds
                  ON ds.id = rs.resource_id
                 AND ds.version = rs.version
                WHERE rs.id = ANY($2::text[])
                {ds_filter_sql}
            """
            ds_placeholders = sum(1 for _ in re.finditer(r"\$(\d+)", ds_filter_sql))
            # It counts all $N in DS part; we expect it to equal len(ds_params) plus
            # the two earlier placeholders in the main query ($1, $2) are outside ds_filter_sql.
            logger.debug("DS SQL: %s", ds_filter_sql)
            logger.debug("DS params (%d): %r", len(ds_params), ds_params)
            sem_rows = await con.fetch(semantic_sql, emb_lit, candidate_ids, *ds_params)

        # -------- local filtering/sorting + optional rerank --------
        thresh = params.min_similarity or 0.0
        filtered = [dict(r) for r in sem_rows if float(r.get("semantic_score", 0.0)) >= thresh]

        for r in filtered:
            dsj = (r.get("datasource") or {})
            pub = ((dsj.get("metadata") or {}).get("metadata") or {}).get("published_time_iso") or dsj.get("published_time_iso")
            r["_published_ts"] = pub

        filtered.sort(
            key=lambda r: (float(r.get("semantic_score", 0.0)),
                           float(r.get("has_embedding", 0.0)),
                           dt_utils.parse_ts_safe(r.get("_published_ts"))),
            reverse=True
        )
        top_sem = filtered[: params.top_n]

        # Optional cross-encoder rerank
        if params.should_rerank:
            try:
                from kdcube_ai_app.infra.rerank.rerank import cross_encoder_rerank
                if top_sem:
                    reranked = cross_encoder_rerank(raw_query or q_norm, top_sem, 'content')
                    if params.rerank_threshold is not None and len(reranked) > (params.rerank_top_k or params.top_n) * 2:
                       reranked = [r for r in reranked if r.get("rerank_score", 0.0) >= params.rerank_threshold]
                    top_sem = reranked[: (params.rerank_top_k or params.top_n)]
            except Exception:
                # Reranker unavailable — keep semantic order
                logger.error(traceback.format_exc())

        # strip helper field
        for r in top_sem:
            r.pop("_published_ts", None)
        return top_sem

    async def hybrid_pipeline_search_nojoin(
            self,
            params: Union[HybridSearchParams, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval over retrieval_segment only (NO JOIN).
        Stage 1: BM25/prefix recall (rs only)
        Stage 2: ANN fallback (rs only)
        Stage 3: semantic scoring + post-filters that read from rs.extensions->'datasource'

        Date filter semantics:
          - published_on / modified_on:  day-closed range  [day, day+1)
          - published_after / modified_after:  >= instant (if date-only, start-of-day)
          - published_before / modified_before: < instant (if date-only, start-of-day)
        """
        # Normalize param container
        if isinstance(params, dict):
            params = HybridSearchParams(**params)

        rs = f"{self.schema}.retrieval_segment"

        # JSON text paths inside extensions.datasource
        ds_node     = "(rs.extensions->'datasource')"
        ds_rn       = f"({ds_node}->>'rn')"
        ds_uri      = f"({ds_node}->>'uri')"
        ds_mime     = f"({ds_node}->>'mime')"
        ds_title    = f"({ds_node}->>'title')"
        ds_exp_text = f"({ds_node}->>'expiration')"
        ds_src_type = f"({ds_node}->>'source_type')"
        ds_pub_text = f"({ds_node}->>'published_time_iso')"
        ds_mod_text = f"({ds_node}->>'modified_time_iso')"

        def normalize_query(q: str) -> str:
            nfkd = unicodedata.normalize('NFKD', q or "")
            cleaned = re.sub(r'[^0-9A-Za-z\s]', ' ', nfkd)
            return cleaned.lower().strip()

        def _build_entity_group(entity_filters, use_and: bool, sink: list) -> str:
            if not entity_filters:
                return ""
            parts = []
            for ent in entity_filters:
                parts.append("entities @> $%s::jsonb")
                sink.append(json.dumps([{"key": ent.key, "value": ent.value}]))
            return "(" + (" AND " if use_and else " OR ").join(parts) + ")"

        def _is_date_only(x: Any) -> bool:
            return isinstance(x, str) and ("T" not in x) and len(x) >= 10 and x[4] == "-" and x[7] == "-"

        # --- Tokens for BM25 prefix recall
        raw_query = params.query or ""
        q_norm = normalize_query(raw_query)
        terms = [t for t in q_norm.split() if len(t) > 3]
        prefix_tsquery = ' & '.join(f"{t}:*" for t in terms) if terms else ""

        # --- Facets on retrieval_segment
        rs_clauses: List[str] = []
        rs_params: List[Any] = []

        if params.resource_ids:
            rs_clauses.append("resource_id = ANY($%s::text[])")
            rs_params.append(params.resource_ids)

        entities_match_all = getattr(params, "entities_match_all", params.match_all)
        ent_params: List[Any] = []
        ent_clause_template = _build_entity_group(getattr(params, "entity_filters", None),
                                                  bool(entities_match_all), ent_params)
        if ent_clause_template:
            base = len(rs_params) + 1
            patched = ent_clause_template
            for i in range(len(ent_params)):
                patched = patched.replace("$%s", f"${base+i}", 1)
            rs_clauses.append(patched)
            rs_params.extend(ent_params)

        rs_facets_sql = " AND ".join(rs_clauses) if rs_clauses else ""

        # --- Recall (query + tags)
        recall_clauses: List[str] = []
        recall_params: List[Any] = []

        if prefix_tsquery:
            recall_clauses.append("search_vector @@ to_tsquery('english', $%s)")
            recall_params.append(prefix_tsquery)

        if params.tags:
            if params.match_all:
                recall_clauses.append("tags @> $%s::text[]")
            else:
                recall_clauses.append("tags && $%s::text[]")
            recall_params.append(params.tags)

        if recall_clauses:
            base = len(rs_params) + 1
            patched_parts = []
            for i, tmpl in enumerate(recall_clauses):
                patched_parts.append(tmpl.replace("$%s", f"${base+i}"))
            recall_sql = "(" + (" AND " if params.match_all else " OR ").join(patched_parts) + ")"
            rs_params.extend(recall_params)
        else:
            recall_sql = ""

        # --- Stage 1: BM25 over rs only
        where_parts = []
        if rs_facets_sql: where_parts.append(f"({rs_facets_sql})")
        if recall_sql:     where_parts.append(recall_sql)
        bm25_where = " AND ".join(where_parts) if where_parts else "TRUE"

        order_params: List[Any] = []
        if prefix_tsquery:
            bm25_order = f"ts_rank_cd(search_vector, to_tsquery('english', ${len(rs_params)+1}), 32) DESC"
            order_params.append(prefix_tsquery)
        else:
            bm25_order = "created_at DESC"

        bm25_k = int(getattr(params, "bm25_k", 100))
        bm25_sql = f"""
            SELECT id AS segment_id
            FROM {rs}
            WHERE {bm25_where}
            ORDER BY {bm25_order}
            LIMIT {bm25_k}
        """

        async with self._pool.acquire() as con:
            bm25_rows = await con.fetch(bm25_sql, *(rs_params + order_params))
            bm25_ids = [r["segment_id"] for r in bm25_rows]

            # --- Stage 2: ANN fallback over rs only
            if params.embedding is None:
                return []

            emb_lit = convert_embedding_to_string(params.embedding)
            fallback_k = int(getattr(params, "fallback_k", params.top_n))

            ann_clauses = ["embedding IS NOT NULL"]
            ann_params: List[Any] = []

            if rs_facets_sql:
                ann_rs_clauses: List[str] = []
                ann_rs_params: List[Any] = []

                if params.resource_ids:
                    ann_rs_clauses.append("resource_id = ANY($%s::text[])")
                    ann_rs_params.append(params.resource_ids)

                ent_params2: List[Any] = []
                ent_clause2 = _build_entity_group(getattr(params, "entity_filters", None),
                                                  bool(entities_match_all), ent_params2)
                if ent_clause2:
                    base = len(ann_rs_params) + 1
                    patched = ent_clause2
                    for i in range(len(ent_params2)):
                        patched = patched.replace("$%s", f"${base+i}", 1)
                    ann_rs_clauses.append(patched)
                    ann_rs_params.extend(ent_params2)

                if ann_rs_clauses:
                    ann_clauses.append("(" + " AND ".join(ann_rs_clauses) + ")")
                    ann_params.extend(ann_rs_params)

            ann_sql = f"""
                SELECT id AS segment_id
                FROM {rs}
                WHERE {" AND ".join(ann_clauses)}
                ORDER BY embedding <=> ${len(ann_params)+1}::vector
                LIMIT {fallback_k}
            """
            ann_rows = await con.fetch(ann_sql, *(ann_params + [emb_lit]))
            ann_ids = [r["segment_id"] for r in ann_rows]

            # --- Stage 3: semantic scoring + post-filters (NO JOIN)
            candidate_ids = list(dict.fromkeys(bm25_ids + ann_ids))
            if not candidate_ids:
                return []

            seg_filters: List[str] = []
            seg_params: List[Any] = []

            # provider facet (rs.provider)
            if params.providers:
                seg_filters.append("rs.provider = ANY($%s::text[])")
                seg_params.append(params.providers)

            # expiration facet (extensions.datasource.expiration TEXT, compare as timestamptz)
            if not getattr(params, "include_expired", True):
                now_dt = datetime.now(timezone.utc)
                seg_filters.append(
                    f"({ds_exp_text} IS NULL OR {ds_exp_text} = '' OR {ds_exp_text}::timestamptz > $%s::timestamptz)"
                )
                seg_params.append(now_dt)

            # Published / Modified (TEXT mode)
            pub_clauses, pub_vals = dt_utils.build_temporal_filters(
                col_expr=ds_pub_text, mode="text",
                on=params.published_on, after=params.published_after, before=params.published_before
            )
            mod_clauses, mod_vals = dt_utils.build_temporal_filters(
                col_expr=ds_mod_text, mode="text",
                on=params.modified_on,  after=params.modified_after,  before=params.modified_before
            )
            seg_filters.extend(pub_clauses + mod_clauses)
            seg_params.extend(pub_vals + mod_vals)

            # Always include $3 for "now" even if include_expired=True? Only if we referenced it.
            patched_seg_filters, last_idx = dt_utils.patch_placeholders(seg_filters, start_index=2)
            seg_filter_sql = (" AND " + " AND ".join(patched_seg_filters)) if patched_seg_filters else ""

            sem_sql = f"""
                SELECT
                  rs.id            AS segment_id,
                  rs.resource_id,
                  rs.version,
                  rs.provider,
                  rs.content,
                  rs.title,
                  rs.lineage,
                  rs.entities,
                  rs.extensions,
                  rs.tags,
                  rs.created_at,
                  (1.0 - (rs.embedding <=> $1::vector)) AS semantic_score,
                  CASE WHEN rs.embedding IS NOT NULL THEN 1.0 ELSE 0.0 END AS has_embedding,
                  jsonb_build_object(
                    'rn',                {ds_rn},
                    'uri',               {ds_uri},
                    'mime',              {ds_mime},
                    'title',             {ds_title},
                    'source_type',       {ds_src_type},
                    'modified_time_iso', {ds_mod_text},
                    'published_time_iso',{ds_pub_text},
                    'expiration',        {ds_exp_text}
                  ) AS datasource
                FROM {rs} rs
                WHERE rs.id = ANY($2::text[])
                {seg_filter_sql}
            """

            # Args order matches: $1 embedding, $2 candidate_ids, then seg_params
            args = [emb_lit, candidate_ids] + seg_params
            sem_rows = await con.fetch(sem_sql, *args)

        # --- local filter/sort + rerank
        thresh = params.min_similarity or 0.0
        filtered = [dict(r) for r in sem_rows if float(r.get("semantic_score", 0.0)) >= thresh]

        for r in filtered:
            ds = r.get("datasource") or {}
            r["_published_ts"] = ds.get("published_time_iso")

        filtered.sort(
            key=lambda r: (float(r.get("semantic_score", 0.0)),
                           float(r.get("has_embedding", 0.0)),
                           dt_utils.parse_ts_safe(r.get("_published_ts"))),
            reverse=True
        )

        top_sem = filtered[: params.top_n]

        if params.should_rerank:
            try:
                from kdcube_ai_app.infra.rerank.rerank import cross_encoder_rerank
                if top_sem:
                    reranked = cross_encoder_rerank(raw_query or q_norm, top_sem, 'content')
                    if params.rerank_threshold is not None and len(reranked) > (params.rerank_top_k or params.top_n) * 2:
                        reranked = [r for r in reranked if r.get("rerank_score", 0.0) >= params.rerank_threshold]
                    top_sem = reranked[: (params.rerank_top_k or params.top_n)]
            except Exception:
                logger.error(traceback.format_exc())

        for r in top_sem:
            r.pop("_published_ts", None)

        return top_sem


