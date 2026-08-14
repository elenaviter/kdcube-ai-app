# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Query embeddings, cached where more than one process can find them.

A hybrid search embeds the QUERY on every search call. The index already keeps
an in-process LRU of query→vector, which covers a burst of identical searches
inside one worker — and covers nothing else. In a distributed runtime that is
most of the traffic:

  * pagination: "Load more" is the same query with a bigger window, and the
    next page may land on another worker;
  * the same person searching the same term ten minutes later, in a new turn;
  * two people converging on the same obvious term;
  * a search-backed tool called once per turn across many turns.

Each of those re-pays an embedder call for a vector that is a pure function of
(query, model, dimension). This is the SHARED half of that cache: a KIND of the
platform KV cache (the favicon cache's sibling), keyed by the query's digest so
the key is bounded and readable, and holding a JSON vector under a TTL.

It is deliberately a wrapper around `KVCache`, not a new cache: the connection,
the namespacing, the tenant/project prefixing, and the failure posture (a cache
that is down returns None and the caller carries on) already exist there.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import struct
import time
from typing import Any, List, Optional, Sequence

from kdcube_ai_app.infra.namespaces import REDIS
from kdcube_ai_app.infra.service_hub.cache import (
    KVCache,
    NamespacedKVCache,
    create_namespaced_kv_cache,
    ensure_namespaced_cache,
)

logger = logging.getLogger("kdcube.index.embedding_cache")

#: A query embedding is valid until the MODEL changes, which the key already
#: covers — so the TTL is about housekeeping, not correctness. A day keeps a
#: working set warm without holding a vector for every phrase ever typed.
DEFAULT_QUERY_EMBEDDING_TTL_SECONDS = int(
    os.getenv("QUERY_EMBEDDING_CACHE_TTL_SECONDS", "86400")
)

#: HOW MANY vectors one scope may hold. A vector is not small — 1536 float32
#: values are ~8KB base64-encoded (and ~17KB as JSON, which is why they are not
#: stored as JSON) — so an unbounded cache is a slow memory leak in Redis, not a
#: cache. At the default: 5000 x ~8KB ~= 40MB per tenant/project, and the
#: coldest entries are evicted on write, before the TTL would have expired them.
DEFAULT_QUERY_EMBEDDING_MAX_ENTRIES = int(
    os.getenv("QUERY_EMBEDDING_CACHE_MAX_ENTRIES", "5000")
)

#: Suffix of the per-scope recency index (a Redis sorted set of cache keys,
#: scored by last use) that makes the bound enforceable and the eviction LRU
#: rather than arbitrary.
INDEX_SUFFIX = "__lru"

_WHITESPACE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """The form the key is built from: trimmed, whitespace-collapsed, casefolded.

    Two searches that differ only in spacing or capitalization produce the same
    vector from the embedder, so they should produce the same key. Nothing else
    is normalized — stemming or stop-word removal would make the key disagree
    with what was actually embedded."""
    return _WHITESPACE.sub(" ", str(query or "").strip()).casefold()


def query_embedding_key(query: str, *, model: str = "", dim: int | None = None) -> str:
    """`<model>:<dim>:<sha256(normalized query)>` — bounded, and safe to log.

    The model and the dimension are part of the identity because a vector from
    another model is not a cheaper answer, it is a wrong one."""
    digest = hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()
    return f"{str(model or 'default').strip()}:{int(dim or 0)}:{digest}"


def encode_vector(vector: Sequence[float]) -> str:
    """float32, packed and base64'd — half the size of the JSON array and exact
    to the precision an embedder actually returns."""
    values = [float(x) for x in vector]
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def decode_vector(raw: Any) -> Optional[List[float]]:
    """The packed form, or a legacy JSON array — whichever is in the cache."""
    if isinstance(raw, list):  # written before the compact codec; still valid
        try:
            return [float(x) for x in raw]
        except Exception:
            return None
    if not isinstance(raw, str) or not raw:
        return None
    try:
        blob = base64.b64decode(raw.encode("ascii"), validate=True)
        if len(blob) % 4:
            return None
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))
    except Exception:
        return None


class QueryEmbeddingCache:
    """A shared query→vector cache over the platform KV cache.

    Every method fails soft: a miss, a malformed payload, a Redis that is not
    answering — all return None and let the caller embed. A search must never
    fail because a cache did."""

    def __init__(
        self,
        cache: KVCache,
        *,
        model: str = "",
        dim: int | None = None,
        ttl_seconds: int = DEFAULT_QUERY_EMBEDDING_TTL_SECONDS,
        max_entries: int = DEFAULT_QUERY_EMBEDDING_MAX_ENTRIES,
    ) -> None:
        self.cache = cache
        self.model = str(model or "")
        self.dim = int(dim or 0)
        self.ttl_seconds = int(ttl_seconds)
        #: The scope's ceiling. Vectors are big; a cache nobody bounds is a
        #: memory leak that happens to answer questions.
        self.max_entries = max(0, int(max_entries))
        #: Observability for the caller: how often the shared cache actually
        #: spared an embed call.
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def key(self, query: str) -> str:
        return query_embedding_key(query, model=self.model, dim=self.dim)

    def _valid(self, values: Optional[List[float]]) -> Optional[List[float]]:
        if not values:
            return None
        if self.dim and len(values) != self.dim:
            # A stored vector of another width belongs to another model/config.
            return None
        return values

    # ---- the bound: a per-scope LRU index in Redis ----
    #
    # The KV cache is a flat keyspace with a TTL; on its own that bounds the AGE
    # of an entry, not how many there are. A sorted set of this scope's keys,
    # scored by last use, makes the ceiling enforceable and makes the eviction
    # least-recently-USED rather than whatever expired first. All of it is
    # best-effort: if the index cannot be maintained the cache still answers,
    # and Redis' own `maxmemory-policy` remains the backstop underneath.

    def _redis(self):
        return getattr(self.cache, "redis", None)

    def _index_key(self) -> str:
        # Same namespace/tenant prefix as the entries themselves.
        keyfn = getattr(self.cache, "_key", None)
        return keyfn(INDEX_SUFFIX) if callable(keyfn) else INDEX_SUFFIX

    async def _touch(self, stored_key: str) -> None:
        redis = self._redis()
        if redis is None or not self.max_entries:
            return
        try:
            index = self._index_key()
            await redis.zadd(index, {stored_key: time.time()})
            await redis.expire(index, max(self.ttl_seconds * 2, self.ttl_seconds))
        except Exception:
            return

    async def _enforce_bound(self) -> None:
        redis = self._redis()
        if redis is None or not self.max_entries:
            return
        try:
            index = self._index_key()
            size = int(await redis.zcard(index) or 0)
            overflow = size - self.max_entries
            if overflow <= 0:
                return
            coldest = await redis.zrange(index, 0, overflow - 1)
            if not coldest:
                return
            keys = [k.decode() if isinstance(k, bytes) else str(k) for k in coldest]
            await redis.delete(*keys)
            await redis.zrem(index, *keys)
            self.evictions += len(keys)
            logger.info(
                "[query-embedding-cache] evicted %d coldest of %d (cap %d)",
                len(keys), size, self.max_entries,
            )
        except Exception:
            return

    async def get(self, query: str) -> Optional[List[float]]:
        key = self.key(query)
        try:
            stored = await self.cache.get(key)
        except Exception:
            return None
        vector = self._valid(decode_vector(stored))
        if vector is None:
            self.misses += 1
            return None
        self.hits += 1
        # A read is a use: re-score the entry so the bound evicts what is
        # genuinely cold, not what happens to be oldest.
        await self._touch(self._stored_key(key))
        return vector

    def _stored_key(self, key: str) -> str:
        keyfn = getattr(self.cache, "_key", None)
        return keyfn(key) if callable(keyfn) else key

    async def set(self, query: str, vector: Sequence[float]) -> bool:
        values = self._valid([float(x) for x in (vector or [])])
        if values is None:
            return False
        key = self.key(query)
        try:
            ok = await self.cache.set(
                key, encode_vector(values), ttl_seconds=self.ttl_seconds
            )
        except Exception:
            return False
        if ok:
            await self._touch(self._stored_key(key))
            await self._enforce_bound()
        return bool(ok)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "model": self.model,
            "dim": self.dim,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }


def create_query_embedding_cache(
    *,
    model: str = "",
    dim: int | None = None,
    tenant: Optional[str] = None,
    project: Optional[str] = None,
    ttl_seconds: int = DEFAULT_QUERY_EMBEDDING_TTL_SECONDS,
    max_entries: int = DEFAULT_QUERY_EMBEDDING_MAX_ENTRIES,
    use_tp_prefix: bool = True,
    cache: KVCache | None = None,
) -> Optional[QueryEmbeddingCache]:
    """The cache for this scope, or None when Redis is unreachable.

    Pass an existing `KVCache` to reuse a connection the caller already holds;
    otherwise one is created for the `kdcube:cache:query-embedding` namespace,
    tenant/project-prefixed like every other namespaced cache (a query is not
    secret, but the vector belongs to a project's model configuration).
    """
    try:
        if cache is not None:
            ns_cache: NamespacedKVCache = ensure_namespaced_cache(
                cache,
                namespace=REDIS.CACHE.QUERY_EMBEDDING,
                tenant=tenant,
                project=project,
                default_ttl_seconds=ttl_seconds,
                use_tp_prefix=use_tp_prefix,
            )
        else:
            ns_cache = create_namespaced_kv_cache(
                namespace=REDIS.CACHE.QUERY_EMBEDDING,
                tenant=tenant,
                project=project,
                default_ttl_seconds=ttl_seconds,
                use_tp_prefix=use_tp_prefix,
            )
    except Exception:
        logger.info(
            "[query-embedding-cache] unavailable; search will embed every query",
            exc_info=True,
        )
        return None
    if ns_cache is None:
        return None
    return QueryEmbeddingCache(
        ns_cache, model=model, dim=dim,
        ttl_seconds=ttl_seconds, max_entries=max_entries,
    )
