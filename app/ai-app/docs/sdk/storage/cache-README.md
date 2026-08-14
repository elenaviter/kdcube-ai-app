---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/storage/cache-README.md
title: "Cache"
summary: "Redis KV cache abstraction for the SDK, including namespaced (tenant/project) caches."
tags: ["sdk", "storage", "cache", "redis"]
keywords: ["KVCache", "NamespacedKVCache", "tenant/project scope", "ServiceHub cache", "redis"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-storage-and-cache-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/storage/sdk-store-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/storage/git-store-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/README.md
---
# KV Cache (Service Hub)

This module provides a platform‑level Redis KV cache abstraction used across the SDK.
It supports:
- raw KV cache (no namespacing)
- namespaced KV cache (optionally tenant/project‑scoped)

All caches are async and backed by Redis.

## Concepts

### KVCache
Raw key/value cache with optional TTL. No namespace or tenant/project prefixing.

### NamespacedKVCache
Wraps KVCache with a namespace prefix. It can optionally include tenant/project
prefixes (default behavior).

## When to use what

- **KVCache**: pass around as a generic cache in integrations / runtime payloads.
- **NamespacedKVCache**: use only at the call site where you need key isolation.

## API

### Create a raw cache
```
from kdcube_ai_app.infra.service_hub.cache import create_kv_cache

cache = create_kv_cache()
```

### Create a raw cache from env
```
from kdcube_ai_app.infra.service_hub.cache import create_kv_cache_from_env

cache = create_kv_cache_from_env()
```

### Create a namespaced cache (tenant/project scoped)
```
from kdcube_ai_app.infra.service_hub.cache import create_namespaced_kv_cache
from kdcube_ai_app.infra.namespaces import REDIS
from kdcube_ai_app.apps.chat.sdk.config import get_settings

settings = get_settings()
cache = create_namespaced_kv_cache(
    namespace=REDIS.CACHE.FAVICON,
    tenant=settings.TENANT,
    project=settings.PROJECT,
)
```

### Convert a KVCache into a namespaced cache
```
from kdcube_ai_app.infra.service_hub.cache import ensure_namespaced_cache
from kdcube_ai_app.infra.namespaces import REDIS

ns_cache = ensure_namespaced_cache(
    cache,
    namespace=REDIS.CACHE.FAVICON,
    tenant="t1",
    project="p1",
)
```

### Cross‑tenant / global cache
If you want a cache shared across all tenants/projects, disable prefixing:
```
from kdcube_ai_app.infra.service_hub.cache import ensure_namespaced_cache
from kdcube_ai_app.infra.namespaces import REDIS

ns_cache = ensure_namespaced_cache(
    cache,
    namespace=REDIS.CACHE.FAVICON,
    use_tp_prefix=False,  # no tenant/project prefix
)
```

## Kinds: a cache with a contract on top

`KVCache` is the mechanism. A **kind** is a small wrapper that owns one key
grammar, one TTL, and one validity rule — so callers share a cache without
sharing conventions. The favicon cache was the first; the query-embedding cache
is the second.

### Query embeddings (`kdcube:cache:query-embedding`)

A hybrid search embeds the QUERY on every call, and that vector is a pure
function of `(query, model, dim)`. The repeats worth catching — pagination, a
later turn, two people typing the same term — happen on **different workers**,
so the cache has to be shared. The index keeps no process-local memo on purpose:
in a distributed runtime that would be a cache one worker out of N can use and
none of them can invalidate.

```python
from kdcube_ai_app.infra.index.embedding_cache import create_query_embedding_cache

cache = create_query_embedding_cache(
    model="text-embedding-3-small", dim=1536,
    tenant=settings.TENANT, project=settings.PROJECT,
)   # None when Redis is unreachable — the caller then simply embeds
```

- **Key**: `<model>:<dim>:<sha256(normalized query)>`. Normalization is trim +
  whitespace collapse + casefold, and nothing else — stemming would make the key
  disagree with what was actually embedded. The model and width are part of the
  identity because a vector from another model is not a cheaper answer, it is a
  wrong one.
- **Size**: the vector is stored as base64-packed **float32**, not JSON — 1536
  values are ~8KB packed against ~17KB as a JSON array, and the JSON form is
  also lossier to read back.
- **Bound**: `QUERY_EMBEDDING_CACHE_MAX_ENTRIES` (default 5000) per scope, so
  ~40MB per tenant/project at 1536 dims. Vectors are big; a cache nobody bounds
  is a memory leak that happens to answer questions. The ceiling is enforced by
  a Redis sorted set of the scope's keys scored by **last use** (`…:__lru`), so
  a write over the cap evicts the coldest entries, not the oldest — and a read
  re-scores its entry. Redis' own `maxmemory-policy` stays the backstop
  underneath; the index is best-effort and never fails a call.
- **TTL**: `QUERY_EMBEDDING_CACHE_TTL_SECONDS` (default 86400). Correctness does
  not depend on it; it bounds age, while the entry count is bounded above.
- **Failure posture**: every method fails soft. A miss, a malformed payload, a
  Redis that is not answering — all return `None`, and the search embeds. A
  search must never fail because a cache did.
- **Wiring**: hand it to `IndexConfig.query_embedding_cache`
  ([Hybrid Index](../solutions/index/hybrid-index-README.md#optional-shared-query-embedding-cache)).
  It is opt-in per index instance, because the caller is the one who knows
  whether its queries repeat and under whose tenant/project the vectors belong.

## Env vars

Required:
- `REDIS_URL`

Optional:
- `KV_CACHE_TTL_SECONDS` (default: 3600)
- `FAVICON_CACHE_TTL_SECONDS` (default: 86400)
- `QUERY_EMBEDDING_CACHE_TTL_SECONDS` (default: 86400)
- `QUERY_EMBEDDING_CACHE_MAX_ENTRIES` (default: 5000 per tenant/project scope)

## Notes

- `NamespacedKVCache` derives from `KVCache` and only overrides `_key()`.
- The runtime should pass **KVCache** through integrations; convert to namespaced
  only where needed.
