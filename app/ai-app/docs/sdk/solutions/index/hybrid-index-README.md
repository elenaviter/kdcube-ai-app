---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/index/hybrid-index-README.md
title: "SQLite + Vector Hybrid Index"
summary: "Reusable per-scope search index that internalizes SQLite lexical (FTS5/bm25), embed-on-write vectors, pluggable vector backends (brute-force / file faiss / cross-process faiss), recency decay, and RRF fusion — so any searchable collection (pins, tasks, memories) gets semantic + lexical + recency + reciprocal-rank search by handing it Documents and a query."
status: active
tags: ["sdk", "solutions", "index", "search", "sqlite", "faiss", "embeddings", "hybrid-search", "fts5"]
keywords:
  [
    "hybrid index",
    "sqlite fts5",
    "bm25",
    "faiss",
    "vector store",
    "embed-on-write",
    "semantic search",
    "lexical search",
    "HybridIndex",
    "IndexConfig",
    "BruteForceVectorStore",
    "LocalFaissStore",
    "CachedFaissStore",
    "ensure_built",
    "per-scope index",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/index/hybrid-scoring-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/canvas/pin-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/synch-mechanisms/critical-section-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/storage/cache-README.md
---
# SQLite + Vector Hybrid Index

`kdcube_ai_app/infra/index/sqlite` is a generic, reusable search index for any
**per-scope collection** — a user's pin boards, a project's issues, a session's
memories. It internalizes the parts every such collection re-invents:

- a SQLite document store (id, text blob, JSON metadata, timestamp),
- **lexical** ranking via SQLite FTS5 / `bm25`,
- **semantic** ranking via embed-on-write vectors over a pluggable vector backend,
- **recency** decay, and
- **fusion** of the three into one ranking (Reciprocal Rank Fusion).

A caller hands it `Document`s and a query string; it returns ranked `SearchHit`s.
It does not know what a pin or an issue is — only `Document(id, text, metadata,
timestamp)`. The scoring half is documented separately in
[Hybrid Scoring](./hybrid-scoring-README.md); this page is the index lifecycle and
the vector backends.

## Why it exists

Pins, tasks, and memories all want "semantic + lexical + recency, reciprocal" search,
but they live in different stores and at different scales. Rather than each bundle
re-deriving FTS5 setup, embed-on-write bookkeeping, faiss build/eval, and fusion, they
share one index and differ only in:

1. how they compose a `Document.text` from their domain object, and
2. which vector backend they pick for their scale.

The canvas pin board is the first adopter (see
[Pin Integration](../canvas/pin-integration-README.md)); the same index serves any
future searchable collection.

## Public surface

```python
from kdcube_ai_app.infra.index.sqlite import (
    HybridIndex, IndexConfig, Document, SearchHit, FusionWeights,
    BruteForceVectorStore,
)
from kdcube_ai_app.infra.index.faiss import LocalFaissStore, CachedFaissStore

idx = HybridIndex(IndexConfig(
    db_path=Path(".../collection.index.sqlite"),
    embed_fn=model_service.embed_texts,    # async batch embedder: List[str] -> List[List[float]]
    dim=1536,                              # text-embedding-3-small
    vector_store=LocalFaissStore(".../collection.index.faiss"),   # production: file-backed faiss
    #            CachedFaissStore(cache, scope)                   # cross-process / Redis-cached faiss
    #            BruteForceVectorStore()                          # no-dep fallback (tests / no-faiss envs)
))

await idx.upsert([Document(id="a:1", text="...", metadata={"board": "b1"}, timestamp=...)])
await idx.delete(["a:1"])
indexed_ids = idx.ids(filters={"board": "b1"})   # to diff against a live collection
await idx.ensure_built()                         # (re)build the vector store iff data changed
hits = await idx.search("query", top_k=20, filters={"board": "b1"})
```

`embed_fn` is exactly the platform's `model_service.embed_texts`, so any bundle's
embedder satisfies it with no adapter.

## Documents

```python
@dataclass
class Document:
    id: str                         # caller-unique within the scope
    text: str                       # the searchable blob the caller composes
    metadata: Dict[str, Any] = {}   # returned on hits; single-key equality filterable
    timestamp: float | None = None  # epoch seconds; drives recency; defaults to now at upsert
```

The caller owns `text` composition — concatenate whatever should be searchable
(label + summary + description + comments…). `metadata` is returned verbatim on
each hit and can be filtered with single-key equality (`filters={"board": id}`),
implemented as `json_extract(metadata_json, '$.key') = ?`. For set membership
(several kinds, several namespaces) over-fetch and post-filter in the adapter; the
sets are small.

## Embed-on-write

`upsert()` embeds **only new or text-changed** documents. A metadata-only or
timestamp-only edit does not re-embed and does not invalidate the vector build —
it just updates the row. This is the cost rule: you pay the embedder when the
searchable text actually changes, not on every touch and never on read of an
already-indexed document. (Query-time embedding is separately gated; see Scoring.)

Internally each upsert that changed vectors bumps a `data_version`; the vector
build records the `built_version` it was built from. They are compared in
`ensure_built()`.

## Build lifecycle: `ensure_built`

The SQLite tables (docs, FTS, the cached vectors) are the source of truth. The
vector store is a derived structure rebuilt from the cached vectors when stale:

```python
await idx.ensure_built()   # no-op if built_version == data_version; else rebuild()
```

- **Volatile backends** (`BruteForceVectorStore`, in-memory) cannot trust a persisted
  `built_version` — a fresh process starts empty — so the index forces a rebuild on
  first use by resetting `built_version = -1` in `__init__`.
- **Non-volatile backends** (`LocalFaissStore` file, `CachedFaissStore` cross-process)
  keep their built artifact and skip the rebuild when versions match.

Call `ensure_built()` after a batch of writes, inside whatever lock serializes
writes for the scope. For the canvas pin board that lock is the runtime's observed
file lock — see [Synchronization Mechanisms](../../../service/synch-mechanisms/critical-section-README.md).
`search()` also calls `ensure_built()` before a semantic pass as a safety net, but
the canonical place to build is on update, not on read.

## Vector backends

All backends key on the SQLite rowids and use cosine similarity on L2-normalized
vectors, so they are interchangeable with no other change. Pick by scale:

| Backend | Deps | Persistence | Use for |
|---|---|---|---|
| `LocalFaissStore(path)` | faiss + numpy | file next to the SQLite (on EFS → shared across the cluster) | **production default** — file-backed faiss; maintenance guarded by a [critical section](../../../service/synch-mechanisms/critical-section-README.md) on the scope |
| `CachedFaissStore(cache, scope)` | faiss + numpy | cross-process via `FaissProjectCache` (Redis-coordinated) | deployments with no shared filesystem |
| `BruteForceVectorStore` | none | volatile (in-memory) | no-dep fallback for tests / no-faiss envs — **not a production path** |

`faiss` and `numpy` are optional imports; the brute-force fallback needs neither, so
the index is importable and unit-testable in any environment without faiss installed.
The faiss stores use `IndexFlatIP + IDMap2` (exact inner product on normalized
vectors = cosine).

## Search modes and degradation

`search(query, mode="hybrid"|"lexical"|"semantic")` defaults to `hybrid`. The
semantic pass is **economically gated** (the embedder call costs money): if the
guard denies or the query is trivial, the semantic pass is skipped and the query
degrades to lexical + recency — still returning results, at no embed cost. The
semantic factor can also be turned **off** outright (`min_semantic_score < 0`, e.g.
`-1`) for a scope with no embeddings/budget, and it is **fail-soft**: a runtime
embedder/vector error degrades to lexical instead of failing the search. A repeated
query re-embeds unless the caller configured the shared query→vector cache below —
there is no process-local memo, deliberately (see the next section). The gating knobs (including
the three `min_semantic_score` regimes) and the fusion math are in
[Hybrid Scoring](./hybrid-scoring-README.md).

### Optional: shared query-embedding cache

A search embeds the query on every call, and that vector is a pure function of
`(query, model, dim)`. The repeats worth catching happen **across processes**:
"Load more" is the same query with a bigger window and may land on another
worker, a later turn re-asks the same thing, two people converge on the same
obvious term, and a search-backed tool runs once per turn for weeks.

The index therefore keeps **no process-local memo**. This runtime is
distributed and an index instance is rebuilt freely, so anything remembered
inside one object is a cache one worker out of N can use and none of them can
invalidate. The cache that means something is shared, and it is the caller's to
configure:

```python
from kdcube_ai_app.infra.index.embedding_cache import create_query_embedding_cache

idx = HybridIndex(IndexConfig(
    db_path=..., embed_fn=model_service.embed_texts, dim=1536,
    vector_store=LocalFaissStore(...),
    query_embedding_cache=create_query_embedding_cache(
        model="text-embedding-3-small", dim=1536,
        tenant=settings.TENANT, project=settings.PROJECT,
    ),
))
```

Opt-in on purpose: the caller knows whether its queries repeat, and under whose
tenant/project the vectors belong. Leave it `None` and every query is embedded.
The cache is bounded per scope (default 5000 entries, evicting least-recently-used)
and stores packed float32 rather than JSON — a vector cache without a ceiling is a
memory leak. The tier is fail-soft in both directions — a cache that cannot be read
or written costs an embed call, never the search — and the key carries the model
and the width, so a re-pointed embedder cannot serve stale vectors. The kind, its
key grammar and TTL: [Cache](../../storage/cache-README.md#query-embeddings-kdcubecachequery-embedding).

## What it does not do

- It is not multi-tenant by itself: one index = one scope (one SQLite file). Callers
  put per-user / per-project scope in the **path** and per-board / per-kind scope in
  **metadata filters**.
- It does not own the write lock. Serializing concurrent writers is the caller's job
  (and a solved one — use an observed file lock for filesystem-backed scopes).
- It does not compose `Document.text` or interpret `metadata` keys — that is the
  domain adapter's job (e.g. the canvas `PinSearchIndex`).
