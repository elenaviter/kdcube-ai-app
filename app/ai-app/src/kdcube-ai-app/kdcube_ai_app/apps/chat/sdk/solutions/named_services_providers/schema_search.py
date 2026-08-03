# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Persistent hybrid search over a named-service capability catalog."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.schema_catalog import (
    catalog_operation_entries,
    search_catalog_lexical,
)
from kdcube_ai_app.infra.index.sqlite import (
    BruteForceVectorStore,
    Document,
    FusionWeights,
    HybridIndex,
    IndexConfig,
)
from kdcube_ai_app.storage.observed_file_locks import observed_file_lock_async


SCHEMA_SEARCH_MODES = frozenset({"lexical", "semantic", "hybrid"})
DEFAULT_SCHEMA_SEARCH_LIMIT = 10
MAX_SCHEMA_SEARCH_LIMIT = 50
DEFAULT_SCHEMA_VECTOR_BACKEND = "faiss-local"
SCHEMA_VECTOR_BACKENDS = frozenset({"faiss-local", "bruteforce"})


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")).strip("_") or "provider"


def normalize_schema_vector_backend(value: Any) -> str:
    name = str(value or DEFAULT_SCHEMA_VECTOR_BACKEND).strip().lower()
    if name in {"faiss", "faiss-local", "local"}:
        return "faiss-local"
    if name in {"brute", "brute-force", "bruteforce", "memory", "in-memory"}:
        return "bruteforce"
    raise ValueError(
        "named-services schema vector backend must be one of: "
        + ", ".join(sorted(SCHEMA_VECTOR_BACKENDS))
    )


def schema_search_vector_path(db_path: str | Path) -> Path:
    return Path(db_path).with_suffix(".faiss")


def _schema_vector_store(db_path: Path, *, backend: str):
    selected = normalize_schema_vector_backend(backend)
    if selected == "faiss-local":
        from kdcube_ai_app.infra.index.faiss import LocalFaissStore

        return selected, LocalFaissStore(schema_search_vector_path(db_path))
    return selected, BruteForceVectorStore()


def schema_search_index_path(
    storage_root: str | Path,
    provider_id: str,
    *,
    embedding_profile: str = "",
) -> Path:
    profile_suffix = (
        "." + hashlib.sha256(embedding_profile.encode("utf-8")).hexdigest()[:16]
        if embedding_profile
        else ""
    )
    return (
        Path(storage_root)
        / ".named-service-schema"
        / _safe(provider_id)
        / f"capabilities{profile_suffix}.sqlite"
    )


def schema_search_embedding_profile(
    model_service: Any,
    *,
    dim: int,
    vector_backend: str = "",
) -> str:
    """Return a stable, non-secret identity for the configured embedder."""

    provider = str(getattr(model_service, "provider", None) or "").strip()
    model = str(getattr(model_service, "model", None) or "").strip()
    underlying = getattr(model_service, "model_service", None) or model_service
    emb_model = getattr(underlying, "_emb_model", None)
    try:
        provider_obj = getattr(getattr(emb_model, "provider", None), "provider", None)
        provider = str(getattr(provider_obj, "value", provider_obj) or provider).strip()
        model = str(getattr(emb_model, "systemName", None) or model).strip()
    except Exception:
        pass
    cfg = getattr(getattr(underlying, "config", None), "embedder_config", None)
    if isinstance(cfg, Mapping):
        provider = str(cfg.get("provider") or provider).strip()
        model = str(cfg.get("model_name") or cfg.get("model") or model).strip()
    service_type = type(underlying)
    return json.dumps(
        {
            "provider": provider,
            "model": model,
            "dim": max(1, int(dim or 1536)),
            "service": f"{service_type.__module__}.{service_type.__qualname__}",
            "vector_backend": (
                normalize_schema_vector_backend(vector_backend)
                if vector_backend
                else ""
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _document_id(entry: Mapping[str, Any], *, embedding_profile: str = "") -> str:
    identity = "\n".join(
        (
            embedding_profile,
            str(entry.get("catalog_path") or ""),
            str(entry.get("object_kind") or ""),
            str(entry.get("schema_operation") or ""),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _document_text(entry: Mapping[str, Any]) -> str:
    return "\n".join(
        text
        for text in (
            str(entry.get("label") or ""),
            str(entry.get("description") or ""),
            " ".join(str(item) for item in entry.get("keywords") or []),
            " ".join(str(item) for item in entry.get("catalog_terms") or []),
            " ".join(str(item) for item in entry.get("catalog_breadcrumbs") or []),
            str(entry.get("catalog_path") or ""),
            str(entry.get("object_kind") or ""),
            str(entry.get("schema_operation") or ""),
        )
        if text
    )


def catalog_search_documents(
    tree: Mapping[str, Any],
    *,
    embedding_profile: str = "",
) -> list[Document]:
    return [
        Document(
            id=_document_id(entry, embedding_profile=embedding_profile),
            text=_document_text(entry),
            metadata={
                key: value
                for key, value in entry.items()
                if key != "catalog_terms"
            },
            timestamp=0,
        )
        for entry in catalog_operation_entries(tree)
    ]


def _catalog_signature(docs: list[Document]) -> str:
    payload = [
        {
            "id": doc.id,
            "text": doc.text,
            "metadata": doc.metadata,
        }
        for doc in sorted(docs, key=lambda item: item.id)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(value, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _result_mode(sources: set[str]) -> str:
    content_sources = sources & {"lexical", "semantic"}
    if content_sources == {"lexical", "semantic"}:
        return "hybrid"
    if len(content_sources) == 1:
        return next(iter(content_sources))
    return "no_matches"


class SchemaCatalogSearchIndex:
    """Shared hybrid index over one provider capability catalog.

    SQLite persists documents, FTS rows, and cached vectors in bundle storage.
    The default FAISS vector view is a derived sibling file; the explicit
    brute-force fallback reconstructs its volatile view from the same cache.
    Document embeddings are computed only for new or changed declarations.
    """

    def __init__(
        self,
        *,
        db_path: str | Path,
        model_service: Any,
        dim: int = 1536,
        min_semantic_score: float = 0.20,
        vector_backend: str = DEFAULT_SCHEMA_VECTOR_BACKEND,
    ) -> None:
        embed_texts = getattr(model_service, "embed_texts", None)
        if not callable(embed_texts):
            raise ValueError("schema catalog hybrid search requires model_service.embed_texts")
        self.db_path = Path(db_path)
        self.signature_path = self.db_path.with_suffix(".signature")
        self.model_service = model_service
        self.dim = max(1, int(dim or 1536))
        self.vector_backend, vector_store = _schema_vector_store(
            self.db_path,
            backend=vector_backend,
        )
        self.vector_path = (
            schema_search_vector_path(self.db_path)
            if self.vector_backend == "faiss-local"
            else None
        )
        self.index = HybridIndex(
            IndexConfig(
                db_path=self.db_path,
                embed_fn=embed_texts,
                model_service=model_service,
                dim=self.dim,
                vector_store=vector_store,
                weights=FusionWeights(lexical=1.0, semantic=1.0, recency=0.0),
                min_semantic_score=float(min_semantic_score),
            )
        )

    async def _ensure_vector_store_built(self) -> None:
        # HybridIndex's persisted build version normally avoids rebuilding a
        # file-backed vector store. If the derived FAISS file was removed while
        # SQLite survived, invalidate that version and reconstruct it from the
        # cached vectors instead of silently serving lexical-only results.
        if self.vector_path is not None and not self.vector_path.exists():
            await self.index.rebuild()
            return
        await self.index.ensure_built()

    async def ensure(self, tree: Mapping[str, Any]) -> dict[str, Any]:
        embedding_profile = schema_search_embedding_profile(
            self.model_service,
            dim=self.dim,
            vector_backend=self.vector_backend,
        )
        docs = catalog_search_documents(
            tree,
            embedding_profile=embedding_profile,
        )
        expected = {doc.id for doc in docs}
        signature = _catalog_signature(docs)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with observed_file_lock_async(
            lock_path=self.db_path.with_name(self.db_path.name + ".lock"),
            resource_id=f"named-services.schema:{self.db_path.parent.name}",
            operation="named-services.schema.index.update",
            wait_seconds=30,
        ):
            stored_signature = ""
            try:
                stored_signature = self.signature_path.read_text(
                    encoding="utf-8"
                ).strip()
            except FileNotFoundError:
                pass
            if stored_signature == signature and set(self.index.ids()) == expected:
                await self._ensure_vector_store_built()
                return {
                    "ok": True,
                    "indexed": len(docs),
                    "updated": False,
                    "signature": signature,
                    "db_path": str(self.db_path),
                    "vector_backend": self.vector_backend,
                    "vector_path": str(self.vector_path) if self.vector_path else "",
                }
            stale = set(self.index.ids()) - expected
            if stale:
                await self.index.delete(stale)
            if docs:
                await self.index.upsert(docs)
            await self._ensure_vector_store_built()
            _atomic_write_text(self.signature_path, signature + "\n")
        return {
            "ok": True,
            "indexed": len(docs),
            "updated": True,
            "signature": signature,
            "db_path": str(self.db_path),
            "vector_backend": self.vector_backend,
            "vector_path": str(self.vector_path) if self.vector_path else "",
        }

    async def search(
        self,
        tree: Mapping[str, Any],
        *,
        query: str,
        mode: str,
        limit: int,
        object_kind: str = "",
    ) -> dict[str, Any]:
        requested_mode = str(mode or "hybrid").strip().lower()
        if requested_mode not in SCHEMA_SEARCH_MODES:
            raise ValueError(
                "search_mode must be one of: " + ", ".join(sorted(SCHEMA_SEARCH_MODES))
            )
        bounded_limit = max(1, min(int(limit or DEFAULT_SCHEMA_SEARCH_LIMIT), MAX_SCHEMA_SEARCH_LIMIT))
        await self.ensure(tree)
        filters = {"object_kind": object_kind} if object_kind else None
        try:
            hits = await self.index.search(
                str(query or "").strip(),
                top_k=bounded_limit,
                filters=filters,
                mode=requested_mode,
            )
        except Exception as exc:
            matches = search_catalog_lexical(
                tree,
                query=query,
                limit=bounded_limit,
                object_kind=object_kind,
            )
            return {
                "matches": matches,
                "requested_search_mode": requested_mode,
                "effective_search_mode": "lexical",
                "match_sources": ["lexical"] if matches else [],
                "backend": "in_memory_lexical",
                "vector_backend": self.vector_backend,
                "degraded_reason": f"catalog_index_error:{type(exc).__name__}",
            }

        all_sources: set[str] = set()
        matches: list[dict[str, Any]] = []
        for hit in hits:
            sources = {
                key[: -len("_rank")]
                for key in hit.sub
                if key.endswith("_rank") and key != "recency_rank"
            }
            all_sources.update(sources)
            matches.append(
                {
                    **dict(hit.metadata),
                    "score": hit.score,
                    "match_sources": sorted(sources),
                }
            )
        return {
            "matches": matches,
            "requested_search_mode": requested_mode,
            "effective_search_mode": _result_mode(all_sources),
            "match_sources": sorted(all_sources),
            "backend": "shared_hybrid_index",
            "vector_backend": self.vector_backend,
            "vector_path": str(self.vector_path) if self.vector_path else "",
        }


__all__ = [
    "DEFAULT_SCHEMA_SEARCH_LIMIT",
    "DEFAULT_SCHEMA_VECTOR_BACKEND",
    "MAX_SCHEMA_SEARCH_LIMIT",
    "SCHEMA_SEARCH_MODES",
    "SCHEMA_VECTOR_BACKENDS",
    "SchemaCatalogSearchIndex",
    "catalog_search_documents",
    "normalize_schema_vector_backend",
    "schema_search_embedding_profile",
    "schema_search_index_path",
    "schema_search_vector_path",
]
