# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Persistent hybrid search over a named-service capability catalog."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
SCHEMA_GENERATIONS_TO_RETAIN = 2
_GENERATION_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
_TIMESTAMPED_GENERATION_RE = re.compile(
    r"^capabilities\.(?P<timestamp>\d{8}T\d{12}Z)\."
    r"(?P<generation_hash>[0-9a-f]{16})(?:\..+)$"
)
_LEGACY_GENERATION_RE = re.compile(
    r"^capabilities\.(?P<generation_hash>[0-9a-f]{16})(?:\..+)$"
)

LOGGER = logging.getLogger("kdcube.sdk.named_services.schema_search")


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
    generation_timestamp: str = "",
    generation_hash: str = "",
) -> Path:
    selected_hash = str(generation_hash or "").strip().lower()
    if selected_hash and not re.fullmatch(r"[0-9a-f]{16}", selected_hash):
        raise ValueError(
            "schema index generation_hash must be 16 lowercase hex characters"
        )
    if not selected_hash and embedding_profile:
        selected_hash = hashlib.sha256(embedding_profile.encode("utf-8")).hexdigest()[
            :16
        ]
    selected_timestamp = str(generation_timestamp or "").strip()
    if selected_timestamp:
        try:
            datetime.strptime(selected_timestamp, _GENERATION_TIMESTAMP_FORMAT)
        except ValueError as exc:
            raise ValueError(
                "schema index generation_timestamp must use YYYYMMDDTHHMMSSffffffZ"
            ) from exc
    suffix = "".join(f".{part}" for part in (selected_timestamp, selected_hash) if part)
    return (
        Path(storage_root)
        / ".named-service-schema"
        / _safe(provider_id)
        / f"capabilities{suffix}.sqlite"
    )


def _generation_hash(*, embedding_profile: str, catalog_signature: str) -> str:
    identity = f"{embedding_profile}\n{catalog_signature}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _generation_marker_path(db_path: Path) -> Path:
    return db_path.with_suffix(".generation")


def _generation_lock_path(base_db_path: Path) -> Path:
    return base_db_path.parent / "capabilities.generations.lock"


@dataclass(frozen=True)
class _SchemaGeneration:
    db_path: Path
    timestamp: str
    generation_hash: str
    legacy: bool = False

    @property
    def family_prefix(self) -> str:
        return self.db_path.name[: -len(".sqlite")]


def _generation_from_path(path: Path) -> _SchemaGeneration | None:
    match = _TIMESTAMPED_GENERATION_RE.match(path.name)
    if match:
        prefix = (
            f"capabilities.{match.group('timestamp')}."
            f"{match.group('generation_hash')}"
        )
        return _SchemaGeneration(
            db_path=path.parent / f"{prefix}.sqlite",
            timestamp=match.group("timestamp"),
            generation_hash=match.group("generation_hash"),
        )
    legacy_match = _LEGACY_GENERATION_RE.match(path.name)
    if legacy_match:
        prefix = f"capabilities.{legacy_match.group('generation_hash')}"
        return _SchemaGeneration(
            db_path=path.parent / f"{prefix}.sqlite",
            timestamp="",
            generation_hash=legacy_match.group("generation_hash"),
            legacy=True,
        )
    if path.name.startswith("capabilities.sqlite"):
        return _SchemaGeneration(
            db_path=path.parent / "capabilities.sqlite",
            timestamp="",
            generation_hash="",
            legacy=True,
        )
    return None


def _discover_generations(directory: Path) -> list[_SchemaGeneration]:
    generations: dict[str, _SchemaGeneration] = {}
    if not directory.exists():
        return []
    for path in directory.glob("capabilities*"):
        generation = _generation_from_path(path)
        if generation is not None:
            generations[generation.family_prefix] = generation
    return list(generations.values())


def _generation_sort_key(generation: _SchemaGeneration) -> tuple[int, int]:
    # Timestamped generations were introduced after legacy files and therefore
    # always sort after them. Their UTC filename form sorts chronologically.
    if not generation.legacy:
        return (1, int(generation.timestamp.replace("T", "").rstrip("Z")))
    modified_ns = max(
        (path.stat().st_mtime_ns for path in _generation_family_files(generation)),
        default=0,
    )
    return (0, modified_ns)


def _generation_family_files(generation: _SchemaGeneration) -> list[Path]:
    if not generation.generation_hash:
        candidates = (
            generation.db_path,
            generation.db_path.with_suffix(".signature"),
            generation.db_path.with_suffix(".faiss"),
            generation.db_path.with_name(generation.db_path.name + ".lock"),
        )
        return sorted(path for path in candidates if path.is_file())
    return sorted(
        path
        for path in generation.db_path.parent.glob(f"{generation.family_prefix}.*")
        if path.is_file()
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
        managed_generations: bool = False,
    ) -> None:
        embed_texts = getattr(model_service, "embed_texts", None)
        if not callable(embed_texts):
            raise ValueError("schema catalog hybrid search requires model_service.embed_texts")
        self.base_db_path = Path(db_path)
        self.db_path = self.base_db_path
        self.signature_path = self.db_path.with_suffix(".signature")
        self.model_service = model_service
        self.dim = max(1, int(dim or 1536))
        self.min_semantic_score = float(min_semantic_score)
        self.vector_backend = normalize_schema_vector_backend(vector_backend)
        self.vector_path: Path | None = None
        self.index: HybridIndex | None = None
        self.managed_generations = bool(managed_generations)
        self.generation_hash = ""
        self.generation_timestamp = ""
        self._last_pruned_generation_hash = ""
        if not self.managed_generations:
            self._bind_index(self.base_db_path)

    def _bind_index(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.signature_path = self.db_path.with_suffix(".signature")
        self.vector_backend, vector_store = _schema_vector_store(
            self.db_path,
            backend=self.vector_backend,
        )
        self.vector_path = (
            schema_search_vector_path(self.db_path)
            if self.vector_backend == "faiss-local"
            else None
        )
        self.index = HybridIndex(
            IndexConfig(
                db_path=self.db_path,
                embed_fn=self.model_service.embed_texts,
                model_service=self.model_service,
                dim=self.dim,
                vector_store=vector_store,
                weights=FusionWeights(lexical=1.0, semantic=1.0, recency=0.0),
                min_semantic_score=self.min_semantic_score,
            )
        )

    def _index(self) -> HybridIndex:
        if self.index is None:
            raise RuntimeError("schema catalog generation has not been bound")
        return self.index

    async def _resolve_generation(self, generation_hash: str) -> _SchemaGeneration:
        self.base_db_path.parent.mkdir(parents=True, exist_ok=True)
        async with observed_file_lock_async(
            lock_path=_generation_lock_path(self.base_db_path),
            resource_id=f"named-services.schema:{self.base_db_path.parent.name}",
            operation="named-services.schema.generation.allocate",
            wait_seconds=30,
        ):
            matching = [
                generation
                for generation in _discover_generations(self.base_db_path.parent)
                if not generation.legacy
                and generation.generation_hash == generation_hash
            ]
            if matching:
                return max(matching, key=_generation_sort_key)

            timestamp = datetime.now(timezone.utc).strftime(
                _GENERATION_TIMESTAMP_FORMAT
            )
            db_path = self.base_db_path.with_name(
                f"capabilities.{timestamp}.{generation_hash}.sqlite"
            )
            marker_path = _generation_marker_path(db_path)
            _atomic_write_text(
                marker_path,
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "generation_hash": generation_hash,
                        "db_file": db_path.name,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )
            return _SchemaGeneration(
                db_path=db_path,
                timestamp=timestamp,
                generation_hash=generation_hash,
            )

    async def _bind_managed_generation(self, generation_hash: str) -> None:
        if (
            self.index is not None
            and self.generation_hash == generation_hash
            and self.db_path.exists()
        ):
            return
        generation = await self._resolve_generation(generation_hash)
        if self.index is None or self.db_path != generation.db_path:
            self._bind_index(generation.db_path)
        self.generation_hash = generation.generation_hash
        self.generation_timestamp = generation.timestamp

    async def _ensure_vector_store_built(self) -> None:
        # HybridIndex's persisted build version normally avoids rebuilding a
        # file-backed vector store. If the derived FAISS file was removed while
        # SQLite survived, invalidate that version and reconstruct it from the
        # cached vectors instead of silently serving lexical-only results.
        if self.vector_path is not None and not self.vector_path.exists():
            await self._index().rebuild()
            return
        await self._index().ensure_built()

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
        if self.managed_generations:
            await self._bind_managed_generation(
                _generation_hash(
                    embedding_profile=embedding_profile,
                    catalog_signature=signature,
                )
            )
        index = self._index()
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
            if stored_signature == signature and set(index.ids()) == expected:
                await self._ensure_vector_store_built()
                return {
                    "ok": True,
                    "indexed": len(docs),
                    "updated": False,
                    "signature": signature,
                    "db_path": str(self.db_path),
                    "vector_backend": self.vector_backend,
                    "vector_path": str(self.vector_path) if self.vector_path else "",
                    "generation_hash": self.generation_hash,
                    "generation_timestamp": self.generation_timestamp,
                }
            stale = set(index.ids()) - expected
            if stale:
                await index.delete(stale)
            if docs:
                await index.upsert(docs)
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
            "generation_hash": self.generation_hash,
            "generation_timestamp": self.generation_timestamp,
        }

    async def prune_stale_generations(self) -> dict[str, Any]:
        """Keep the active managed generation and its immediate predecessor."""

        if not self.managed_generations or not self.generation_hash:
            return {"pruned": False, "reason": "managed_generations_disabled"}
        active = _generation_from_path(self.db_path)
        if active is None or active.legacy:
            return {"pruned": False, "reason": "active_generation_not_timestamped"}

        removed: list[str] = []
        skipped: list[str] = []
        async with observed_file_lock_async(
            lock_path=_generation_lock_path(self.base_db_path),
            resource_id=f"named-services.schema:{self.base_db_path.parent.name}",
            operation="named-services.schema.generation.cleanup",
            wait_seconds=30,
        ):
            generations = sorted(
                _discover_generations(self.base_db_path.parent),
                key=_generation_sort_key,
                reverse=True,
            )
            if not generations:
                return {"pruned": False, "reason": "no_generations"}
            if generations[0].family_prefix != active.family_prefix:
                return {
                    "pruned": False,
                    "reason": "active_generation_is_not_newest",
                    "active": active.family_prefix,
                    "newest": generations[0].family_prefix,
                }

            retained = [generations[0]]
            for candidate in generations[1:]:
                if len(retained) >= SCHEMA_GENERATIONS_TO_RETAIN:
                    break
                if candidate.generation_hash != active.generation_hash:
                    retained.append(candidate)
            retained_prefixes = {item.family_prefix for item in retained}
            stale_generations = [
                generation
                for generation in generations
                if generation.family_prefix not in retained_prefixes
            ]
            for stale in stale_generations:
                index_lock_path = stale.db_path.with_name(stale.db_path.name + ".lock")
                try:
                    async with observed_file_lock_async(
                        lock_path=index_lock_path,
                        resource_id=(
                            f"named-services.schema:{self.base_db_path.parent.name}:"
                            f"{stale.family_prefix}"
                        ),
                        operation="named-services.schema.generation.delete",
                        wait_seconds=0.25,
                    ):
                        for path in _generation_family_files(stale):
                            if path != index_lock_path:
                                path.unlink(missing_ok=True)
                    index_lock_path.unlink(missing_ok=True)
                    removed.append(stale.family_prefix)
                except (OSError, TimeoutError) as exc:
                    skipped.append(stale.family_prefix)
                    LOGGER.warning(
                        "Could not prune named-service schema index generation: "
                        "namespace=%s generation=%s error=%s",
                        self.base_db_path.parent.name,
                        stale.family_prefix,
                        type(exc).__name__,
                    )

        if removed:
            LOGGER.info(
                "Pruned named-service schema index generations: namespace=%s "
                "active=%s removed=%s skipped=%s",
                self.base_db_path.parent.name,
                active.family_prefix,
                removed,
                skipped,
            )
        return {
            "pruned": bool(removed),
            "active": active.family_prefix,
            "retained": [item.family_prefix for item in retained],
            "removed": removed,
            "skipped": skipped,
        }

    async def prune_stale_generations_once(self) -> dict[str, Any]:
        """Run retention once per generation in this provider process."""

        if self.generation_hash == self._last_pruned_generation_hash:
            return {"pruned": False, "reason": "already_pruned_by_process"}
        result = await self.prune_stale_generations()
        if not result.get("skipped"):
            self._last_pruned_generation_hash = self.generation_hash
        return result

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
        await self.prune_stale_generations_once()
        filters = {"object_kind": object_kind} if object_kind else None
        try:
            hits = await self._index().search(
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
