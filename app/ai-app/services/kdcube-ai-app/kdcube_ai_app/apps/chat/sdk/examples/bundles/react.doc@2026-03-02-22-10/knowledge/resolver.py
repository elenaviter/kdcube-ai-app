# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

"""Knowledge-space resolver for the react.doc bundle (ks: read + search)."""

import json
import pathlib
import importlib.util
import sys
from typing import Any, Dict, List, Optional

try:
    from .index_builder import prepare_knowledge_space as _prepare
except Exception:
    # When this module is loaded via spec_from_file_location (no package),
    # relative imports won't work. Load index_builder by file path instead.
    _module_name = "_kdcube_react_doc_index_builder"
    if _module_name in sys.modules:
        _mod = sys.modules[_module_name]
    else:
        _path = pathlib.Path(__file__).resolve().parent / "index_builder.py"
        _spec = importlib.util.spec_from_file_location(_module_name, str(_path))
        if not _spec or not _spec.loader:
            raise ImportError(f"Cannot load index_builder: {_path}")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_module_name] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore
    _prepare = getattr(_mod, "prepare_knowledge_space")

KNOWLEDGE_ROOT: Optional[pathlib.Path] = None


def prepare_knowledge_space(
    *,
    bundle_root: pathlib.Path,
    knowledge_root: pathlib.Path,
    docs_root: Optional[pathlib.Path] = None,
    src_root: Optional[pathlib.Path] = None,
    deploy_root: Optional[pathlib.Path] = None,
    validate_refs: bool = True,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    global KNOWLEDGE_ROOT
    KNOWLEDGE_ROOT = knowledge_root
    return _prepare(
        bundle_root=bundle_root,
        knowledge_root=knowledge_root,
        docs_root=docs_root,
        src_root=src_root,
        deploy_root=deploy_root,
        validate_refs=validate_refs,
        logger=logger,
    )


def _load_index(root: pathlib.Path) -> Dict[str, Any]:
    index_path = root / "index.json"
    if not index_path.exists():
        return {}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tokenize(text: str) -> List[str]:
    parts = []
    for raw in text.lower().strip().split():
        raw = raw.strip(".,;:!?()[]{}<>\"'`")
        if raw:
            parts.append(raw)
    return parts


def search_knowledge(
    *,
    query: str,
    root: str = "",
    max_hits: int = 20,
    keywords: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Metadata search over index.json (title/summary/tags/keywords/path)."""
    if not query:
        return []
    if not KNOWLEDGE_ROOT:
        return []
    index = _load_index(KNOWLEDGE_ROOT)
    items = list(index.get("items") or [])
    if not items:
        return []

    # Compute root filter (ks:docs, ks:src, ks:deploy, or custom).
    root_rel = ""
    if root:
        root = root.strip()
        if root.startswith("ks:"):
            root_rel = root[3:].lstrip("/").rstrip("/")
        elif KNOWLEDGE_ROOT:
            try:
                base = pathlib.Path(root).resolve()
                root_rel = base.relative_to(KNOWLEDGE_ROOT.resolve()).as_posix().rstrip("/")
            except Exception:
                root_rel = root.strip("/").rstrip("/")
    elif KNOWLEDGE_ROOT:
        try:
            root_rel = ""
        except Exception:
            root_rel = ""

    q = query.lower().strip()
    terms = _tokenize(q)
    extra_terms: List[str] = []
    if keywords:
        joined = " ".join([str(k) for k in keywords if str(k).strip()])
        extra_terms = _tokenize(joined)
    scored: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if root_rel and not path.startswith(f"ks:{root_rel}"):
            continue
        title = str(item.get("title") or "")
        summary = str(item.get("summary") or "")
        tags = [str(t).lower() for t in (item.get("tags") or []) if str(t).strip()]
        keywords = [str(t).lower() for t in (item.get("keywords") or []) if str(t).strip()]

        title_l = title.lower()
        summary_l = summary.lower()
        path_l = path.lower()
        tag_set = set(tags + keywords)

        # Phrase-level scoring
        score = 0.0
        matched = False
        if q and q in title_l:
            score += 3.0
            matched = True
        if q and q in summary_l:
            score += 1.5
            matched = True
        if q and q in path_l:
            score += 0.7
            matched = True

        # Term-level scoring
        for t in terms + extra_terms:
            if t in title_l:
                score += 1.0
                matched = True
            if t in tag_set:
                score += 0.8
                matched = True
            if t in summary_l:
                score += 0.4
                matched = True
            if t in path_l:
                score += 0.2
                matched = True

        if not matched:
            continue

        scored.append({
            "path": path,
            "title": title,
            "score": score,
        })

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored[: max_hits or 20]


def read_knowledge(*, path: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Resolve and read a knowledge-space path (ks:<relpath>).
    Returns dict with keys: text, base64, mime, physical_path, missing.
    """
    if not path or not isinstance(path, str):
        return {"missing": True}
    if not KNOWLEDGE_ROOT:
        return {"missing": True}
    raw = path.strip()
    if raw.startswith("ks:"):
        rel = raw[len("ks:"):].lstrip("/")
    else:
        rel = raw.lstrip("/")
    if not rel:
        return {"missing": True}
    try:
        from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.solution_workspace import (
            _safe_relpath,
            _guess_mime_from_path,
            _read_local_file,
        )
    except Exception:
        return {"missing": True}
    if not _safe_relpath(rel):
        return {"missing": True}
    abs_path = (KNOWLEDGE_ROOT / rel).resolve()
    if not abs_path.exists() or not abs_path.is_file():
        return {"missing": True}
    mime = _guess_mime_from_path(str(abs_path))
    text, base64 = _read_local_file(abs_path, mime)
    return {
        "text": text,
        "base64": base64,
        "mime": mime,
        "physical_path": str(abs_path),
    }
