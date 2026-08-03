# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Recursive capability catalogs for named-service provider schemas.

The catalog indexes provider *capabilities*, not provider objects. A provider
may arrange its stable operation ids under domain-owned nodes of arbitrary
depth. Clients browse one node at a time or search the flattened operation
entries, then request one exact operation contract from ``object.schema``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


SCHEMA_CATALOG_CONTRACT = "kdcube.named-service.schema-catalog.v1"
SCHEMA_CATALOG_ROOT_PATH = "/"


class SchemaCatalogError(ValueError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        raw: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = ()
    return [text for text in (str(item or "").strip() for item in raw) if text]


def _humanize(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("object.action:"):
        text = text.split(":", 1)[1]
    elif text.startswith("object."):
        text = text.split(".", 1)[1]
    text = re.sub(r"[_.:-]+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else "Capability"


def normalize_schema_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "/":
        return SCHEMA_CATALOG_ROOT_PATH
    segments = [segment.strip() for segment in text.strip("/").split("/")]
    if any(not segment for segment in segments):
        raise SchemaCatalogError(
            code="named_service_schema_path_invalid",
            message="schema_path contains an empty catalog segment.",
            details={"schema_path": text},
        )
    return "/" + "/".join(segments)


def _child_specs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        children: list[dict[str, Any]] = []
        for child_id, raw_child in value.items():
            child = _mapping(raw_child)
            child.setdefault("id", str(child_id))
            children.append(child)
        return children
    if isinstance(value, (list, tuple)):
        return [_mapping(item) for item in value if isinstance(item, Mapping)]
    return []


def _operation_specs(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        entries: list[dict[str, Any]] = []
        for operation, raw in value.items():
            entry = _mapping(raw)
            entry.setdefault("schema_operation", str(operation))
            entries.append(entry)
        return entries
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _normalize_operation_entry(
    raw: Any,
    *,
    inherited_kind: str,
    path: str,
    breadcrumbs: list[str],
    operations_by_kind: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if isinstance(raw, str):
        entry_spec: dict[str, Any] = {"schema_operation": raw}
    elif isinstance(raw, Mapping):
        entry_spec = dict(raw)
    else:
        raise SchemaCatalogError(
            code="named_service_schema_catalog_invalid",
            message="Catalog operations must be operation ids or JSON objects.",
            details={"catalog_path": path},
        )

    object_kind = str(entry_spec.get("object_kind") or inherited_kind or "").strip()
    schema_operation = str(
        entry_spec.get("schema_operation") or entry_spec.get("operation") or ""
    ).strip()
    if not object_kind or not schema_operation:
        raise SchemaCatalogError(
            code="named_service_schema_catalog_invalid",
            message=(
                "Every catalog operation needs schema_operation and an "
                "object_kind, directly or inherited from its catalog node."
            ),
            details={
                "catalog_path": path,
                "object_kind": object_kind,
                "schema_operation": schema_operation,
            },
        )
    available = operations_by_kind.get(object_kind)
    if available is None or schema_operation not in available:
        raise SchemaCatalogError(
            code="named_service_schema_catalog_invalid",
            message="Catalog operation does not match the provider projection.",
            details={
                "catalog_path": path,
                "object_kind": object_kind,
                "schema_operation": schema_operation,
                "available_operations": list(available or {}),
            },
        )

    base = _mapping(available[schema_operation])
    description = str(
        entry_spec.get("description") or base.get("description") or ""
    ).strip()
    return {
        "object_kind": object_kind,
        "schema_operation": schema_operation,
        "label": str(entry_spec.get("label") or _humanize(schema_operation)).strip(),
        "description": description,
        "keywords": _strings(entry_spec.get("keywords")),
        "catalog_path": path,
        "catalog_breadcrumbs": list(breadcrumbs),
    }


def _normalize_node(
    raw: Mapping[str, Any],
    *,
    parent_path: str,
    inherited_kind: str,
    breadcrumbs: list[str],
    operations_by_kind: Mapping[str, Mapping[str, Mapping[str, Any]]],
    root: bool = False,
) -> dict[str, Any]:
    node_id = str(raw.get("id") or ("root" if root else "")).strip()
    if not node_id:
        raise SchemaCatalogError(
            code="named_service_schema_catalog_invalid",
            message="Every non-root catalog node requires a stable id.",
            details={"parent_path": parent_path},
        )
    if "/" in node_id:
        raise SchemaCatalogError(
            code="named_service_schema_catalog_invalid",
            message="Catalog node ids cannot contain '/'.",
            details={"node_id": node_id, "parent_path": parent_path},
        )
    path = (
        SCHEMA_CATALOG_ROOT_PATH
        if root
        else normalize_schema_path(parent_path.rstrip("/") + "/" + node_id)
    )
    label = str(raw.get("label") or _humanize(node_id)).strip()
    node_breadcrumbs = [*breadcrumbs, label] if not root else [label]
    object_kind = str(raw.get("object_kind") or inherited_kind or "").strip()

    child_ids: set[str] = set()
    children: list[dict[str, Any]] = []
    for child_spec in _child_specs(raw.get("children")):
        child_id = str(child_spec.get("id") or "").strip()
        if child_id in child_ids:
            raise SchemaCatalogError(
                code="named_service_schema_catalog_invalid",
                message="Sibling catalog node ids must be unique.",
                details={"catalog_path": path, "duplicate_node_id": child_id},
            )
        child_ids.add(child_id)
        children.append(
            _normalize_node(
                child_spec,
                parent_path=path,
                inherited_kind=object_kind,
                breadcrumbs=node_breadcrumbs,
                operations_by_kind=operations_by_kind,
            )
        )

    operations = [
        _normalize_operation_entry(
            raw_operation,
            inherited_kind=object_kind,
            path=path,
            breadcrumbs=node_breadcrumbs,
            operations_by_kind=operations_by_kind,
        )
        for raw_operation in _operation_specs(raw.get("operations"))
    ]
    return {
        "id": node_id,
        "path": path,
        "label": label,
        "description": str(raw.get("description") or "").strip(),
        "keywords": _strings(raw.get("keywords")),
        "object_kind": object_kind,
        "children": children,
        "operations": operations,
    }


def _auto_catalog(
    *,
    namespace: str,
    operations_by_kind: Mapping[str, Mapping[str, Mapping[str, Any]]],
    kind_descriptions: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "id": namespace or "root",
        "label": _humanize(namespace or "Capabilities"),
        "description": "Browse provider capabilities by object kind.",
        "children": [
            {
                "id": re.sub(r"[^A-Za-z0-9_.-]", "-", kind).strip("-") or "kind",
                "label": _humanize(kind),
                "description": str(kind_descriptions.get(kind) or ""),
                "object_kind": kind,
                "operations": list(operations),
            }
            for kind, operations in operations_by_kind.items()
        ],
    }


def _iter_nodes(node: Mapping[str, Any]):
    yield node
    for child in node.get("children") or []:
        if isinstance(child, Mapping):
            yield from _iter_nodes(child)


def catalog_operation_entries(tree: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for node in _iter_nodes(tree):
        node_terms = [
            str(node.get("label") or ""),
            str(node.get("description") or ""),
            *_strings(node.get("keywords")),
        ]
        for raw_entry in node.get("operations") or []:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            entry["catalog_terms"] = [term for term in node_terms if term]
            entries.append(entry)
    return entries


def build_recursive_catalog(
    *,
    namespace: str,
    projection_index: Mapping[str, Any],
    operations_by_kind: Mapping[str, Mapping[str, Mapping[str, Any]]],
    kind_descriptions: Mapping[str, str],
) -> dict[str, Any]:
    explicit = projection_index.get("catalog")
    raw_root = (
        _mapping(explicit)
        if isinstance(explicit, Mapping)
        else _auto_catalog(
            namespace=namespace,
            operations_by_kind=operations_by_kind,
            kind_descriptions=kind_descriptions,
        )
    )
    raw_root.setdefault("id", namespace or "root")
    tree = _normalize_node(
        raw_root,
        parent_path=SCHEMA_CATALOG_ROOT_PATH,
        inherited_kind="",
        breadcrumbs=[],
        operations_by_kind=operations_by_kind,
        root=True,
    )

    if isinstance(explicit, Mapping):
        available = {
            (kind, operation)
            for kind, operations in operations_by_kind.items()
            for operation in operations
        }
        cataloged = {
            (entry["object_kind"], entry["schema_operation"])
            for entry in catalog_operation_entries(tree)
        }
        missing = sorted(available - cataloged)
        if missing:
            raise SchemaCatalogError(
                code="named_service_schema_catalog_invalid",
                message="Explicit schema catalog does not place every projected operation.",
                details={
                    "missing_operations": [
                        {"object_kind": kind, "schema_operation": operation}
                        for kind, operation in missing
                    ]
                },
            )
    return tree


def _descendant_operation_count(node: Mapping[str, Any]) -> int:
    own = len(node.get("operations") or [])
    return own + sum(
        _descendant_operation_count(child)
        for child in node.get("children") or []
        if isinstance(child, Mapping)
    )


def catalog_node(tree: Mapping[str, Any], path: Any = "/") -> dict[str, Any]:
    requested = normalize_schema_path(path)
    selected: Mapping[str, Any] | None = None
    for node in _iter_nodes(tree):
        if str(node.get("path") or "") == requested:
            selected = node
            break
    if selected is None:
        raise SchemaCatalogError(
            code="named_service_schema_path_unknown",
            message=f"Unknown schema catalog path {requested!r}.",
            details={"schema_path": requested},
        )
    return {
        "contract": SCHEMA_CATALOG_CONTRACT,
        "path": requested,
        "label": str(selected.get("label") or ""),
        "description": str(selected.get("description") or ""),
        "keywords": _strings(selected.get("keywords")),
        "children": [
            {
                "id": str(child.get("id") or ""),
                "path": str(child.get("path") or ""),
                "label": str(child.get("label") or ""),
                "description": str(child.get("description") or ""),
                "child_count": len(child.get("children") or []),
                "operation_count": _descendant_operation_count(child),
            }
            for child in selected.get("children") or []
            if isinstance(child, Mapping)
        ],
        "operations": [
            {
                key: value
                for key, value in dict(operation).items()
                if key != "catalog_terms"
            }
            for operation in selected.get("operations") or []
            if isinstance(operation, Mapping)
        ],
    }


_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _tokens(value: Any, *, query: bool = False) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", str(value or "").casefold())
    if not query:
        return tokens
    return [token for token in tokens if token not in _QUERY_STOP_WORDS]


def _lexical_score(
    entry: Mapping[str, Any],
    query: str,
    *,
    require_all: bool,
) -> tuple[float, set[str]]:
    terms = _tokens(query, query=True)
    if not terms:
        return 0.0, set()
    weighted_fields = (
        ("label", str(entry.get("label") or ""), 6.0),
        ("keywords", " ".join(_strings(entry.get("keywords"))), 5.0),
        ("catalog", " ".join(_strings(entry.get("catalog_terms"))), 4.0),
        ("path", str(entry.get("catalog_path") or ""), 3.0),
        ("operation", str(entry.get("schema_operation") or ""), 3.0),
        ("kind", str(entry.get("object_kind") or ""), 2.0),
        ("description", str(entry.get("description") or ""), 1.0),
    )
    matched_terms: set[str] = set()
    score = 0.0
    matched_fields: set[str] = set()
    for field, text, weight in weighted_fields:
        field_tokens = set(_tokens(text))
        matches = {term for term in terms if any(token.startswith(term) for token in field_tokens)}
        if matches:
            matched_terms.update(matches)
            matched_fields.add(field)
            score += weight * len(matches)
        if query.casefold() in text.casefold():
            score += weight * 1.5
    if not matched_terms or (require_all and len(matched_terms) != len(set(terms))):
        return 0.0, set()
    coverage = len(matched_terms) / len(set(terms))
    return score * (0.5 + coverage), matched_fields


def search_catalog_lexical(
    tree: Mapping[str, Any],
    *,
    query: str,
    limit: int,
    object_kind: str = "",
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in catalog_operation_entries(tree)
        if not object_kind or str(entry.get("object_kind") or "") == object_kind
    ]

    def _score(require_all: bool) -> list[tuple[float, dict[str, Any], set[str]]]:
        scored_entries: list[tuple[float, dict[str, Any], set[str]]] = []
        for entry in entries:
            score, fields = _lexical_score(
                entry,
                query,
                require_all=require_all,
            )
            if score > 0:
                scored_entries.append((score, entry, fields))
        return scored_entries

    # Match the shared FTS behavior: preserve precision when every meaningful
    # term lands, then widen to any-term matches only when the strict pass is
    # empty. This keeps lexical fallback useful for natural-language queries.
    scored = _score(require_all=True) or _score(require_all=False)
    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("catalog_path") or ""),
            str(item[1].get("schema_operation") or ""),
        )
    )
    return [
        {
            **{key: value for key, value in entry.items() if key != "catalog_terms"},
            "score": score,
            "match_sources": ["lexical"],
            "matched_fields": sorted(fields),
        }
        for score, entry, fields in scored[: max(1, int(limit or 10))]
    ]


__all__ = [
    "SCHEMA_CATALOG_CONTRACT",
    "SCHEMA_CATALOG_ROOT_PATH",
    "SchemaCatalogError",
    "build_recursive_catalog",
    "catalog_node",
    "catalog_operation_entries",
    "normalize_schema_path",
    "search_catalog_lexical",
]
