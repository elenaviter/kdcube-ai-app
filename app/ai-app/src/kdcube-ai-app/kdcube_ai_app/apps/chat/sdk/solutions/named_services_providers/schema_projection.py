# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping

from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.schema_catalog import (
    SchemaCatalogError,
    build_recursive_catalog,
    catalog_node,
    catalog_operation_entries,
    normalize_schema_path,
    search_catalog_lexical,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    NamedServiceRequest,
    NamedServiceResponse,
)


SCHEMA_PROJECTION_CONTRACT = "kdcube.named-service.schema-projection.v1"
SCHEMA_VIEW_CATALOG = "catalog"
SCHEMA_VIEW_KIND = "kind"
SCHEMA_VIEW_OPERATION = "operation"
SCHEMA_VIEW_SEARCH = "search"
SCHEMA_VIEW_FULL = "full"
SCHEMA_VIEWS = {
    SCHEMA_VIEW_CATALOG,
    SCHEMA_VIEW_KIND,
    SCHEMA_VIEW_OPERATION,
    SCHEMA_VIEW_SEARCH,
    SCHEMA_VIEW_FULL,
}

_DEFAULT_GLOBAL_SECTIONS = (
    "account_selection",
    "consent_errors",
    "grant_hints",
    "connected_account_claims",
)
_GENERIC_OPERATION_ALIASES = {
    "list": "object.list",
    "search": "object.search",
    "get": "object.get",
    "upsert": "object.upsert",
    "create": "object.upsert",
    "update": "object.upsert",
    "delete": "object.delete",
    "host_file": "object.host_file",
}
_GENERIC_OPERATION_DESCRIPTIONS = {
    "object.list": "Browse objects of this kind with bounded pagination.",
    "object.search": "Find objects of this kind using provider-declared search.",
    "object.get": "Read one object of this kind by canonical ref.",
    "object.upsert": "Create or update an object of this kind.",
    "object.delete": "Delete or archive an object of this kind when allowed.",
    "object.host_file": "Host a runtime file into this provider realm.",
}


class SchemaProjectionError(ValueError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status: int = 400,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = dict(details or {})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple, set)):
        values = tuple(value)
    else:
        values = ()
    return tuple(
        text
        for text in (str(item or "").strip() for item in values)
        if text
    )


def _kind_specs(index: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(index.get("kinds"))


def _operation_specs(kind_spec: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(kind_spec.get("operations"))


def _action_operation(action: str) -> str:
    return f"object.action:{action}"


def _available_operations(kind_spec: Mapping[str, Any]) -> tuple[str, ...]:
    operations = list(_operation_specs(kind_spec))
    operations.extend(_action_operation(action) for action in _strings(kind_spec.get("actions")))
    return tuple(dict.fromkeys(operations))


def _validate_projection_index(
    schema: Mapping[str, Any],
    index: Mapping[str, Any],
) -> None:
    schema_kinds = _mapping(schema.get("object_kinds"))
    schema_refs = _mapping(schema.get("refs"))
    schema_selectors = _mapping(schema.get("selectors"))
    schema_actions = _mapping(schema.get("actions"))
    kind_specs = _kind_specs(index)

    unknown_index_sections = sorted(
        {
            section
            for key in ("catalog_sections", "global_sections")
            for section in _strings(index.get(key))
            if section not in schema
        }
    )
    if unknown_index_sections:
        raise SchemaProjectionError(
            code="named_service_schema_projection_invalid",
            message="Provider schema projection references unknown schema sections.",
            status=500,
            details={"unknown_sections": unknown_index_sections},
        )

    missing_kinds = sorted(set(schema_kinds) - set(kind_specs))
    unknown_kinds = sorted(set(kind_specs) - set(schema_kinds))
    if missing_kinds or unknown_kinds:
        raise SchemaProjectionError(
            code="named_service_schema_projection_invalid",
            message="Provider schema projection does not match its object kinds.",
            status=500,
            details={
                "missing_projection_kinds": missing_kinds,
                "unknown_projection_kinds": unknown_kinds,
            },
        )

    assigned_actions: set[str] = set()
    for kind, raw_spec in kind_specs.items():
        spec = _mapping(raw_spec)
        unknown_operations = sorted(
            set(_operation_specs(spec)) - set(_GENERIC_OPERATION_DESCRIPTIONS)
        )
        unknown_refs = sorted(set(_strings(spec.get("refs"))) - set(schema_refs))
        unknown_selectors = sorted(
            set(_strings(spec.get("selectors"))) - set(schema_selectors)
        )
        unknown_related_kinds = sorted(
            set(_strings(spec.get("related_kinds"))) - set(schema_kinds)
        )
        actions = set(_strings(spec.get("actions")))
        assigned_actions.update(actions)
        unknown_actions = sorted(actions - set(schema_actions))
        unknown_sections: set[str] = set()
        unknown_section_keys: dict[str, list[str]] = {}
        for raw_operation in _operation_specs(spec).values():
            operation_spec = _mapping(raw_operation)
            unknown_sections.update(
                section
                for section in _strings(operation_spec.get("sections"))
                if section not in schema
            )
            for section, raw_keys in _mapping(
                operation_spec.get("section_keys")
            ).items():
                section_value = schema.get(section)
                if not isinstance(section_value, Mapping):
                    unknown_section_keys[str(section)] = list(_strings(raw_keys))
                    continue
                missing = sorted(set(_strings(raw_keys)) - set(section_value))
                if missing:
                    unknown_section_keys[str(section)] = missing
        if (
            unknown_operations
            or unknown_refs
            or unknown_selectors
            or unknown_related_kinds
            or unknown_actions
            or unknown_sections
            or unknown_section_keys
        ):
            raise SchemaProjectionError(
                code="named_service_schema_projection_invalid",
                message=f"Provider schema projection for {kind!r} references unknown schema entries.",
                status=500,
                details={
                    "object_kind": kind,
                    "unknown_operations": unknown_operations,
                    "unknown_refs": unknown_refs,
                    "unknown_selectors": unknown_selectors,
                    "unknown_related_kinds": unknown_related_kinds,
                    "unknown_actions": unknown_actions,
                    "unknown_sections": sorted(unknown_sections),
                    "unknown_section_keys": unknown_section_keys,
                },
            )

    unassigned_actions = sorted(set(schema_actions) - assigned_actions)
    if unassigned_actions:
        raise SchemaProjectionError(
            code="named_service_schema_projection_invalid",
            message="Provider schema contains actions that are not assigned to an object kind.",
            status=500,
            details={"unassigned_actions": unassigned_actions},
        )


def _projection_metadata(
    *,
    view: str,
    kind_specs: Mapping[str, Any],
    object_kind: str = "",
    schema_operation: str = "",
    schema_path: str = "",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "contract": SCHEMA_PROJECTION_CONTRACT,
        "view": view,
        "available_object_kinds": list(kind_specs),
    }
    if object_kind:
        metadata["object_kind"] = object_kind
    if object_kind and view == SCHEMA_VIEW_KIND:
        metadata["available_operations"] = list(
            _available_operations(_mapping(kind_specs.get(object_kind)))
        )
    if schema_operation:
        metadata["schema_operation"] = schema_operation
    if schema_path:
        metadata["schema_path"] = schema_path
    return metadata


def _copy_sections(
    *,
    schema: Mapping[str, Any],
    target: dict[str, Any],
    sections: tuple[str, ...],
) -> None:
    for section in sections:
        if section in schema:
            target[section] = schema[section]


def _operation_contract_key(operation: str) -> str:
    if operation.startswith("object.action:"):
        return "object.action." + operation.split(":", 1)[1]
    return operation


def _copy_kind_globals(
    *,
    schema: Mapping[str, Any],
    index: Mapping[str, Any],
    target: dict[str, Any],
    operations: tuple[str, ...],
) -> None:
    sections = _strings(index.get("global_sections")) or _DEFAULT_GLOBAL_SECTIONS
    operation_keys = {_operation_contract_key(value) for value in operations}
    for section in sections:
        if section not in schema:
            continue
        value = schema[section]
        if section == "grant_hints" and isinstance(value, Mapping):
            target[section] = {
                key: item for key, item in value.items() if key in operation_keys
            }
            continue
        target[section] = value


def _operation_summary(
    *,
    schema: Mapping[str, Any],
    operation: str,
    operation_spec: Mapping[str, Any],
) -> dict[str, Any]:
    description = str(operation_spec.get("description") or "").strip()
    if not description:
        descriptions: list[str] = []
        for section in _strings(operation_spec.get("sections")):
            section_value = schema.get(section)
            if isinstance(section_value, Mapping):
                section_description = str(
                    section_value.get("description") or ""
                ).strip()
                if section_description:
                    descriptions.append(section_description)
        description = " ".join(descriptions) or _GENERIC_OPERATION_DESCRIPTIONS.get(
            operation, "Schema-declared operation."
        )
    return {"description": description}


def _action_summary(schema: Mapping[str, Any], action: str) -> dict[str, Any]:
    action_spec = _mapping(_mapping(schema.get("actions")).get(action))
    summary = {
        "description": str(action_spec.get("description") or "").strip()
        or "Schema-declared bounded action."
    }
    if action_spec.get("claim"):
        summary["claim"] = action_spec["claim"]
    return summary


def _kind_operation_summaries(
    *,
    schema: Mapping[str, Any],
    kind_spec: Mapping[str, Any],
) -> dict[str, Any]:
    summaries = {
        operation: _operation_summary(
            schema=schema,
            operation=operation,
            operation_spec=_mapping(operation_spec),
        )
        for operation, operation_spec in _operation_specs(kind_spec).items()
    }
    for action in _strings(kind_spec.get("actions")):
        summaries[_action_operation(action)] = _action_summary(schema, action)
    return summaries


def build_schema_tree(
    schema: Mapping[str, Any],
    index: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one provider-owned recursive capability catalog."""

    full_schema = _mapping(schema)
    projection_index = _mapping(index)
    _validate_projection_index(full_schema, projection_index)
    schema_kinds = _mapping(full_schema.get("object_kinds"))
    kind_specs = _kind_specs(projection_index)
    operations_by_kind = {
        kind: _kind_operation_summaries(
            schema=full_schema,
            kind_spec=_mapping(kind_spec),
        )
        for kind, kind_spec in kind_specs.items()
    }
    kind_descriptions = {
        kind: str(_mapping(schema_kinds.get(kind)).get("description") or "")
        for kind in kind_specs
    }
    try:
        return build_recursive_catalog(
            namespace=str(full_schema.get("namespace") or ""),
            projection_index=projection_index,
            operations_by_kind=operations_by_kind,
            kind_descriptions=kind_descriptions,
        )
    except SchemaCatalogError as exc:
        raise SchemaProjectionError(
            code=exc.code,
            message=exc.message,
            status=(
                500
                if exc.code == "named_service_schema_catalog_invalid"
                else 400
            ),
            details=exc.details,
        ) from exc


def build_schema_catalog(
    schema: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    schema_path: str = "/",
) -> dict[str, Any]:
    full_schema = _mapping(schema)
    projection_index = _mapping(index)
    _validate_projection_index(full_schema, projection_index)
    schema_kinds = _mapping(full_schema.get("object_kinds"))
    kind_specs = _kind_specs(projection_index)

    tree = build_schema_tree(full_schema, projection_index)
    entries = catalog_operation_entries(tree)
    catalog_paths_by_kind: dict[str, set[str]] = {
        kind: set() for kind in kind_specs
    }
    for entry in entries:
        kind = str(entry.get("object_kind") or "")
        if kind in catalog_paths_by_kind:
            catalog_paths_by_kind[kind].add(
                str(entry.get("catalog_path") or "/")
            )

    catalog: dict[str, Any] = {}
    for key in ("namespace", "schema_version"):
        if key in full_schema:
            catalog[key] = full_schema[key]
    catalog["object_kinds"] = {
        kind: {
            "description": str(_mapping(schema_kinds.get(kind)).get("description") or ""),
            "operation_count": len(_available_operations(_mapping(kind_spec))),
            "catalog_paths": sorted(catalog_paths_by_kind.get(kind) or []),
        }
        for kind, kind_spec in kind_specs.items()
    }
    try:
        catalog["catalog"] = catalog_node(tree, schema_path)
    except SchemaCatalogError as exc:
        raise SchemaProjectionError(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ) from exc
    _copy_sections(
        schema=full_schema,
        target=catalog,
        sections=_strings(projection_index.get("catalog_sections")),
    )
    catalog["schema_projection"] = _projection_metadata(
        view=SCHEMA_VIEW_CATALOG,
        kind_specs=kind_specs,
        schema_path=normalize_schema_path(schema_path),
    )
    return catalog


def build_schema_search(
    schema: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    query: str,
    search_mode: str = "hybrid",
    limit: int = 10,
    object_kind: str = "",
    search_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return compact operation candidates from the capability catalog."""

    full_schema = _mapping(schema)
    projection_index = _mapping(index)
    _validate_projection_index(full_schema, projection_index)
    kind_specs = _kind_specs(projection_index)
    clean_query = str(query or "").strip()
    if not clean_query:
        raise SchemaProjectionError(
            code="named_service_schema_query_required",
            message="Capability schema search requires a non-blank query.",
        )
    requested_mode = str(search_mode or "hybrid").strip().lower()
    if requested_mode not in {"lexical", "semantic", "hybrid"}:
        raise SchemaProjectionError(
            code="named_service_schema_search_mode_invalid",
            message="search_mode must be lexical, semantic, or hybrid.",
            details={"search_mode": requested_mode},
        )
    selected_kind = str(object_kind or "").strip()
    if selected_kind and selected_kind not in kind_specs:
        raise SchemaProjectionError(
            code="named_service_schema_kind_unknown",
            message=f"Unknown object kind {selected_kind!r}.",
            details={"available_object_kinds": list(kind_specs)},
        )
    bounded_limit = max(1, min(int(limit or 10), 50))
    result = dict(search_result or {})
    if not result:
        matches = search_catalog_lexical(
            build_schema_tree(full_schema, projection_index),
            query=clean_query,
            limit=bounded_limit,
            object_kind=selected_kind,
        )
        result = {
            "matches": matches,
            "requested_search_mode": requested_mode,
            "effective_search_mode": "lexical" if matches else "no_matches",
            "match_sources": ["lexical"] if matches else [],
            "backend": "in_memory_lexical",
            "degraded_reason": (
                "shared_capability_index_not_configured"
                if requested_mode != "lexical"
                else ""
            ),
        }
    matches = list(result.get("matches") or [])[:bounded_limit]
    projected = {
        key: full_schema[key]
        for key in ("namespace", "schema_version")
        if key in full_schema
    }
    projected["catalog_search"] = {
        "query": clean_query,
        "count": len(matches),
        "limit": bounded_limit,
        **{
            key: value
            for key, value in result.items()
            if key != "matches" and value not in (None, "")
        },
        "matches": matches,
    }
    projected["schema_projection"] = _projection_metadata(
        view=SCHEMA_VIEW_SEARCH,
        kind_specs=kind_specs,
        object_kind=selected_kind,
    )
    return projected


def _resolve_kind(
    *,
    request: NamedServiceRequest,
    kind_specs: Mapping[str, Any],
    schema_operation: str,
    object_kind_from_ref: Callable[[str], str | None] | None,
) -> str:
    requested_kind = str(
        getattr(request, "object_kind", None)
        or _mapping(request.payload).get("object_kind")
        or ""
    ).strip()
    ref_kind = ""
    if request.object_ref and object_kind_from_ref is not None:
        ref_kind = str(object_kind_from_ref(request.object_ref) or "").strip()
    if requested_kind and ref_kind and requested_kind != ref_kind:
        raise SchemaProjectionError(
            code="named_service_schema_kind_conflict",
            message="object_kind does not match the concrete object_ref.",
            details={
                "object_kind": requested_kind,
                "object_ref_kind": ref_kind,
            },
        )
    object_kind = requested_kind or ref_kind
    if object_kind:
        if object_kind not in kind_specs:
            raise SchemaProjectionError(
                code="named_service_schema_kind_unknown",
                message=f"Unknown object kind {object_kind!r}.",
                details={"available_object_kinds": list(kind_specs)},
            )
        return object_kind

    if request.object_ref and object_kind_from_ref is not None:
        raise SchemaProjectionError(
            code="named_service_schema_ref_kind_unknown",
            message="The provider could not map this object ref to a schema kind.",
            details={
                "object_ref": request.object_ref,
                "available_object_kinds": list(kind_specs),
            },
        )

    if schema_operation:
        matching_kinds = [
            kind
            for kind, spec in kind_specs.items()
            if _normalize_schema_operation(schema_operation, _mapping(spec), strict=False)
        ]
        if len(matching_kinds) == 1:
            return matching_kinds[0]
        if matching_kinds:
            raise SchemaProjectionError(
                code="named_service_schema_kind_required",
                message="This schema operation applies to more than one object kind.",
                details={
                    "schema_operation": schema_operation,
                    "candidate_object_kinds": matching_kinds,
                },
            )
        raise SchemaProjectionError(
            code="named_service_schema_operation_unknown",
            message=f"Unknown schema operation {schema_operation!r}.",
            details={
                "available_operations_by_kind": {
                    kind: list(_available_operations(_mapping(spec)))
                    for kind, spec in kind_specs.items()
                }
            },
        )

    raise SchemaProjectionError(
        code="named_service_schema_kind_required",
        message="A kind or concrete object ref is required for this schema view.",
        details={"available_object_kinds": list(kind_specs)},
    )


def _normalize_schema_operation(
    value: str,
    kind_spec: Mapping[str, Any],
    *,
    strict: bool = True,
) -> str:
    raw = str(value or "").strip()
    available = _available_operations(kind_spec)
    if raw in available:
        return raw
    alias = _GENERIC_OPERATION_ALIASES.get(raw)
    if alias in available:
        return str(alias)
    action_operation = _action_operation(raw)
    if action_operation in available:
        return action_operation
    if strict:
        raise SchemaProjectionError(
            code="named_service_schema_operation_unknown",
            message=f"Unknown schema operation {raw!r} for the selected object kind.",
            details={"available_operations": list(available)},
        )
    return ""


def _base_kind_projection(
    *,
    schema: Mapping[str, Any],
    index: Mapping[str, Any],
    object_kind: str,
    operations: tuple[str, ...],
    include_fields: bool,
) -> dict[str, Any]:
    kind_specs = _kind_specs(index)
    kind_spec = _mapping(kind_specs.get(object_kind))
    schema_kinds = _mapping(schema.get("object_kinds"))
    schema_refs = _mapping(schema.get("refs"))
    schema_selectors = _mapping(schema.get("selectors"))
    projected: dict[str, Any] = {}
    for key in ("namespace", "schema_version"):
        if key in schema:
            projected[key] = schema[key]
    projected["refs"] = {
        name: schema_refs[name]
        for name in _strings(kind_spec.get("refs"))
        if name in schema_refs
    }
    kind_contract = _mapping(schema_kinds[object_kind])
    if not include_fields:
        kind_contract.pop("fields", None)
    projected["object_kinds"] = {object_kind: kind_contract}
    related = {
        kind: {
            "description": str(
                _mapping(schema_kinds.get(kind)).get("description") or ""
            )
        }
        for kind in _strings(kind_spec.get("related_kinds"))
        if kind in schema_kinds and kind != object_kind
    }
    if related:
        projected["related_object_kinds"] = related
    selectors = {
        name: schema_selectors[name]
        for name in _strings(kind_spec.get("selectors"))
        if name in schema_selectors
    }
    if selectors:
        projected["selectors"] = selectors
    _copy_kind_globals(
        schema=schema,
        index=index,
        target=projected,
        operations=operations,
    )
    return projected


def _project_operation_sections(
    *,
    schema: Mapping[str, Any],
    target: dict[str, Any],
    operation_spec: Mapping[str, Any],
) -> None:
    section_keys = _mapping(operation_spec.get("section_keys"))
    for section in _strings(operation_spec.get("sections")):
        if section not in schema:
            continue
        value = schema[section]
        requested_keys = _strings(section_keys.get(section))
        if requested_keys and isinstance(value, Mapping):
            target[section] = {
                key: value[key] for key in requested_keys if key in value
            }
        else:
            target[section] = value


def project_schema(
    *,
    schema: Mapping[str, Any],
    index: Mapping[str, Any],
    request: NamedServiceRequest,
    object_kind_from_ref: Callable[[str], str | None] | None = None,
    catalog_search_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    full_schema = _mapping(schema)
    projection_index = _mapping(index)
    _validate_projection_index(full_schema, projection_index)
    kind_specs = _kind_specs(projection_index)
    payload = _mapping(request.payload)
    requested_view = str(
        getattr(request, "schema_view", None)
        or payload.get("schema_view")
        or payload.get("view")
        or ""
    ).strip().lower()
    requested_operation = str(
        getattr(request, "schema_operation", None)
        or payload.get("schema_operation")
        or ""
    ).strip()
    requested_path = str(
        getattr(request, "schema_path", None)
        or payload.get("schema_path")
        or ""
    ).strip()
    query = str(request.query or payload.get("query") or "").strip()
    search_mode = str(
        request.search_mode or payload.get("search_mode") or "hybrid"
    ).strip().lower()
    if requested_view and requested_view not in SCHEMA_VIEWS:
        raise SchemaProjectionError(
            code="named_service_schema_view_invalid",
            message=f"Unknown schema view {requested_view!r}.",
            details={"available_views": sorted(SCHEMA_VIEWS)},
        )
    view = requested_view
    if not view:
        if requested_operation:
            view = SCHEMA_VIEW_OPERATION
        elif query:
            view = SCHEMA_VIEW_SEARCH
        elif requested_path:
            view = SCHEMA_VIEW_CATALOG
        elif getattr(request, "object_kind", None) or request.object_ref or payload.get("object_kind"):
            view = SCHEMA_VIEW_KIND
        else:
            view = SCHEMA_VIEW_CATALOG

    if view == SCHEMA_VIEW_FULL:
        if requested_operation or requested_path or query:
            raise SchemaProjectionError(
                code="named_service_schema_view_conflict",
                message="The full schema view does not accept progressive selectors.",
                details={"schema_view": view},
            )
        projected = dict(full_schema)
        projected["schema_projection"] = _projection_metadata(
            view=view,
            kind_specs=kind_specs,
        )
        return projected
    if view == SCHEMA_VIEW_CATALOG:
        if requested_operation or query:
            raise SchemaProjectionError(
                code="named_service_schema_view_conflict",
                message=(
                    "schema_operation requires the operation view and query "
                    "requires the search view."
                ),
                details={"schema_view": view},
            )
        return build_schema_catalog(
            full_schema,
            projection_index,
            schema_path=requested_path or "/",
        )
    if view == SCHEMA_VIEW_SEARCH:
        if requested_operation or requested_path or request.object_ref:
            raise SchemaProjectionError(
                code="named_service_schema_view_conflict",
                message=(
                    "Capability search accepts query and an optional object_kind; "
                    "browse schema_path or expand schema_operation separately."
                ),
                details={"schema_view": view},
            )
        return build_schema_search(
            full_schema,
            projection_index,
            query=query,
            search_mode=search_mode,
            limit=request.limit or 10,
            object_kind=str(
                getattr(request, "object_kind", None)
                or payload.get("object_kind")
                or ""
            ),
            search_result=catalog_search_result,
        )

    if requested_path or query:
        raise SchemaProjectionError(
            code="named_service_schema_view_conflict",
            message=(
                "schema_path requires the catalog view and query requires "
                "the search view."
            ),
            details={"schema_view": view},
        )

    object_kind = _resolve_kind(
        request=request,
        kind_specs=kind_specs,
        schema_operation=requested_operation,
        object_kind_from_ref=object_kind_from_ref,
    )
    kind_spec = _mapping(kind_specs.get(object_kind))
    if view == SCHEMA_VIEW_KIND:
        if requested_operation:
            raise SchemaProjectionError(
                code="named_service_schema_view_conflict",
                message="schema_operation requires the operation schema view.",
                details={"schema_view": view},
            )
        available_operations = _available_operations(kind_spec)
        projected = _base_kind_projection(
            schema=full_schema,
            index=projection_index,
            object_kind=object_kind,
            operations=available_operations,
            include_fields=True,
        )
        projected["operations"] = _kind_operation_summaries(
            schema=full_schema,
            kind_spec=kind_spec,
        )
        projected["schema_projection"] = _projection_metadata(
            view=view,
            kind_specs=kind_specs,
            object_kind=object_kind,
        )
        return projected

    if not requested_operation:
        raise SchemaProjectionError(
            code="named_service_schema_operation_required",
            message="The operation schema view requires schema_operation.",
            details={
                "object_kind": object_kind,
                "available_operations": list(_available_operations(kind_spec)),
            },
        )
    schema_operation = _normalize_schema_operation(requested_operation, kind_spec)
    projected = _base_kind_projection(
        schema=full_schema,
        index=projection_index,
        object_kind=object_kind,
        operations=(schema_operation,),
        include_fields=not schema_operation.startswith("object.action:"),
    )
    if schema_operation.startswith("object.action:"):
        action = schema_operation.split(":", 1)[1]
        projected["actions"] = {
            action: _mapping(full_schema.get("actions"))[action]
        }
    else:
        operation_spec = _mapping(_operation_specs(kind_spec).get(schema_operation))
        _project_operation_sections(
            schema=full_schema,
            target=projected,
            operation_spec=operation_spec,
        )
    projected["operation"] = {
        "id": schema_operation,
        **_kind_operation_summaries(
            schema=full_schema,
            kind_spec=kind_spec,
        )[schema_operation],
    }
    projected["schema_projection"] = _projection_metadata(
        view=view,
        kind_specs=kind_specs,
        object_kind=object_kind,
        schema_operation=schema_operation,
    )
    return projected


def project_schema_response(
    *,
    response: NamedServiceResponse | Mapping[str, Any],
    request: NamedServiceRequest,
    index: Mapping[str, Any],
    object_kind_from_ref: Callable[[str], str | None] | None = None,
    catalog_search_result: Mapping[str, Any] | None = None,
) -> NamedServiceResponse | Mapping[str, Any]:
    if not index:
        return response
    coerced = NamedServiceResponse.coerce(response)
    if not coerced.ok:
        return response
    extra = dict(coerced.extra)
    schema = extra.get("schema")
    if not isinstance(schema, Mapping):
        return response
    try:
        extra["schema"] = project_schema(
            schema=schema,
            index=index,
            request=request,
            object_kind_from_ref=object_kind_from_ref,
            catalog_search_result=catalog_search_result,
        )
    except SchemaProjectionError as exc:
        return NamedServiceResponse.error_response(
            code=exc.code,
            message=exc.message,
            status=exc.status,
            details=exc.details,
            provider=coerced.provider,
            namespace=coerced.namespace or request.namespace,
            object_ref=request.object_ref,
        )
    ret = dict(coerced.ret)
    ret["extra"] = extra
    return dataclasses.replace(coerced, ret=ret)


async def project_schema_response_async(
    *,
    response: NamedServiceResponse | Mapping[str, Any],
    request: NamedServiceRequest,
    index: Mapping[str, Any],
    object_kind_from_ref: Callable[[str], str | None] | None = None,
    search_index: Any = None,
) -> NamedServiceResponse | Mapping[str, Any]:
    """Project a schema response, using the shared hybrid catalog when bound."""

    if not index or search_index is None:
        return project_schema_response(
            response=response,
            request=request,
            index=index,
            object_kind_from_ref=object_kind_from_ref,
        )
    coerced = NamedServiceResponse.coerce(response)
    if not coerced.ok:
        return response
    schema = coerced.extra.get("schema")
    if not isinstance(schema, Mapping):
        return response
    payload = _mapping(request.payload)
    query = str(request.query or payload.get("query") or "").strip()
    requested_view = str(
        request.schema_view or payload.get("schema_view") or payload.get("view") or ""
    ).strip().lower()
    if not query and requested_view != SCHEMA_VIEW_SEARCH:
        return project_schema_response(
            response=response,
            request=request,
            index=index,
            object_kind_from_ref=object_kind_from_ref,
        )
    try:
        # Validate selector combinations before touching the shared index.
        project_schema(
            schema=schema,
            index=index,
            request=request,
            object_kind_from_ref=object_kind_from_ref,
            catalog_search_result={"matches": []},
        )
        tree = build_schema_tree(schema, index)
        result = await search_index.search(
            tree,
            query=query,
            mode=str(request.search_mode or payload.get("search_mode") or "hybrid"),
            limit=request.limit or 10,
            object_kind=str(
                request.object_kind or payload.get("object_kind") or ""
            ).strip(),
        )
    except SchemaProjectionError as exc:
        return NamedServiceResponse.error_response(
            code=exc.code,
            message=exc.message,
            status=exc.status,
            details=exc.details,
            provider=coerced.provider,
            namespace=coerced.namespace or request.namespace,
            object_ref=request.object_ref,
        )
    except ValueError as exc:
        return NamedServiceResponse.error_response(
            code="named_service_schema_search_invalid",
            message=str(exc),
            status=400,
            provider=coerced.provider,
            namespace=coerced.namespace or request.namespace,
            object_ref=request.object_ref,
        )
    return project_schema_response(
        response=response,
        request=request,
        index=index,
        object_kind_from_ref=object_kind_from_ref,
        catalog_search_result=result,
    )
