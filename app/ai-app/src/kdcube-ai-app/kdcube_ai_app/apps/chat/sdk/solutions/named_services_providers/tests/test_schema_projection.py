# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kdcube_ai_app.apps.chat.sdk.integrations.docs.named_service import (
    DOCS_SCHEMA,
    DOCS_SCHEMA_PROJECTION,
)
from kdcube_ai_app.apps.chat.sdk.integrations.mail.named_service import (
    MAIL_SCHEMA,
    MAIL_SCHEMA_PROJECTION,
)
from kdcube_ai_app.apps.chat.sdk.integrations.sheets.named_service import (
    SHEETS_SCHEMA,
    SHEETS_SCHEMA_PROJECTION,
)
from kdcube_ai_app.apps.chat.sdk.integrations.slack.named_service import (
    SLACK_SCHEMA,
    SLACK_SCHEMA_PROJECTION,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceContext,
    NamedServiceProvider,
    NamedServiceRequest,
    NamedServiceResponse,
    SchemaProjectionError,
    build_schema_catalog,
    build_schema_tree,
    project_schema,
    project_schema_response_async,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.schema_search import (
    SchemaCatalogSearchIndex,
    schema_search_embedding_profile,
    schema_search_index_path,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    OBJECT_SCHEMA,
    PROVIDER_ABOUT,
)


def _large_schema() -> dict[str, Any]:
    actions = {
        f"action_{index:03d}": {
            "description": (
                f"Run domain action {index:03d}. "
                + "This intentionally verbose contract proves that unrelated "
                "action payloads do not enter a focused schema response. " * 3
            ),
            "object_ref": "realm:item:<item_id>",
            "payload": [f"field_{index}", "reason", "account_id"],
            "result": ["object_ref", "revision", "status"],
        }
        for index in range(100)
    }
    return {
        "namespace": "realm",
        "schema_version": "1",
        "refs": {"item": "realm:item:<item_id>"},
        "object_kinds": {
            "realm.item": {
                "description": "One item in the synthetic large realm.",
                "fields": ["ref", "title", "status", "revision"],
            }
        },
        "search": {
            "description": "Search item titles supplied by the provider.",
            "filters": {
                "account_id": {"type": "string"},
                "status": {"type": "string"},
            },
        },
        "get": {"description": "Read one item by ref."},
        "upsert": {
            "description": "Create or update one item.",
            "object": ["title", "status"],
        },
        "actions": actions,
        "grant_hints": {
            "object.search": ["realm:read"],
            "object.get": ["realm:read"],
            "object.upsert": ["realm:write"],
            **{
                f"object.action.{action}": ["realm:act"]
                for action in actions
            },
        },
        "account_selection": {
            "description": "Select one connected provider account when required."
        },
    }


def _large_projection() -> dict[str, Any]:
    return {
        "catalog": {
            "id": "realm",
            "label": "Synthetic realm",
            "children": [
                {
                    "id": "records",
                    "label": "Find and edit records",
                    "object_kind": "realm.item",
                    "operations": [
                        "object.search",
                        "object.get",
                        "object.upsert",
                    ],
                },
                {
                    "id": "automation",
                    "label": "Automations",
                    "children": [
                        {
                            "id": f"group-{group:02d}",
                            "label": f"Automation group {group:02d}",
                            "object_kind": "realm.item",
                            "operations": [
                                f"object.action:action_{index:03d}"
                                for index in range(group * 10, group * 10 + 10)
                            ],
                        }
                        for group in range(10)
                    ],
                },
            ],
        },
        "kinds": {
            "realm.item": {
                "refs": ["item"],
                "operations": {
                    "object.search": {"sections": ["search"]},
                    "object.get": {"sections": ["get"]},
                    "object.upsert": {"sections": ["upsert"]},
                },
                "actions": [f"action_{index:03d}" for index in range(100)],
            }
        }
    }


def _request(
    operation: str = OBJECT_SCHEMA,
    **kwargs: Any,
) -> NamedServiceRequest:
    return NamedServiceRequest(operation=operation, namespace="realm", **kwargs)


def test_schema_request_round_trips_recursive_and_search_selectors() -> None:
    browse = NamedServiceRequest.from_dict(
        {
            "operation": "object.schema",
            "namespace": "realm",
            "schema_path": "/automation/group-04",
        }
    )
    search = NamedServiceRequest.from_dict(
        {
            "operation": "object.schema",
            "namespace": "realm",
            "query": "action 042",
            "search_mode": "hybrid",
            "limit": 7,
        }
    )
    assert browse.schema_path == "/automation/group-04"
    assert browse.to_dict()["schema_path"] == "/automation/group-04"
    assert search.query == "action 042"
    assert search.search_mode == "hybrid"


def test_large_provider_discloses_catalog_kind_and_one_operation_progressively() -> None:
    schema = _large_schema()
    index = _large_projection()

    catalog = project_schema(schema=schema, index=index, request=_request())
    assert catalog["schema_projection"]["view"] == "catalog"
    assert catalog["object_kinds"]["realm.item"]["operation_count"] == 103
    assert "operations" not in catalog["object_kinds"]["realm.item"]
    assert [node["path"] for node in catalog["catalog"]["children"]] == [
        "/records",
        "/automation",
    ]
    assert catalog["catalog"]["operations"] == []
    assert "actions" not in catalog

    catalog_branch = project_schema(
        schema=schema,
        index=index,
        request=_request(schema_path="/automation/group-04"),
    )
    assert catalog_branch["schema_projection"]["schema_path"] == (
        "/automation/group-04"
    )
    assert len(catalog_branch["catalog"]["operations"]) == 10
    assert catalog_branch["catalog"]["operations"][2]["schema_operation"] == (
        "object.action:action_042"
    )

    search = project_schema(
        schema=schema,
        index=index,
        request=_request(query="run domain action 042", search_mode="hybrid"),
    )
    assert search["schema_projection"]["view"] == "search"
    assert search["catalog_search"]["effective_search_mode"] == "lexical"
    assert search["catalog_search"]["matches"][0]["schema_operation"] == (
        "object.action:action_042"
    )

    kind = project_schema(
        schema=schema,
        index=index,
        request=_request(object_kind="realm.item"),
    )
    assert kind["schema_projection"]["view"] == "kind"
    assert kind["object_kinds"]["realm.item"]["fields"] == [
        "ref",
        "title",
        "status",
        "revision",
    ]
    assert len(kind["operations"]) == 103
    assert "actions" not in kind

    focused = project_schema(
        schema=schema,
        index=index,
        request=_request(
            object_kind="realm.item",
            schema_operation="object.action:action_042",
        ),
    )
    assert focused["schema_projection"] == {
        "contract": "kdcube.named-service.schema-projection.v1",
        "view": "operation",
        "available_object_kinds": ["realm.item"],
        "object_kind": "realm.item",
        "schema_operation": "object.action:action_042",
    }
    assert list(focused["actions"]) == ["action_042"]
    assert "fields" not in focused["object_kinds"]["realm.item"]
    assert focused["grant_hints"] == {
        "object.action.action_042": ["realm:act"]
    }

    full = project_schema(
        schema=schema,
        index=index,
        request=_request(schema_view="full"),
    )
    assert len(full["actions"]) == 100
    assert len(json.dumps(focused)) < len(json.dumps(full)) // 10


def test_exact_generic_operation_contains_only_its_declared_contract() -> None:
    focused = project_schema(
        schema=_large_schema(),
        index=_large_projection(),
        request=_request(
            object_kind="realm.item",
            schema_operation="search",
        ),
    )

    assert focused["operation"]["id"] == "object.search"
    assert focused["search"]["filters"]["status"]["type"] == "string"
    assert "get" not in focused
    assert "upsert" not in focused
    assert "actions" not in focused
    assert focused["grant_hints"] == {"object.search": ["realm:read"]}


def test_concrete_ref_infers_kind_and_conflicting_kind_is_rejected() -> None:
    kwargs = {
        "schema": _large_schema(),
        "index": _large_projection(),
        "object_kind_from_ref": lambda ref: "realm.item" if ref.startswith("realm:item:") else None,
    }
    focused = project_schema(
        **kwargs,
        request=_request(
            object_ref="realm:item:item-7",
            schema_operation="object.get",
        ),
    )
    assert focused["schema_projection"]["object_kind"] == "realm.item"

    with pytest.raises(SchemaProjectionError) as raised:
        project_schema(
            **kwargs,
            request=_request(
                object_ref="realm:item:item-7",
                object_kind="realm.other",
            ),
        )
    assert raised.value.code == "named_service_schema_kind_conflict"

    with pytest.raises(SchemaProjectionError) as unknown_ref:
        project_schema(
            **kwargs,
            request=_request(object_ref="realm:other:item-7"),
        )
    assert unknown_ref.value.code == "named_service_schema_ref_kind_unknown"


def test_unknown_operation_and_invalid_provider_index_return_specific_errors() -> None:
    with pytest.raises(SchemaProjectionError) as unknown:
        project_schema(
            schema=_large_schema(),
            index=_large_projection(),
            request=_request(
                object_kind="realm.item",
                schema_operation="object.action:missing",
            ),
        )
    assert unknown.value.code == "named_service_schema_operation_unknown"
    assert "object.action:action_000" in unknown.value.details["available_operations"]

    with pytest.raises(SchemaProjectionError) as unknown_without_kind:
        project_schema(
            schema=_large_schema(),
            index=_large_projection(),
            request=_request(schema_operation="missing"),
        )
    assert unknown_without_kind.value.code == "named_service_schema_operation_unknown"

    invalid_index = _large_projection()
    invalid_index["kinds"]["realm.item"]["actions"].remove("action_099")
    with pytest.raises(SchemaProjectionError) as invalid:
        build_schema_catalog(_large_schema(), invalid_index)
    assert invalid.value.code == "named_service_schema_projection_invalid"
    assert invalid.value.details["unassigned_actions"] == ["action_099"]

    invalid_relation = _large_projection()
    invalid_relation["kinds"]["realm.item"]["related_kinds"] = ["realm.missing"]
    with pytest.raises(SchemaProjectionError) as invalid_related:
        build_schema_catalog(_large_schema(), invalid_relation)
    assert invalid_related.value.code == "named_service_schema_projection_invalid"
    assert invalid_related.value.details["unknown_related_kinds"] == [
        "realm.missing"
    ]

    invalid_section = _large_projection()
    invalid_section["catalog_sections"] = ["missing_section"]
    with pytest.raises(SchemaProjectionError) as invalid_catalog:
        build_schema_catalog(_large_schema(), invalid_section)
    assert invalid_catalog.value.code == "named_service_schema_projection_invalid"
    assert invalid_catalog.value.details["unknown_sections"] == ["missing_section"]

    with pytest.raises(SchemaProjectionError) as unknown_path:
        project_schema(
            schema=_large_schema(),
            index=_large_projection(),
            request=_request(schema_path="/missing"),
        )
    assert unknown_path.value.code == "named_service_schema_path_unknown"


@pytest.mark.parametrize(
    ("schema", "index", "expected_kinds"),
    [
        (DOCS_SCHEMA, DOCS_SCHEMA_PROJECTION, 3),
        (SHEETS_SCHEMA, SHEETS_SCHEMA_PROJECTION, 2),
        (MAIL_SCHEMA, MAIL_SCHEMA_PROJECTION, 3),
        (SLACK_SCHEMA, SLACK_SCHEMA_PROJECTION, 5),
    ],
)
def test_shipped_provider_projection_indexes_cover_their_full_schemas(
    schema: dict[str, Any],
    index: dict[str, Any],
    expected_kinds: int,
) -> None:
    catalog = build_schema_catalog(schema, index)
    assert catalog["schema_projection"]["view"] == "catalog"
    assert len(catalog["object_kinds"]) == expected_kinds
    assert "actions" not in catalog


class _ProjectedProvider(NamedServiceProvider):
    schema_projection_index = _large_projection()

    async def provider_about(
        self,
        ctx: NamedServiceContext,
        request: NamedServiceRequest,
    ) -> NamedServiceResponse:
        del ctx, request
        return NamedServiceResponse.ok_response(
            namespace="realm",
            extra={"schema": _large_schema(), "purpose": "Synthetic realm"},
        )

    async def object_schema(
        self,
        ctx: NamedServiceContext,
        request: NamedServiceRequest,
    ) -> NamedServiceResponse:
        del ctx, request
        return NamedServiceResponse.ok_response(
            namespace="realm",
            extra={"schema": _large_schema()},
        )

    def schema_object_kind_from_ref(self, object_ref: str) -> str | None:
        return "realm.item" if object_ref.startswith("realm:item:") else None


class _UnprojectedProvider(NamedServiceProvider):
    async def object_schema(
        self,
        ctx: NamedServiceContext,
        request: NamedServiceRequest,
    ) -> NamedServiceResponse:
        del ctx, request
        return NamedServiceResponse.ok_response(
            namespace="realm",
            extra={"schema": _large_schema()},
        )


@pytest.mark.asyncio
async def test_provider_dispatch_projects_about_and_schema_and_bounds_errors() -> None:
    provider = _ProjectedProvider()
    ctx = NamedServiceContext(tenant="tenant", project="project", user_id="user")

    about = await provider.dispatch(ctx, _request(operation=PROVIDER_ABOUT))
    assert NamedServiceResponse.coerce(about).extra["schema"]["schema_projection"]["view"] == "catalog"

    schema = await provider.dispatch(
        ctx,
        _request(
            object_ref="realm:item:item-7",
            schema_operation="object.action:action_007",
        ),
    )
    coerced = NamedServiceResponse.coerce(schema)
    assert coerced.ok is True
    assert list(coerced.extra["schema"]["actions"]) == ["action_007"]

    rejected = await provider.dispatch(
        ctx,
        _request(
            object_kind="realm.item",
            schema_operation="object.action:missing",
        ),
    )
    error = NamedServiceResponse.coerce(rejected)
    assert error.ok is False
    assert error.status == 400
    assert error.error is not None
    assert error.error.code == "named_service_schema_operation_unknown"

    searched = await provider.dispatch(
        ctx,
        _request(query="action 007", search_mode="lexical"),
    )
    search_schema = NamedServiceResponse.coerce(searched).extra["schema"]
    assert search_schema["catalog_search"]["matches"][0]["schema_operation"] == (
        "object.action:action_007"
    )


class _EmbeddingModel:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_search_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = str(text or "").lower()
        return [
            float(lowered.count("comment") + lowered.count("reply")),
            float(lowered.count("action 042") + lowered.count("action_042")),
            float(lowered.count("record")),
            1.0,
        ]


class _TrackedEmbeddingModel(_EmbeddingModel):
    provider = "test"

    def __init__(self, model: str) -> None:
        self.model = model
        self.document_embed_calls = 0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.document_embed_calls += 1
        return await super().embed_texts(texts)


@pytest.mark.asyncio
async def test_shared_capability_index_supports_hybrid_search(tmp_path) -> None:
    tree = build_schema_tree(_large_schema(), _large_projection())
    index = SchemaCatalogSearchIndex(
        db_path=tmp_path / "capabilities.sqlite",
        model_service=_EmbeddingModel(),
        dim=4,
        min_semantic_score=0.0,
        vector_backend="bruteforce",
    )
    ensured = await index.ensure(tree)
    assert ensured["indexed"] == 103
    assert ensured["updated"] is True
    reused = await index.ensure(tree)
    assert reused["updated"] is False

    result = await index.search(
        tree,
        query="action 042",
        mode="hybrid",
        limit=5,
    )
    assert result["backend"] == "shared_hybrid_index"
    assert result["effective_search_mode"] == "hybrid"
    assert result["matches"][0]["schema_operation"] == (
        "object.action:action_042"
    )


@pytest.mark.asyncio
async def test_capability_index_reembeds_when_embedding_profile_changes(tmp_path) -> None:
    tree = build_schema_tree(_large_schema(), _large_projection())
    db_path = tmp_path / "capabilities.sqlite"
    first_model = _TrackedEmbeddingModel("embed-v1")
    first_index = SchemaCatalogSearchIndex(
        db_path=db_path,
        model_service=first_model,
        dim=4,
        vector_backend="bruteforce",
    )
    first = await first_index.ensure(tree)
    first_ids = set(first_index.index.ids())
    assert first["updated"] is True
    assert first_model.document_embed_calls == 1

    reused_model = _TrackedEmbeddingModel("embed-v1")
    reused_index = SchemaCatalogSearchIndex(
        db_path=db_path,
        model_service=reused_model,
        dim=4,
        vector_backend="bruteforce",
    )
    reused = await reused_index.ensure(tree)
    assert reused["updated"] is False
    assert reused_model.document_embed_calls == 0

    replacement_model = _TrackedEmbeddingModel("embed-v2")
    replacement_index = SchemaCatalogSearchIndex(
        db_path=db_path,
        model_service=replacement_model,
        dim=4,
        vector_backend="bruteforce",
    )
    replaced = await replacement_index.ensure(tree)
    assert replaced["updated"] is True
    assert replacement_model.document_embed_calls == 1
    assert set(replacement_index.index.ids()).isdisjoint(first_ids)


def test_capability_index_path_isolated_by_embedding_profile(tmp_path) -> None:
    first_profile = schema_search_embedding_profile(
        _TrackedEmbeddingModel("embed-v1"),
        dim=4,
    )
    replacement_profile = schema_search_embedding_profile(
        _TrackedEmbeddingModel("embed-v2"),
        dim=4,
    )
    first_path = schema_search_index_path(
        tmp_path,
        "provider.docs",
        embedding_profile=first_profile,
    )
    replacement_path = schema_search_index_path(
        tmp_path,
        "provider.docs",
        embedding_profile=replacement_profile,
    )

    assert first_path != replacement_path
    assert first_path.parent == replacement_path.parent
    assert first_path.name.startswith("capabilities.")


def test_capability_index_path_isolated_by_vector_backend(tmp_path) -> None:
    faiss_profile = schema_search_embedding_profile(
        _TrackedEmbeddingModel("embed-v1"),
        dim=4,
        vector_backend="faiss-local",
    )
    brute_profile = schema_search_embedding_profile(
        _TrackedEmbeddingModel("embed-v1"),
        dim=4,
        vector_backend="bruteforce",
    )

    faiss_path = schema_search_index_path(
        tmp_path,
        "provider.docs",
        embedding_profile=faiss_profile,
    )
    brute_path = schema_search_index_path(
        tmp_path,
        "provider.docs",
        embedding_profile=brute_profile,
    )

    assert faiss_path != brute_path
    assert faiss_path.parent == brute_path.parent


@pytest.mark.asyncio
async def test_invalid_search_selectors_do_not_touch_shared_index() -> None:
    class _SearchIndex:
        calls = 0

        async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            self.calls += 1
            return {"matches": []}

    search_index = _SearchIndex()
    response = await project_schema_response_async(
        response=NamedServiceResponse.ok_response(
            namespace="realm",
            extra={"schema": _large_schema()},
        ),
        request=_request(
            query="action 042",
            schema_path="/automation/group-04",
        ),
        index=_large_projection(),
        search_index=search_index,
    )

    rejected = NamedServiceResponse.coerce(response)
    assert rejected.ok is False
    assert rejected.error is not None
    assert rejected.error.code == "named_service_schema_view_conflict"
    assert search_index.calls == 0


@pytest.mark.asyncio
async def test_provider_prepares_and_uses_bound_capability_index(tmp_path) -> None:
    provider = _ProjectedProvider()
    provider.configure_schema_catalog_search(
        storage_root=tmp_path,
        model_service=_EmbeddingModel(),
        dim=4,
        vector_backend="bruteforce",
    )
    prepared = await provider.ensure_schema_catalog_index(namespace="realm")
    assert prepared[0]["indexed"] == 103
    assert prepared[0]["vector_backend"] == "bruteforce"
    assert prepared[0]["vector_path"] == ""

    response = await provider.dispatch(
        NamedServiceContext(),
        _request(query="action 042", search_mode="hybrid"),
    )
    search = NamedServiceResponse.coerce(response).extra["schema"]["catalog_search"]
    assert search["backend"] == "shared_hybrid_index"
    assert search["vector_backend"] == "bruteforce"
    assert search["matches"][0]["schema_operation"] == (
        "object.action:action_042"
    )


@pytest.mark.asyncio
async def test_provider_degrades_to_lexical_when_embedding_binding_is_removed(
    tmp_path,
) -> None:
    provider = _ProjectedProvider()
    provider.configure_schema_catalog_search(
        storage_root=tmp_path,
        model_service=_EmbeddingModel(),
        dim=4,
        vector_backend="bruteforce",
    )
    assert await provider.ensure_schema_catalog_index(namespace="realm")

    provider.configure_schema_catalog_search(
        storage_root=tmp_path,
        model_service=None,
        dim=4,
        vector_backend="bruteforce",
    )
    assert await provider.ensure_schema_catalog_index(namespace="realm") == []

    response = await provider.dispatch(
        NamedServiceContext(),
        _request(query="action 042", search_mode="hybrid"),
    )
    search = NamedServiceResponse.coerce(response).extra["schema"]["catalog_search"]
    assert search["backend"] == "in_memory_lexical"
    assert search["effective_search_mode"] == "lexical"
    assert search["degraded_reason"] == "shared_capability_index_not_configured"


@pytest.mark.asyncio
async def test_provider_defaults_to_persistent_faiss_capability_index(
    tmp_path,
) -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    provider = _ProjectedProvider()
    provider.configure_schema_catalog_search(
        storage_root=tmp_path,
        model_service=_EmbeddingModel(),
        dim=4,
    )

    prepared = await provider.ensure_schema_catalog_index(namespace="realm")
    vector_path = Path(prepared[0]["vector_path"])
    assert prepared[0]["vector_backend"] == "faiss-local"
    assert vector_path.suffix == ".faiss"
    assert vector_path.exists()

    response = await provider.dispatch(
        NamedServiceContext(),
        _request(query="action 042", search_mode="hybrid"),
    )
    search = NamedServiceResponse.coerce(response).extra["schema"]["catalog_search"]
    assert search["backend"] == "shared_hybrid_index"
    assert search["vector_backend"] == "faiss-local"
    assert search["vector_path"] == str(vector_path)
    assert search["matches"][0]["schema_operation"] == (
        "object.action:action_042"
    )

    vector_path.unlink()
    assert not vector_path.exists()
    restored = await provider.ensure_schema_catalog_index(namespace="realm")
    assert restored[0]["updated"] is False
    assert vector_path.exists()


@pytest.mark.asyncio
async def test_provider_without_projection_index_preserves_its_schema() -> None:
    response = await _UnprojectedProvider().dispatch(
        NamedServiceContext(),
        _request(),
    )
    schema = NamedServiceResponse.coerce(response).extra["schema"]
    assert len(schema["actions"]) == 100
    assert "schema_projection" not in schema
