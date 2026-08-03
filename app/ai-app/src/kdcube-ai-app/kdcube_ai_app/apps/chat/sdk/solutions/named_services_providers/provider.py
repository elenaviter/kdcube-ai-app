# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.schema_projection import (
    build_schema_tree,
    project_schema_response_async,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.schema_search import (
    DEFAULT_SCHEMA_VECTOR_BACKEND,
    SchemaCatalogSearchIndex,
    normalize_schema_vector_backend,
    schema_search_embedding_profile,
    schema_search_index_path,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    OBJECT_SCHEMA,
    NamedServiceContext,
    NamedServiceProviderSpec,
    NamedServiceRequest,
    NamedServiceResponse,
    build_default_operations,
)

LOGGER = logging.getLogger("kdcube.sdk.named_services.provider")

# Uniform batch get: a single object.get carrying a list of refs
# (filters.refs / filters.object_refs) fans out to the provider's single-object
# object.get and returns the objects as items. Handled once here so EVERY provider
# (mem, task, conv, canvas, ...) supports batch get identically — providers only
# ever implement single-object object.get.
BATCH_GET_MAX = 50
_BATCH_GET_KEYS = ("refs", "object_refs")


def batch_get_refs(request: NamedServiceRequest) -> list[str] | None:
    """Extract the batch ref list from an object.get request, or None if single.

    Returns a de-duplicated, order-preserving list of ref strings when
    filters.refs (or filters.object_refs) is present (even if empty), else None.
    """
    filters = request.filters or {}
    raw = None
    for key in _BATCH_GET_KEYS:
        if key in filters and filters.get(key) is not None:
            raw = filters.get(key)
            break
    if raw is None:
        return None
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    seen: set[str] = set()
    refs: list[str] = []
    for value in values:
        ref = str(value or "").strip()
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def named_service_provider(
    *,
    provider_id: str,
    bundle_id: str | None = None,
    namespace: str | None = None,
    namespaces: Sequence[str] = (),
    refs: Sequence[str] = (),
    object_kinds: Sequence[str] = (),
    search_scopes: Sequence[Any] = (),
    operations: Mapping[str, Any] | None = None,
    label: str | None = None,
    description: str | None = None,
    intro: str = "",
    metadata: Mapping[str, Any] | None = None,
):
    """Attach named service provider metadata to a class or factory.

    Runtime loader integration is intentionally separate. This decorator only
    provides stable metadata that a bundle or registry can inspect.
    """

    spec = NamedServiceProviderSpec(
        provider_id=provider_id,
        bundle_id=bundle_id,
        namespace=namespace,
        namespaces=tuple(namespaces or ()),
        refs=tuple(refs or ()),
        object_kinds=tuple(object_kinds or ()),
        search_scopes=tuple(search_scopes or ()),
        operations=dict(operations or build_default_operations()),
        label=label,
        description=description,
        intro=str(intro or "").strip(),
        metadata=dict(metadata or {}),
    )

    def decorate(target):
        setattr(target, "__kdcube_named_service_provider__", spec)
        return target

    return decorate


class NamedServiceProvider:
    """Base class for async named service providers.

    Providers may override ``dispatch`` directly, or implement async methods
    named after operations, for example ``object_search`` or ``object_action``.
    """

    spec: NamedServiceProviderSpec
    schema_projection_index: Mapping[str, Any] | None = None

    def __init__(self, spec: NamedServiceProviderSpec | None = None) -> None:
        inferred = spec or getattr(self, "__kdcube_named_service_provider__", None)
        if inferred is None:
            inferred = NamedServiceProviderSpec(
                provider_id=self.__class__.__name__,
                operations=build_default_operations(),
            )
        self.spec = inferred
        self._schema_search_storage_root: Path | None = None
        self._schema_search_model_service: Any = None
        self._schema_search_dim = 1536
        self._schema_search_vector_backend = DEFAULT_SCHEMA_VECTOR_BACKEND
        self._schema_search_indexes: dict[str, SchemaCatalogSearchIndex] = {}

    async def dispatch(
        self,
        ctx: NamedServiceContext,
        request: NamedServiceRequest,
    ) -> NamedServiceResponse | Mapping[str, Any]:
        # Uniform batch get: object.get + a list of refs -> fan out to single get.
        if request.operation == "object.get":
            batch_refs = batch_get_refs(request)
            if batch_refs is not None:
                return await self._dispatch_batch_get(ctx, request, batch_refs)
        method_name = request.operation.replace(".", "_")
        method = getattr(self, method_name, None)
        if method is None:
            LOGGER.warning(
                "Named-service provider operation missing: provider=%s namespace=%s operation=%s object_ref=%s",
                self.spec.provider_id,
                request.namespace,
                request.operation,
                request.object_ref or "",
            )
            return NamedServiceResponse.error_response(
                code="named_service_operation_not_supported",
                message=f"Provider does not implement {request.operation}",
                status=404,
                provider=self.provider_identity(),
                namespace=request.namespace,
                object_ref=request.object_ref,
            )
        LOGGER.info(
            "Named-service provider dispatch start: provider=%s namespace=%s operation=%s object_ref=%s user_type=%s user_id=%s",
            self.spec.provider_id,
            request.namespace,
            request.operation,
            request.object_ref or "",
            ctx.user_type or "",
            ctx.user_id or "",
        )
        result = method(ctx, request)
        if not hasattr(result, "__await__"):
            raise TypeError(f"Named service provider method {method_name} must be async")
        response = await result
        if (
            request.operation in {"provider.about", "object.schema"}
            and self.schema_projection_index
        ):
            response = await project_schema_response_async(
                response=response,
                request=request,
                index=self.schema_projection_index,
                object_kind_from_ref=self.schema_object_kind_from_ref,
                search_index=self._schema_search_index_for(
                    request.namespace or self.spec.namespace or "default"
                ),
            )
        if isinstance(response, NamedServiceResponse):
            ok = response.ok
        elif isinstance(response, Mapping):
            ok = bool(response.get("ok"))
        else:
            ok = True
        LOGGER.info(
            "Named-service provider dispatch complete: provider=%s namespace=%s operation=%s object_ref=%s ok=%s",
            self.spec.provider_id,
            request.namespace,
            request.operation,
            request.object_ref or "",
            ok,
        )
        return response

    def schema_object_kind_from_ref(self, object_ref: str) -> str | None:
        """Return the provider-owned object kind for one opaque ref, when known."""

        del object_ref
        return None

    def configure_schema_catalog_search(
        self,
        *,
        storage_root: str | Path | None,
        model_service: Any,
        dim: int = 1536,
        vector_backend: str = DEFAULT_SCHEMA_VECTOR_BACKEND,
    ) -> None:
        """Bind optional shared hybrid search for this provider's capabilities."""

        if not self.schema_projection_index:
            return
        if not storage_root or not callable(getattr(model_service, "embed_texts", None)):
            self._schema_search_storage_root = None
            self._schema_search_model_service = None
            self._schema_search_indexes.clear()
            return
        next_storage_root = Path(storage_root)
        next_dim = max(1, int(dim or 1536))
        next_vector_backend = normalize_schema_vector_backend(vector_backend)
        changed = (
            self._schema_search_storage_root != next_storage_root
            or self._schema_search_model_service is not model_service
            or self._schema_search_dim != next_dim
            or self._schema_search_vector_backend != next_vector_backend
        )
        self._schema_search_storage_root = next_storage_root
        self._schema_search_model_service = model_service
        self._schema_search_dim = next_dim
        self._schema_search_vector_backend = next_vector_backend
        if changed:
            self._schema_search_indexes.clear()

    def _schema_search_index_for(
        self,
        namespace: str,
    ) -> SchemaCatalogSearchIndex | None:
        if self._schema_search_storage_root is None:
            return None
        if self._schema_search_model_service is None:
            return None
        namespace_key = str(
            namespace or self.spec.namespace or "default"
        ).strip().lower()
        embedding_profile = schema_search_embedding_profile(
            self._schema_search_model_service,
            dim=self._schema_search_dim,
            vector_backend=self._schema_search_vector_backend,
        )
        identity = f"{self.spec.provider_id}.{namespace_key}"
        db_path = schema_search_index_path(
            self._schema_search_storage_root,
            identity,
            embedding_profile=embedding_profile,
        )
        key = str(db_path)
        if key not in self._schema_search_indexes:
            namespace_dir = str(db_path.parent) + "/"
            self._schema_search_indexes = {
                cached_path: cached_index
                for cached_path, cached_index in self._schema_search_indexes.items()
                if not cached_path.startswith(namespace_dir)
            }
            self._schema_search_indexes[key] = SchemaCatalogSearchIndex(
                db_path=db_path,
                model_service=self._schema_search_model_service,
                dim=self._schema_search_dim,
                vector_backend=self._schema_search_vector_backend,
            )
        return self._schema_search_indexes[key]

    async def ensure_schema_catalog_index(
        self,
        *,
        context: NamedServiceContext | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        """Prepare shared capability indexes; request-time search also heals lazily."""

        if not self.schema_projection_index:
            return []
        method = getattr(self, "object_schema", None)
        if method is None:
            return []
        namespaces = [str(namespace or "").strip()] if namespace else list(
            self.spec.namespaces or ((self.spec.namespace,) if self.spec.namespace else ())
        )
        if not namespaces:
            namespaces = ["default"]
        results: list[dict[str, Any]] = []
        for selected_namespace in namespaces:
            search_index = self._schema_search_index_for(selected_namespace)
            if search_index is None:
                continue
            request = NamedServiceRequest(
                operation=OBJECT_SCHEMA,
                namespace=(
                    selected_namespace if selected_namespace != "default" else None
                ),
                schema_view="full",
            )
            raw = method(context or NamedServiceContext(), request)
            if not hasattr(raw, "__await__"):
                raise TypeError("Named service provider object_schema must be async")
            response = NamedServiceResponse.coerce(await raw)
            schema = response.extra.get("schema") if response.ok else None
            if not isinstance(schema, Mapping):
                continue
            tree = build_schema_tree(schema, self.schema_projection_index)
            result = await search_index.ensure(tree)
            results.append(
                {
                    **result,
                    "namespace": selected_namespace,
                    "provider_id": self.spec.provider_id,
                }
            )
        return results

    async def _dispatch_batch_get(
        self,
        ctx: NamedServiceContext,
        request: NamedServiceRequest,
        refs: list[str],
    ) -> NamedServiceResponse:
        object_get = getattr(self, "object_get", None)
        if object_get is None:
            return NamedServiceResponse.error_response(
                code="named_service_operation_not_supported",
                message="Provider does not implement object.get",
                status=404,
                provider=self.provider_identity(),
                namespace=request.namespace,
            )
        if not refs:
            return NamedServiceResponse.error_response(
                code="object_refs_required",
                message="Batch object.get requires filters.refs: a non-empty list of object refs.",
                status=400,
                provider=self.provider_identity(),
                namespace=request.namespace,
            )
        capped = refs[:BATCH_GET_MAX]
        base_filters = {k: v for k, v in (request.filters or {}).items() if k not in _BATCH_GET_KEYS}
        items: list[Any] = []
        missing: list[str] = []
        for ref in capped:
            sub_request = dataclasses.replace(
                request, object_ref=ref, object_id=None, filters=dict(base_filters),
            )
            resp = await object_get(ctx, sub_request)
            obj: Any = None
            if isinstance(resp, NamedServiceResponse):
                obj = resp.object or None if resp.ok else None
            elif isinstance(resp, Mapping) and resp.get("ok"):
                ret = resp.get("ret")
                obj = ret.get("object") if isinstance(ret, Mapping) else None
            if obj:
                items.append(obj)
            else:
                missing.append(ref)
        extra: dict[str, Any] = {"count": len(items), "requested": len(capped)}
        if missing:
            extra["missing"] = missing
        if len(refs) > BATCH_GET_MAX:
            extra["truncated"] = len(refs) - BATCH_GET_MAX
        LOGGER.info(
            "Named-service batch get: provider=%s namespace=%s requested=%s returned=%s missing=%s",
            self.spec.provider_id,
            request.namespace,
            len(capped),
            len(items),
            len(missing),
        )
        return NamedServiceResponse.ok_response(
            provider=self.provider_identity(),
            namespace=request.namespace,
            items=items,
            extra=extra,
        )

    def provider_identity(self) -> dict[str, Any]:
        return {
            "provider_id": self.spec.provider_id,
            "bundle_id": self.spec.bundle_id,
        }
