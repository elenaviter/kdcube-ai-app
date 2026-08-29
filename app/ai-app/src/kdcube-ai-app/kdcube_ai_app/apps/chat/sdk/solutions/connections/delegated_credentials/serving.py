# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube storage and descriptor bindings for Prokura serving readers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_core = import_module("prokura.delegated_credentials.serving")

SERVING_RESOLVERS_ATTR = _core.SERVING_RESOLVERS_ATTR
DelegatedServingResolvers = _core.DelegatedServingResolvers
delegated_serving_resolvers = _core.delegated_serving_resolvers


def connection_hub_app_id() -> str:
    from kdcube_ai_app.apps.chat.sdk.solutions.connections.authentication_surface import (
        connection_hub_app_id as resolve,
    )

    return resolve()


def _kdcube_storage_root_resolver(**kwargs: Any) -> Any:
    from kdcube_ai_app.infra.plugin.bundle_storage import bundle_storage_dir

    return bundle_storage_dir(**kwargs)


async def _kdcube_bundle_props_loader(**kwargs: Any) -> Any:
    from kdcube_ai_app.infra.plugin.bundle_store import (
        get_bundle_props_from_authority,
    )

    return await get_bundle_props_from_authority(**kwargs)


async def build_delegated_serving_resolvers(
    *,
    redis: Any,
    tenant: str,
    project: str,
    bundle_id: str = "",
) -> Any:
    return await _core.build_delegated_serving_resolvers(
        redis=redis,
        tenant=tenant,
        project=project,
        bundle_id=bundle_id,
        app_id_resolver=connection_hub_app_id,
        storage_root_resolver=_kdcube_storage_root_resolver,
        bundle_props_loader=_kdcube_bundle_props_loader,
    )


async def install_delegated_serving_resolvers(
    app: Any,
    *,
    redis: Any,
    tenant: str,
    project: str,
) -> bool:
    return await _core.install_delegated_serving_resolvers(
        app,
        redis=redis,
        tenant=tenant,
        project=project,
        app_id_resolver=connection_hub_app_id,
        storage_root_resolver=_kdcube_storage_root_resolver,
        bundle_props_loader=_kdcube_bundle_props_loader,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = [
    "SERVING_RESOLVERS_ATTR",
    "DelegatedServingResolvers",
    "build_delegated_serving_resolvers",
    "connection_hub_app_id",
    "delegated_serving_resolvers",
    "install_delegated_serving_resolvers",
]
