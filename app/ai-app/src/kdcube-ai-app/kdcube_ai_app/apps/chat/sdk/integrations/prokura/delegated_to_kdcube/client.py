# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube configuration bindings for the Prokura connected-account client."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kdcube_ai_app.apps.chat.sdk import config as sdk_config
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_to_kdcube.store import (
    DelegatedToKdcubeStore,
)


_core = import_module("prokura.delegated_to_kdcube.client")


async def _kdcube_bundle_props_loader(redis: Any, **kwargs: Any) -> Any:
    from kdcube_ai_app.infra.plugin.bundle_store import get_bundle_props

    return await get_bundle_props(redis, **kwargs)


async def _kdcube_secret_loader(secret_ref: str, **kwargs: Any) -> Any:
    return await sdk_config.get_secret(secret_ref, **kwargs)


class DelegatedToKdcubeClient(_core.DelegatedToKdcubeClient):
    @classmethod
    def from_user(
        cls,
        *,
        user_id: str,
        config: Any,
        bundle_id: str = "",
        store: Any | None = None,
        store_factory: Any | None = None,
        client_secret_resolver: Any = None,
    ) -> "DelegatedToKdcubeClient":
        return super().from_user(
            user_id=user_id,
            config=config,
            bundle_id=bundle_id,
            store=store,
            store_factory=store_factory or DelegatedToKdcubeStore,
            client_secret_resolver=client_secret_resolver,
        )

    @classmethod
    def from_entrypoint(
        cls,
        entrypoint: Any,
        *,
        user_id: str,
        store: Any | None = None,
    ) -> "DelegatedToKdcubeClient":
        return super().from_entrypoint(
            entrypoint,
            user_id=user_id,
            store=store,
            store_factory=DelegatedToKdcubeStore,
        )

    @classmethod
    async def from_connection_hub(
        cls,
        entrypoint: Any,
        *,
        user_id: str,
        connection_hub_bundle_id: str | None = None,
        tenant: str | None = None,
        project: str | None = None,
        store: Any | None = None,
    ) -> "DelegatedToKdcubeClient":
        return await super().from_connection_hub(
            entrypoint,
            user_id=user_id,
            connection_hub_bundle_id=connection_hub_bundle_id,
            tenant=tenant,
            project=project,
            store=store,
            store_factory=DelegatedToKdcubeStore,
            bundle_props_loader=_kdcube_bundle_props_loader,
            secret_loader=_kdcube_secret_loader,
        )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = ["DelegatedToKdcubeClient"]
