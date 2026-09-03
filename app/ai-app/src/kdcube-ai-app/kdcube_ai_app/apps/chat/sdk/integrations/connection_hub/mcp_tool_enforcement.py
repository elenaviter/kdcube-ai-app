# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube runtime bindings for Connection Hub plain-MCP enforcement."""

from __future__ import annotations

import logging
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    resolve_connected_account_claim,
)
from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import (
    get_current_request_context,
    get_current_user_identity,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connector_app_resolution import (
    resolve_connector_app_id,
    set_service_connector_apps,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.consent_denial import (
    connect_first_denial_for_identity,
)
from connection_hub.delegated_credentials.credential_view import (
    delegated_credential_view,
)


logger = logging.getLogger(__name__)
_core = import_module("connection_hub.mcp_tool_enforcement")


def bind_service_connector_apps_from_config(
    config: Mapping[str, Any] | None,
) -> None:
    _core.bind_service_connector_apps_from_config(
        config,
        connector_apps_binder=set_service_connector_apps,
    )


def _resolution_source(request: Any) -> Mapping[str, Any]:
    registry: dict[str, Any] = {}
    try:
        from kdcube_ai_app.apps.chat.sdk.config import get_settings
        from kdcube_ai_app.infra.redis.client import get_async_redis_client

        registry["redis"] = get_async_redis_client(get_settings().REDIS_URL)
    except Exception:
        logger.debug(
            "[mcp-tool-enforcement] settings redis unavailable",
            exc_info=True,
        )
    context = get_current_request_context()
    if context is not None:
        registry["comm_context"] = context
    del request
    return {"_TOOL_SUBSYSTEM": SimpleNamespace(registry=registry)}


async def enforce_tool_requirements(
    request: Any,
    *,
    tool_name: str,
    operation: str,
    requirements: Sequence[Mapping[str, Any]],
    account_id: str = "",
    tenant: str = "",
    project: str = "",
    hub_bundle_id: str = "connection-hub@1-0",
) -> dict[str, Any] | None:
    return await _core.enforce_tool_requirements(
        request,
        tool_name=tool_name,
        operation=operation,
        requirements=requirements,
        account_id=account_id,
        tenant=tenant,
        project=project,
        hub_bundle_id=hub_bundle_id,
        identity=get_current_user_identity() or {},
        resolution_source=_resolution_source(request),
        connector_app_resolver=resolve_connector_app_id,
        credential_resolver=resolve_connected_account_claim,
        credential_view_resolver=delegated_credential_view,
        connect_first_denial_builder=connect_first_denial_for_identity,
    )


async def resolve_tool_requirements(
    request: Any,
    *,
    tool_name: str,
    operation: str,
    requirements: Sequence[Mapping[str, Any]],
    account_id: str = "",
    tenant: str = "",
    project: str = "",
    hub_bundle_id: str = "connection-hub@1-0",
    accounts_lister: Any = None,
) -> Any:
    """The form that also answers WHICH provider account satisfied each
    requirement (a ``ToolRequirementResolution``): what a tool declared with
    an ``any_of`` group needs in order to route. ``accounts_lister`` is the
    optional ``provider_id -> connected accounts`` coroutine that lets a
    cross-provider account_required carry labels."""
    return await _core.resolve_tool_requirements(
        request,
        tool_name=tool_name,
        operation=operation,
        requirements=requirements,
        account_id=account_id,
        tenant=tenant,
        project=project,
        hub_bundle_id=hub_bundle_id,
        identity=get_current_user_identity() or {},
        resolution_source=_resolution_source(request),
        connector_app_resolver=resolve_connector_app_id,
        credential_resolver=resolve_connected_account_claim,
        credential_view_resolver=delegated_credential_view,
        connect_first_denial_builder=connect_first_denial_for_identity,
        accounts_lister=accounts_lister,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = [
    "bind_service_connector_apps_from_config",
    "enforce_tool_requirements",
    "resolve_tool_requirements",
]
