# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube client binding for Prokura connected-account preflight."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable

from kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_to_kdcube.client import (
    DelegatedToKdcubeClient,
)


_core = import_module("prokura.delegated_to_kdcube.preflight")

CONSENT_NEEDED_CODE = _core.CONSENT_NEEDED_CODE
PREFLIGHT_SCHEMA = _core.PREFLIGHT_SCHEMA
connected_account_consent_payload = _core.connected_account_consent_payload
consent_action_message = _core.consent_action_message
unavailable_tools_by_provider = _core.unavailable_tools_by_provider
unavailable_tools_message = _core.unavailable_tools_message


async def _kdcube_client_factory(entrypoint: Any, **kwargs: Any) -> Any:
    return await DelegatedToKdcubeClient.from_connection_hub(entrypoint, **kwargs)


async def preflight_tool_claim_policies(
    *,
    entrypoint: Any,
    user_id: str,
    policies: Iterable[Any],
    tenant: str = "",
    project: str = "",
    connection_hub_bundle_id: str = _core.DEFAULT_CONNECTION_HUB_BUNDLE_ID,
) -> dict[str, Any]:
    return await _core.preflight_tool_claim_policies(
        entrypoint=entrypoint,
        user_id=user_id,
        policies=policies,
        tenant=tenant,
        project=project,
        connection_hub_bundle_id=connection_hub_bundle_id,
        client_factory=_kdcube_client_factory,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = list(_core.__all__)
