# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube chat bindings for Prokura MCP-result handling."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List


_core = import_module("prokura.mcp_result")

DELEGATED_CONSENT_REQUIRED = _core.DELEGATED_CONSENT_REQUIRED
NEEDS_CONNECTED_ACCOUNT_CONSENT = _core.NEEDS_CONNECTED_ACCOUNT_CONSENT


async def _connected_consent_announcer(
    payload: Dict[str, Any],
    *,
    namespace: str,
    tool_name: str,
) -> Any:
    from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.consent import (
        raise_named_service_consent_demand,
    )

    return await raise_named_service_consent_demand(
        payload,
        namespace=namespace,
        tool_name=tool_name,
    )


async def _agent_consent_announcer(consent: Any) -> None:
    from kdcube_ai_app.apps.chat.sdk.integrations.prokura.mcp_consent import (
        announce_agent_consent,
    )

    await announce_agent_consent(consent)


async def _file_deliverer(payload: Dict[str, Any]) -> Dict[str, Any]:
    from kdcube_ai_app.apps.chat.sdk.solutions.widgets.send_to_user import (
        deliver_result_files,
    )

    return await deliver_result_files(payload)


async def announce_result_consent(
    parsed: Dict[str, Any],
) -> Dict[str, Any] | None:
    return await _core.announce_result_consent(
        parsed,
        connected_consent_announcer=_connected_consent_announcer,
        agent_consent_announcer=_agent_consent_announcer,
    )


def bind_chat_result_handling(tools: List[Any]) -> List[Any]:
    return _core.bind_chat_result_handling(
        tools,
        connected_consent_announcer=_connected_consent_announcer,
        agent_consent_announcer=_agent_consent_announcer,
        file_deliverer=_file_deliverer,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = ["announce_result_consent", "bind_chat_result_handling"]
