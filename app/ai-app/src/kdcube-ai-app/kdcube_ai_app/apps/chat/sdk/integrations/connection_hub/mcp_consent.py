# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube chat bindings for Connection Hub MCP consent contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping


_core = import_module("connection_hub.mcp_consent")


async def _kdcube_demand_announcer(**kwargs: Any) -> bool:
    from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_to_kdcube.consent_demand import (
        announce_consent_demand,
    )

    return bool(await announce_consent_demand(**kwargs))


async def _kdcube_event_emitter(payload: Mapping[str, Any]) -> bool:
    from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import get_comm

    communicator = get_comm()
    event = getattr(communicator, "event", None) if communicator is not None else None
    if not callable(event):
        return False
    result = event(
        agent="connection-hub",
        type="chat.step",
        route="chat.step",
        title="Access consent needed",
        step="delegated_to_kdcube.consent",
        data=dict(payload or {}),
        status="completed",
        broadcast=False,
    )
    if hasattr(result, "__await__"):
        await result
    return True


async def announce_agent_consent(consent: Any) -> None:
    await _core.announce_agent_consent(
        consent,
        demand_announcer=_kdcube_demand_announcer,
        event_emitter=_kdcube_event_emitter,
    )


def __getattr__(name: str) -> Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = sorted(name for name in dir(_core) if not name.startswith("_"))
