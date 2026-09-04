"""Translate the native example's YAML tool inventory into SDK bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.shared.configuration import (
    ConfiguredTool,
    configured_tools,
    require_supported_tools,
)


EXEC_TOOL_ID = "exec_tools.execute_code_python"
SUPPORTED_TOOL_IDS = {
    "demo.web_search",
    "demo.create_briefing",
    EXEC_TOOL_ID,
}


@dataclass(frozen=True)
class NativeToolPlan:
    configured: tuple[ConfiguredTool, ...]
    tools_specs: tuple[dict[str, Any], ...]
    tool_runtime: dict[str, str]
    allowed_tool_names_by_alias: dict[str, list[str]]
    exec_runtime: dict[str, Any] | None

    @property
    def enabled_ids(self) -> tuple[str, ...]:
        return tuple(tool.id for tool in self.configured if tool.enabled)

    @property
    def allowed_plugins(self) -> list[str]:
        return sorted(self.allowed_tool_names_by_alias)


def build_native_tool_plan(
    config: dict[str, Any],
    *,
    tools_file: Path,
    platform_exec_runtime: dict[str, Any],
) -> NativeToolPlan:
    configured = configured_tools(config)
    require_supported_tools(
        configured,
        supported=SUPPORTED_TOOL_IDS,
        adapter="native direct example",
    )
    enabled = tuple(tool for tool in configured if tool.enabled)
    aliases = {tool.id.split(".", 1)[0] for tool in enabled}
    specs: list[dict[str, Any]] = []
    if "demo" in aliases:
        specs.append({"ref": str(tools_file), "alias": "demo", "use_sk": False})
    if "exec_tools" in aliases:
        specs.append(
            {
                "module": "kdcube_ai_app.apps.chat.sdk.tools.exec_tools",
                "alias": "exec_tools",
                "use_sk": True,
            }
        )

    allowed: dict[str, list[str]] = {}
    for tool in enabled:
        alias, name = tool.id.split(".", 1)
        allowed.setdefault(alias, []).append(name)

    exec_tool = next((tool for tool in enabled if tool.id == EXEC_TOOL_ID), None)
    if exec_tool is not None and exec_tool.runtime != "docker":
        raise ValueError(f"{EXEC_TOOL_ID} must use runtime: docker in this example")
    return NativeToolPlan(
        configured=configured,
        tools_specs=tuple(specs),
        tool_runtime={tool.id: tool.runtime for tool in enabled},
        allowed_tool_names_by_alias=allowed,
        exec_runtime=dict(platform_exec_runtime) if exec_tool is not None else None,
    )


__all__ = ["EXEC_TOOL_ID", "NativeToolPlan", "build_native_tool_plan"]
