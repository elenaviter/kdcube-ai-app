"""Resolve configured tools into bindings for a directly hosted Native agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (
    ConfiguredTool,
    configured_tools,
    require_supported_tools,
)


@dataclass(frozen=True)
class NativeToolSource:
    """Trusted Python source and exposed names for one tool alias."""

    tool_names: tuple[str, ...]
    module: str | None = None
    path: str | Path | None = None
    discovery: Literal["plain", "semantic_kernel"] = "plain"

    def __post_init__(self) -> None:
        names = tuple(
            dict.fromkeys(str(name or "").strip() for name in self.tool_names)
        )
        if not names or any(not name or "." in name for name in names):
            raise ValueError("Native tool source names must be non-empty unqualified names")
        module = str(self.module or "").strip() or None
        path = Path(self.path).expanduser().resolve() if self.path is not None else None
        if (module is None) == (path is None):
            raise ValueError("Native tool source requires exactly one of module or path")
        if path is not None and not path.is_file():
            raise ValueError(f"Native tool source path does not exist: {path}")
        if self.discovery not in {"plain", "semantic_kernel"}:
            raise ValueError(
                "Native tool source discovery must be plain or semantic_kernel"
            )
        object.__setattr__(self, "tool_names", names)
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "path", path)

    def tool_spec(self, *, alias: str) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "alias": alias,
            "use_sk": self.discovery == "semantic_kernel",
        }
        if self.module is not None:
            spec["module"] = self.module
        else:
            spec["ref"] = str(self.path)
        return spec


@dataclass(frozen=True)
class NativeToolBindings:
    """Concrete ToolSubsystem inputs selected by the agent configuration."""

    configured: tuple[ConfiguredTool, ...]
    tool_specs: tuple[dict[str, Any], ...]
    tool_runtime: dict[str, str]
    allowed_tool_names_by_alias: dict[str, list[str]]

    @property
    def enabled_ids(self) -> tuple[str, ...]:
        return tuple(tool.id for tool in self.configured if tool.enabled)

    @property
    def allowed_plugins(self) -> list[str]:
        return list(self.allowed_tool_names_by_alias)


def resolve_native_tool_bindings(
    config: Mapping[str, Any],
    *,
    sources: Mapping[str, NativeToolSource],
    adapter_name: str,
) -> NativeToolBindings:
    """Resolve YAML-selected IDs against trusted, host-supplied Python sources."""
    normalized_sources: dict[str, NativeToolSource] = {}
    for raw_alias, source in sources.items():
        alias = str(raw_alias or "").strip()
        if not alias or "." in alias:
            raise ValueError("Native tool source aliases must be non-empty and unqualified")
        if not isinstance(source, NativeToolSource):
            raise TypeError(f"Native tool source {alias!r} must be NativeToolSource")
        normalized_sources[alias] = source

    configured = configured_tools(config)
    supported = {
        f"{alias}.{tool_name}"
        for alias, source in normalized_sources.items()
        for tool_name in source.tool_names
    }
    require_supported_tools(
        configured,
        supported=supported,
        adapter=adapter_name,
    )

    enabled = tuple(tool for tool in configured if tool.enabled)
    tool_specs: list[dict[str, Any]] = []
    tool_runtime: dict[str, str] = {}
    allowed: dict[str, list[str]] = {}
    for tool in enabled:
        alias, tool_name = tool.id.split(".", 1)
        if alias not in allowed:
            allowed[alias] = []
            tool_specs.append(normalized_sources[alias].tool_spec(alias=alias))
        allowed[alias].append(tool_name)
        tool_runtime[tool.id] = tool.runtime

    return NativeToolBindings(
        configured=configured,
        tool_specs=tuple(tool_specs),
        tool_runtime=tool_runtime,
        allowed_tool_names_by_alias=allowed,
    )


__all__ = [
    "NativeToolBindings",
    "NativeToolSource",
    "resolve_native_tool_bindings",
]
