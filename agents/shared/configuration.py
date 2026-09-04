"""YAML-owned agent configuration shared by the direct examples."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfiguredTool:
    id: str
    enabled: bool = True
    runtime: str = "local"


@dataclass(frozen=True)
class ConfiguredSkills:
    root: Path | None
    enabled: tuple[str, ...]


def _agent(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("agent")
    if not isinstance(value, Mapping):
        raise ValueError("configuration section 'agent' must be a mapping")
    return value


def agent_instructions(config: Mapping[str, Any], *, fallback: str) -> str:
    value = str(_agent(config).get("instructions") or "").strip()
    return value or fallback


def configured_tools(config: Mapping[str, Any]) -> tuple[ConfiguredTool, ...]:
    raw_tools = _agent(config).get("tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        raise ValueError("agent.tools must be a list of tool mappings")

    tools: list[ConfiguredTool] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_tools):
        if not isinstance(raw, Mapping):
            raise ValueError(f"agent.tools[{index}] must be a mapping")
        tool_id = str(raw.get("id") or "").strip()
        if not tool_id:
            raise ValueError(f"agent.tools[{index}].id is required")
        if tool_id in seen:
            raise ValueError(f"agent.tools contains duplicate id {tool_id!r}")
        seen.add(tool_id)
        tools.append(
            ConfiguredTool(
                id=tool_id,
                enabled=bool(raw.get("enabled", True)),
                runtime=str(raw.get("runtime") or "local").strip().lower(),
            )
        )
    if not tools:
        raise ValueError("agent.tools must declare at least one tool")
    return tuple(tools)


def enabled_tool_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(tool.id for tool in configured_tools(config) if tool.enabled)


def require_supported_tools(
    tools: Sequence[ConfiguredTool],
    *,
    supported: set[str],
    adapter: str,
) -> None:
    unknown = sorted(tool.id for tool in tools if tool.id not in supported)
    if unknown:
        raise ValueError(f"{adapter} does not expose configured tools: {', '.join(unknown)}")


def configured_skills(config: Mapping[str, Any], *, config_path: Path) -> ConfiguredSkills:
    raw = _agent(config).get("skills") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("agent.skills must be a mapping")
    enabled_raw = raw.get("enabled") or []
    if not isinstance(enabled_raw, Sequence) or isinstance(enabled_raw, (str, bytes)):
        raise ValueError("agent.skills.enabled must be a list")
    enabled = tuple(dict.fromkeys(str(item).strip() for item in enabled_raw if str(item).strip()))
    root_raw = str(raw.get("root") or "").strip()
    root = None
    if root_raw:
        candidate = Path(root_raw).expanduser()
        root = (
            candidate.resolve()
            if candidate.is_absolute()
            else (config_path.parent / candidate).resolve()
        )
        if not root.is_dir():
            raise ValueError(f"agent.skills.root does not exist: {root}")
    return ConfiguredSkills(root=root, enabled=enabled)


def activate_configured_skills(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    consumers: Sequence[str],
) -> tuple[Any, ConfiguredSkills]:
    from kdcube_ai_app.apps.chat.sdk.skills.skills_registry import (
        SkillsSubsystem,
        set_active_skills_subsystem,
    )

    selection = configured_skills(config, config_path=config_path)
    policy = {"enabled": list(selection.enabled)} if selection.enabled else {"disabled": ["*"]}
    visibility = {str(consumer): dict(policy) for consumer in consumers}
    subsystem = SkillsSubsystem(
        descriptor={
            "custom_skills_root": str(selection.root) if selection.root else None,
            "agents_config": visibility,
        },
        bundle_root=config_path.parent,
    )
    set_active_skills_subsystem(subsystem)
    registry = subsystem.get_skill_registry()
    missing = sorted(skill_id for skill_id in selection.enabled if skill_id not in registry)
    if missing:
        raise ValueError(f"agent.skills.enabled contains unknown skills: {', '.join(missing)}")
    return subsystem, selection


def verify_docker_image(profile: Mapping[str, Any]) -> str:
    image = str(profile.get("image") or "").strip()
    if not image:
        raise ValueError("isolated execution profile has no image")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required when the isolated code-execution tool is enabled")
    result = subprocess.run(
        [docker, "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"isolated execution image {image!r} is not available; "
            "build it before enabling code execution"
        )
    return image


__all__ = [
    "ConfiguredSkills",
    "ConfiguredTool",
    "activate_configured_skills",
    "agent_instructions",
    "configured_skills",
    "configured_tools",
    "enabled_tool_ids",
    "require_supported_tools",
    "verify_docker_image",
]
