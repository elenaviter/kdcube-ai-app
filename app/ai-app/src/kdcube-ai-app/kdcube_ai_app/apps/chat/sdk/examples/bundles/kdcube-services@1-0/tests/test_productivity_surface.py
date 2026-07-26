# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The productivity surface is the reference PURE-MCP door: every tool
declares its connected-account requirements (ToolClaimPolicy shape) and the
surface builds with exactly the declared tool roster."""

from __future__ import annotations

from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.dynamic_module_loader import load_dynamic_module_for_path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def _surface_module():
    _name, module = load_dynamic_module_for_path(
        BUNDLE_ROOT / "surfaces" / "mcp" / "productivity.py"
    )
    return module


def test_every_tool_declares_provider_claims():
    module = _surface_module()
    declared = {
        name: config
        for name, config in module.PRODUCTIVITY_TOOLS.items()
    }
    assert set(declared) == {
        "productivity_slack_search",
        "productivity_mail_search",
        "productivity_mail_get",
    }
    expectations = {
        "productivity_slack_search": ("slack", ["slack:search"]),
        "productivity_mail_search": ("google", ["gmail:read"]),
        "productivity_mail_get": ("google", ["gmail:read"]),
    }
    for name, (provider_id, claims) in expectations.items():
        requirements = module.tool_requirements(name)
        assert requirements, f"{name} declares no requirements"
        assert requirements[0]["provider_id"] == provider_id
        assert requirements[0]["claims"] == claims


@pytest.mark.asyncio
async def test_surface_builds_with_declared_tool_roster():
    module = _surface_module()
    app = module.build_productivity_mcp_app(
        name="KDCube productivity",
        config_factory=lambda: {"connector_apps": {"slack": "slack-demo", "google": "gmail"}},
        tenant_factory=lambda: "t",
        project_factory=lambda: "p",
        request=None,
    )
    tools = {tool.name for tool in await app.list_tools()}
    assert tools == {
        "productivity_slack_search",
        "productivity_mail_search",
        "productivity_mail_get",
    }
