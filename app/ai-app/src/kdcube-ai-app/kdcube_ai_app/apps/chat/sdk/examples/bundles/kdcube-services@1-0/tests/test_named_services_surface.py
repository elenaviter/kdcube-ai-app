# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.dynamic_module_loader import (
    load_dynamic_module_for_path,
)


BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def _surface_module():
    _name, module = load_dynamic_module_for_path(
        BUNDLE_ROOT / "surfaces" / "mcp" / "named_services.py"
    )
    return module


@pytest.mark.asyncio
async def test_named_services_get_exposes_provider_filters() -> None:
    module = _surface_module()
    app = module.build_named_services_mcp_app(
        name="KDCube named services",
        config_factory=dict,
        tenant_factory=lambda: "tenant",
        project_factory=lambda: "project",
        request=None,
        bridge_factory=lambda **_kwargs: None,
    )

    schemas = {tool.name: tool.input_schema for tool in await app.list_tools()}
    get_schema = schemas["named_services_get"]
    assert "filters_json" in get_schema["properties"]
    assert "ranges" in get_schema["properties"]["filters_json"]["description"]


@pytest.mark.asyncio
async def test_search_and_generic_call_expose_pagination_cursor() -> None:
    module = _surface_module()
    app = module.build_named_services_mcp_app(
        name="KDCube named services",
        config_factory=dict,
        tenant_factory=lambda: "tenant",
        project_factory=lambda: "project",
        request=None,
        bridge_factory=lambda **_kwargs: None,
    )

    schemas = {tool.name: tool.input_schema for tool in await app.list_tools()}

    assert "cursor" in schemas["named_services_search"]["properties"]
    assert "cursor" in schemas["named_services_call"]["properties"]
