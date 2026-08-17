from __future__ import annotations

from contextlib import asynccontextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.sdk.frameworks.langchain import mcp as mcp_binding


class _FakeClient:
    instructions = "Use echo for exact repetition."

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(
            name="echo",
            description="Return the supplied value.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            output_schema=None,
        )])

    async def call_tool(self, name, params):
        assert name == "echo"
        return SimpleNamespace(
            structured_content={"value": params["value"]},
            content=[],
            is_error=False,
        )


class _FakeJsonTextResultClient(_FakeClient):
    async def call_tool(self, name, params):
        assert name == "echo"
        return SimpleNamespace(
            structured_content=None,
            content=[
                SimpleNamespace(
                    type="text",
                    text=(
                        '{"ok": false, '
                        '"error": {"code": "file_path_required", '
                        '"message": "Pass the conv:fi artifact path as file_path."}, '
                        '"status": 400}'
                    ),
                )
            ],
            is_error=True,
        )


@pytest.mark.asyncio
async def test_loader_uses_sdk_tools_and_records_errors_per_server(monkeypatch) -> None:
    monkeypatch.setattr(mcp_binding, "mcp_adapters_available", lambda: True)

    @asynccontextmanager
    async def open_entry(entry):
        if entry["url"].endswith("denied"):
            raise PermissionError("403 Forbidden")
        yield _FakeClient()

    monkeypatch.setattr(mcp_binding, "_open_server_entry", open_entry)
    error_sink = {}
    tools = await mcp_binding.load_mcp_tools_from_server_map(
        {
            "good": {"url": "https://mcp.example/good"},
            "denied": {"url": "https://mcp.example/denied"},
        },
        error_sink=error_sink,
    )

    assert [tool.name for tool in tools] == ["echo"]
    assert await tools[0].ainvoke({"value": "hello"}) == {"value": "hello"}
    assert set(error_sink["_server_errors"]) == {"denied"}
    assert mcp_binding.load_error_looks_like_denial(
        error_sink["_server_errors"]["denied"]
    )


@pytest.mark.asyncio
async def test_tool_call_json_text_result_reaches_model_as_payload(monkeypatch) -> None:
    monkeypatch.setattr(mcp_binding, "mcp_adapters_available", lambda: True)

    @asynccontextmanager
    async def open_entry(_entry):
        yield _FakeJsonTextResultClient()

    monkeypatch.setattr(mcp_binding, "_open_server_entry", open_entry)
    tools = await mcp_binding.load_mcp_tools_from_server_map({
        "kdcube-services": {"url": "https://mcp.example/named-services"},
    })

    result = await tools[0].ainvoke({"value": "ignored"})

    assert result["ok"] is False
    assert result["status"] == 400
    assert result["is_error"] is True
    assert result["error"]["code"] == "file_path_required"
    assert "content" not in result


@pytest.mark.asyncio
async def test_server_instructions_use_negotiated_client(monkeypatch) -> None:
    mcp_mod = ModuleType("mcp")
    client_mod = ModuleType("mcp.client")
    client_mod.Client = object
    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)

    @asynccontextmanager
    async def open_entry(_entry):
        yield _FakeClient()

    monkeypatch.setattr(mcp_binding, "_open_server_entry", open_entry)
    instructions = await mcp_binding.load_mcp_server_instructions({
        "knowledge": {"url": "https://mcp.example/knowledge"},
    })

    assert instructions == {"knowledge": "Use echo for exact repetition."}
