from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

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


@pytest.mark.asyncio
async def test_loader_uses_sdk_tools_and_records_errors_per_server(monkeypatch) -> None:
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
async def test_server_instructions_use_negotiated_client(monkeypatch) -> None:
    @asynccontextmanager
    async def open_entry(_entry):
        yield _FakeClient()

    monkeypatch.setattr(mcp_binding, "_open_server_entry", open_entry)
    instructions = await mcp_binding.load_mcp_server_instructions({
        "knowledge": {"url": "https://mcp.example/knowledge"},
    })

    assert instructions == {"knowledge": "Use echo for exact repetition."}
