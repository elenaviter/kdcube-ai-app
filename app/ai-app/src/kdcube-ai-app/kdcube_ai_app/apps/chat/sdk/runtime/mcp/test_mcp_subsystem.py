# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.mcp.mcp_adapter import (
    MCPServerSpec,
    MCPToolSchema,
    PythonSDKMCPAdapter,
)
from kdcube_ai_app.apps.chat.sdk.runtime.mcp.mcp_tools_subsystem import MCPToolsSubsystem


class _DummyAdapter:
    def __init__(self, server: MCPServerSpec):
        self.server = server

    async def list_tools(self):
        return [
            MCPToolSchema(
                id="so_search",
                name="so_search",
                description="Search",
                params_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
            MCPToolSchema(
                id="get_content",
                name="get_content",
                description="Get content",
                params_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            ),
        ]

    async def call_tool(self, tool_id, params, *, trace_id=None):
        return {"ok": True, "tool_id": tool_id, "params": params, "trace_id": trace_id}


def _dummy_factory(server: MCPServerSpec):
    return _DummyAdapter(server)


class _MemoryCache:
    def __init__(self):
        self.values = {}
        self.get_calls = 0
        self.set_calls = 0

    async def get_json(self, key):
        self.get_calls += 1
        return self.values.get(key)

    async def set_json(self, key, value, *, ttl_seconds=None):
        self.set_calls += 1
        self.values[key] = value
        return True


def test_mcp_services_env_accepts_mcpServers():
    env_json = """
    {
      "mcpServers": {
        "stack": { "transport": "stdio", "command": "npx", "args": ["mcp-remote", "mcp.stackoverflow.com"] }
      }
    }
    """
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "stack", "alias": "stack"}],
        adapter_factory=_dummy_factory,
        env_json=env_json,
    )
    server = ss._server_spec("stack")
    assert server is not None
    assert server.transport == "stdio"
    assert server.command == "npx"


def test_mcp_services_config_accepts_dict_payload():
    services_cfg = {
        "mcpServers": {
            "docs": {
                "transport": "http",
                "url": "https://mcp.example.com",
                "protocol_mode": "legacy",
                "read_timeout_seconds": 17.5,
                "auth": {"type": "bearer", "secret": "bundles.react.mcp@2026-03-09.secrets.docs.token"},
            }
        }
    }
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "docs", "alias": "docs"}],
        adapter_factory=_dummy_factory,
        services_config=services_cfg,
    )
    server = ss._server_spec("docs")
    assert server is not None
    assert server.transport == "http"
    assert server.endpoint == "https://mcp.example.com"
    assert server.protocol_mode == "legacy"
    assert server.read_timeout_seconds == 17.5
    assert server.auth_profile == {"type": "bearer", "secret": "bundles.react.mcp@2026-03-09.secrets.docs.token"}


def test_trusted_process_provider_only_binds_explicitly_marked_server():
    async def _headers(_server_id):
        return {"Authorization": "Bearer current-card"}

    services_cfg = {
        "mcpServers": {
            "gateway": {
                "transport": "http",
                "url": "https://hub.example/mcp",
                "auth": {"type": "trusted_process"},
            },
            "static": {
                "transport": "http",
                "url": "https://static.example/mcp",
                "auth": {"type": "bearer", "secret": "b:mcp.static.token"},
            },
        }
    }
    subsystem = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[
            {"server_id": "gateway", "alias": "gateway"},
            {"server_id": "static", "alias": "static"},
        ],
        adapter_factory=_dummy_factory,
        services_config=services_cfg,
        auth_headers_provider=_headers,
    )

    gateway = subsystem._server_spec("gateway")
    static = subsystem._server_spec("static")
    assert gateway is not None
    assert gateway.auth_headers_provider is _headers
    assert static is not None
    assert static.auth_headers_provider is None
    assert static.auth_profile == {
        "type": "bearer",
        "secret": "b:mcp.static.token",
    }


def test_trusted_process_server_without_provider_fails_closed():
    subsystem = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "gateway", "alias": "gateway"}],
        adapter_factory=_dummy_factory,
        services_config={
            "mcpServers": {
                "gateway": {
                    "transport": "http",
                    "url": "https://hub.example/mcp",
                    "auth": {"type": "trusted_process"},
                }
            }
        },
    )

    assert subsystem._server_spec("gateway") is None


def test_auth_provider_failure_is_secret_safe_and_provider_is_not_repr():
    marker = "provider-secret-must-not-escape"

    async def _failing_provider(_server_id):
        raise RuntimeError(marker)

    spec = MCPServerSpec(
        server_id="gateway",
        display_name="Gateway",
        transport="http",
        endpoint="https://hub.example/mcp",
        auth_headers_provider=_failing_provider,
    )
    assert marker not in repr(spec)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(PythonSDKMCPAdapter(spec)._auth_headers())
    assert str(caught.value) == "MCP authorization provider unavailable"
    assert marker not in str(caught.value)


@pytest.mark.asyncio
async def test_trusted_process_transport_errors_are_secret_safe(caplog):
    marker = "transport-error-with-runtime-only-token"

    async def _headers(_server_id):
        return {"Authorization": "Bearer runtime-only-token"}

    class _FailingAdapter:
        def __init__(self, server):
            self.server = server

        async def list_tools(self):
            raise RuntimeError(marker)

        async def call_tool(self, tool_id, params, *, trace_id=None):
            del tool_id, params, trace_id
            raise RuntimeError(marker)

    subsystem = MCPToolsSubsystem(
        bundle_id="bundle@1-0",
        mcp_tool_specs=[
            {"server_id": "gateway", "alias": "gateway", "tools": ["search"]}
        ],
        services_config={
            "mcpServers": {
                "gateway": {
                    "transport": "streamable-http",
                    "url": "https://gateway.example.test/mcp",
                    "auth": {"type": "trusted_process"},
                    "ttl_seconds": 0,
                }
            }
        },
        auth_headers_provider=_headers,
        adapter_factory=_FailingAdapter,
        cache=None,
    )

    assert await subsystem.build_tool_entries() == []
    result = await subsystem.call_tool(
        alias="gateway",
        tool_id="search",
        params={"query": "test"},
    )

    assert result == {
        "error": "MCP call_tool failed: authorized transport unavailable"
    }
    assert marker not in caplog.text
    assert "runtime-only-token" not in caplog.text
    assert marker not in repr(result)


def test_export_services_config_round_trips_server_map():
    services_cfg = {
        "mcpServers": {
            "stack": {"transport": "stdio", "command": "npx", "args": ["mcp-remote", "mcp.stackoverflow.com"]},
        }
    }
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "stack", "alias": "stack"}],
        adapter_factory=_dummy_factory,
        services_config=services_cfg,
    )
    assert ss.export_services_config() == services_cfg


def test_interactive_auth_is_hidden():
    env_json = """
    {
      "mcpServers": {
        "stack": { "transport": "stdio", "command": "npx", "auth": { "type": "oauth_gui" } }
      }
    }
    """
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "stack", "alias": "stack"}],
        adapter_factory=_dummy_factory,
        env_json=env_json,
    )
    assert ss._server_spec("stack") is None


def test_transport_validation():
    env_json = """
    {
      "mcpServers": {
        "s1": { "transport": "stdio" },
        "s2": { "transport": "http", "url": "https://mcp.example.com" },
        "s3": { "transport": "sse", "url": "http://127.0.0.1:8787/sse" }
      }
    }
    """
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[
            {"server_id": "s1", "alias": "s1"},
            {"server_id": "s2", "alias": "s2"},
            {"server_id": "s3", "alias": "s3"},
        ],
        adapter_factory=_dummy_factory,
        env_json=env_json,
    )
    assert ss._server_spec("s1") is None  # stdio requires command
    assert ss._server_spec("s2") is not None
    assert ss._server_spec("s3") is not None


def test_build_entries_and_execute():
    env_json = """
    {
      "mcpServers": {
        "stack": { "transport": "stdio", "command": "npx", "args": ["mcp-remote", "mcp.stackoverflow.com"] }
      }
    }
    """
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "stack", "alias": "stack", "tools": ["so_search"]}],
        adapter_factory=_dummy_factory,
        env_json=env_json,
    )

    async def _run():
        entries = await ss.build_tool_entries()
        assert any(e["id"] == "mcp.stack.so_search" for e in entries)
        assert all(e["id"] != "mcp.stack.get_content" for e in entries)
        out = await ss.execute_tool(alias="stack", tool_name="so_search", params={"q": "auth"})
        assert out.get("ok") is True

    asyncio.run(_run())


def test_sdk_adapter_sends_stable_invocation_id_as_request_metadata():
    adapter = PythonSDKMCPAdapter(
        MCPServerSpec(server_id="hub", display_name="Connection Hub")
    )
    calls = []

    class Session:
        async def call_tool(self, name, arguments, **kwargs):
            calls.append({"name": name, "arguments": arguments, **kwargs})
            return SimpleNamespace(structured_content={"ok": True})

    @asynccontextmanager
    async def session():
        yield Session()

    adapter._session = session

    async def _run():
        result = await adapter.call_tool(
            "restart",
            {"service": "api"},
            trace_id="mcp_hub_restart-20260901203000000.json",
        )
        assert result == {"ok": True}

    asyncio.run(_run())
    assert calls == [
        {
            "name": "restart",
            "arguments": {"service": "api"},
            "meta": {
                "connection_hub/invocation_id": (
                    "mcp_hub_restart-20260901203000000.json"
                )
            },
        }
    ]


def test_tools_cache_key_includes_auth_fingerprint(monkeypatch):
    services_cfg = {
        "mcpServers": {
            "knowledge": {
                "transport": "http",
                "url": "https://mcp.example.com",
                "auth": {
                    "type": "header",
                    "header": "X-Knowledge-MCP-Token",
                    "secret": "b:mcp.knowledge.token",
                },
            }
        }
    }
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "knowledge", "alias": "knowledge"}],
        adapter_factory=_dummy_factory,
        services_config=services_cfg,
    )
    server = ss._server_spec("knowledge")
    assert server is not None

    tokens = iter(["token-a", "token-b"])

    async def _fake_get_secret(key):
        assert key == "b:mcp.knowledge.token"
        return next(tokens)

    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.runtime.mcp.mcp_tools_subsystem.get_secret",
        _fake_get_secret,
    )

    async def _run():
        key_a = await ss._tools_cache_key(server)
        key_b = await ss._tools_cache_key(server)
        assert key_a != key_b
        assert key_a.startswith("knowledge:tools:")
        assert key_b.startswith("knowledge:tools:")
        assert "token-a" not in key_a
        assert "token-b" not in key_b

    asyncio.run(_run())


def test_tools_cache_ttl_zero_disables_cache():
    services_cfg = {
        "mcpServers": {
            "docs": {
                "transport": "http",
                "url": "https://mcp.example.com",
                "ttl_seconds": 0,
            }
        }
    }
    cache = _MemoryCache()
    ss = MCPToolsSubsystem(
        bundle_id="b1",
        mcp_tool_specs=[{"server_id": "docs", "alias": "docs"}],
        adapter_factory=_dummy_factory,
        cache=cache,
        services_config=services_cfg,
    )
    ss.cache = cache
    server = ss._server_spec("docs")
    assert server is not None

    async def _run():
        tools = await ss._tools_for_server(server)
        assert [t.id for t in tools] == ["so_search", "get_content"]
        assert cache.get_calls == 0
        assert cache.set_calls == 0

    asyncio.run(_run())
