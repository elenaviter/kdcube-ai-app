from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from connection_hub.remote_mcp import (
    RemoteMCPEndpointDenied,
    RemoteMCPEndpointPolicy,
)

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.remote_mcp import host


@pytest.mark.asyncio
async def test_transport_discovery_disables_redirects(monkeypatch):
    calls: list[dict] = []

    class _Client:
        server_info = SimpleNamespace(name="Fixture", version="1.2.3")
        protocol_version = "2026-07-28"

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="records.search",
                        description="Search records",
                        inputSchema={"type": "object"},
                    )
                ]
            )

    @asynccontextmanager
    async def open_client(**kwargs):
        calls.append(kwargs)
        yield _Client()

    monkeypatch.setattr(host, "open_mcp_client", open_client)
    http_transport = object()
    transport = host.KDCubeRemoteMCPTransport(
        endpoint_policy=host.remote_mcp_endpoint_policy({}),
        read_timeout_seconds=17,
    )
    monkeypatch.setattr(transport, "_http_transport", lambda: http_transport)
    discovery = await transport.discover(
        connector_id="mcp_0123456789abcdef01234567",
        endpoint="https://mcp.example.test/mcp",
        transport="streamable-http",
        headers={"Authorization": "Bearer secret"},
    )

    assert calls == [
        {
            "transport": "streamable-http",
            "endpoint": "https://mcp.example.test/mcp",
            "headers": {"Authorization": "Bearer secret"},
            "read_timeout_seconds": 17.0,
            "follow_redirects": False,
            "http_transport": http_transport,
            "trust_env": False,
        }
    ]
    assert [tool.name for tool in discovery.tools] == ["records.search"]
    assert discovery.server_name == "Fixture"


@pytest.mark.asyncio
async def test_transport_call_disables_redirects_and_bounds_result(monkeypatch):
    calls: list[dict] = []

    class _Client:
        async def call_tool(self, name, arguments):
            calls.append({"tool": name, "arguments": arguments})
            return SimpleNamespace(structuredContent={"answer": "ok"})

    @asynccontextmanager
    async def open_client(**kwargs):
        calls.append(kwargs)
        yield _Client()

    monkeypatch.setattr(host, "open_mcp_client", open_client)
    http_transport = object()
    transport = host.KDCubeRemoteMCPTransport(
        endpoint_policy=host.remote_mcp_endpoint_policy({}),
        max_result_bytes=128,
    )
    monkeypatch.setattr(transport, "_http_transport", lambda: http_transport)
    result = await transport.call_tool(
        connector_id="mcp_0123456789abcdef01234567",
        endpoint="https://mcp.example.test/mcp",
        transport="streamable-http",
        headers={},
        tool_name="records.search",
        arguments={"query": "failed jobs"},
    )

    assert result == {"answer": "ok"}
    assert calls[0]["follow_redirects"] is False
    assert calls[0]["http_transport"] is http_transport
    assert calls[0]["trust_env"] is False
    assert calls[1] == {
        "tool": "records.search",
        "arguments": {"query": "failed jobs"},
    }


@pytest.mark.asyncio
async def test_user_secret_adapter_never_uses_a_global_secret(monkeypatch):
    calls: list[tuple] = []

    async def set_secret(path, value, **kwargs):
        calls.append(("set", path, value, kwargs))

    async def get_secret(path, **kwargs):
        calls.append(("get", path, kwargs))
        return "stored-value"

    async def delete_secret(path, **kwargs):
        calls.append(("delete", path, kwargs))

    monkeypatch.setattr(host, "set_user_secret", set_secret)
    monkeypatch.setattr(host, "get_secret", get_secret)
    monkeypatch.setattr(host, "delete_user_secret", delete_secret)
    store = host.KDCubeRemoteMCPSecretStore(bundle_id="connection-hub@1-0")

    await store.set(owner_subject="user-1", secret_ref="remote_mcp.ref", value="s")
    assert await store.get(owner_subject="user-1", secret_ref="remote_mcp.ref") == "stored-value"
    await store.delete(owner_subject="user-1", secret_ref="remote_mcp.ref")

    assert calls == [
        (
            "set",
            "remote_mcp.ref",
            "s",
            {"user_id": "user-1", "bundle_id": "connection-hub@1-0"},
        ),
        (
            "get",
            "u:remote_mcp.ref",
            {"user_id": "user-1", "bundle_id": "connection-hub@1-0"},
        ),
        (
            "delete",
            "remote_mcp.ref",
            {"user_id": "user-1", "bundle_id": "connection-hub@1-0"},
        ),
    ]


def test_endpoint_policy_is_public_https_by_default_and_deployment_owned():
    default = host.remote_mcp_endpoint_policy({})
    assert default.allow_http is False
    assert default.allow_private_networks is False
    assert default.allowed_hosts == frozenset()

    local_fixture = host.remote_mcp_endpoint_policy(
        {
            "remote_mcp": {
                "outbound": {
                    "allow_http": True,
                    "allow_private_networks": False,
                    "allowed_hosts": ["HOST.DOCKER.INTERNAL."],
                }
            }
        }
    )
    assert local_fixture.allow_http is True
    assert local_fixture.allow_private_networks is False
    assert local_fixture.allowed_hosts == frozenset({"host.docker.internal"})


@pytest.mark.asyncio
async def test_guarded_backend_rechecks_dns_and_never_connects_rebound_private_ip():
    answers = iter(
        [
            ("93.184.216.34",),
            ("127.0.0.1",),
        ]
    )

    async def resolver(_host: str, _port: int):
        return next(answers)

    class Inner:
        def __init__(self) -> None:
            self.calls = []

        async def connect_tcp(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return object()

    policy = RemoteMCPEndpointPolicy(resolver=resolver)
    assert (
        await policy.validate("https://rebinder.example/mcp")
        == "https://rebinder.example/mcp"
    )
    inner = Inner()
    backend = host.GuardedRemoteMCPNetworkBackend(
        inner=inner,
        endpoint_policy=policy,
    )

    with pytest.raises(RemoteMCPEndpointDenied) as denied:
        await backend.connect_tcp("rebinder.example", 443)

    assert denied.value.reason == "endpoint_private_network_forbidden"
    assert inner.calls == []


@pytest.mark.asyncio
async def test_guarded_backend_pins_the_validated_public_address():
    async def resolver(_host: str, _port: int):
        return ("93.184.216.34",)

    class Inner:
        def __init__(self) -> None:
            self.calls = []

        async def connect_tcp(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "stream"

        async def sleep(self, _seconds):
            return None

    inner = Inner()
    backend = host.GuardedRemoteMCPNetworkBackend(
        inner=inner,
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=resolver),
    )

    stream = await backend.connect_tcp(
        "public.example",
        443,
        timeout=5.0,
        socket_options=[(1, 2, 3)],
    )

    assert stream == "stream"
    assert inner.calls == [
        (
            ("93.184.216.34", 443),
            {
                "timeout": 5.0,
                "local_address": None,
                "socket_options": [(1, 2, 3)],
            },
        )
    ]


@pytest.mark.asyncio
async def test_transport_installs_guarded_backend_on_httpx2_pool():
    async def resolver(_host: str, _port: int):
        return ("93.184.216.34",)

    policy = RemoteMCPEndpointPolicy(resolver=resolver)
    transport = host.KDCubeRemoteMCPTransport(endpoint_policy=policy)

    http_transport = transport._http_transport()
    try:
        assert isinstance(
            http_transport._pool._network_backend,
            host.GuardedRemoteMCPNetworkBackend,
        )
        assert http_transport._pool._network_backend._endpoint_policy is policy
    finally:
        await http_transport.aclose()
