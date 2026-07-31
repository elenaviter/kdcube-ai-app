from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from kdcube_ai_app.apps.chat.proc.rest.integrations import integrations
from kdcube_ai_app.apps.chat.sdk.runtime.mcp.server import KDCubeMCPServer


class _WireCapture:
    """Record HTTP request/response bytes without changing the ASGI exchange."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.exchanges: list[dict[str, Any]] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_body = bytearray()
        response_body = bytearray()
        response_status = 0
        response_headers: list[tuple[bytes, bytes]] = []

        async def recording_receive() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                request_body.extend(message.get("body") or b"")
            return message

        async def recording_send(message: dict[str, Any]) -> None:
            nonlocal response_status, response_headers
            if message.get("type") == "http.response.start":
                response_status = int(message.get("status") or 0)
                response_headers = list(message.get("headers") or [])
            elif message.get("type") == "http.response.body":
                response_body.extend(message.get("body") or b"")
            await send(message)

        await self.app(scope, recording_receive, recording_send)
        self.exchanges.append({
            "request_headers": {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in scope.get("headers") or []
            },
            "request_body": bytes(request_body),
            "response_status": response_status,
            "response_headers": {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in response_headers
            },
            "response_body": bytes(response_body),
        })


@pytest.mark.asyncio
async def test_kdcube_server_serves_modern_and_legacy_wire_clients() -> None:
    server = KDCubeMCPServer("dual-era-wire")

    @server.tool()
    async def echo(value: str) -> str:
        return value

    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
        ) as http_client:
            for mode, expected_version in (
                ("auto", "2026-07-28"),
                ("legacy", "2025-11-25"),
            ):
                transport = streamable_http_client(
                    "http://127.0.0.1:8000/mcp",
                    http_client=http_client,
                    terminate_on_close=False,
                )
                async with Client(transport, mode=mode, cache=None) as client:
                    tools = await client.list_tools()
                    result = await client.call_tool("echo", {"value": mode})

                    assert client.protocol_version == expected_version
                    assert [tool.name for tool in tools.tools] == ["echo"]
                    assert [block.text for block in result.content] == [mode]


@pytest.mark.asyncio
async def test_modern_wire_is_stateless_self_contained_and_cacheable() -> None:
    server = KDCubeMCPServer("modern-wire-contract", json_response=True)

    @server.tool()
    async def echo(value: str) -> str:
        return value

    app = server.streamable_http_app()
    capture = _WireCapture(app)
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=capture),
            base_url="http://127.0.0.1:8000",
        ) as http_client:
            transport = streamable_http_client(
                "http://127.0.0.1:8000/mcp",
                http_client=http_client,
                terminate_on_close=False,
            )
            async with Client(transport, mode="auto", cache=None) as client:
                await client.list_tools()
                await client.call_tool("echo", {"value": "modern"})

    by_method: dict[str, dict[str, Any]] = {}
    for exchange in capture.exchanges:
        request_payload = json.loads(exchange["request_body"])
        by_method[str(request_payload["method"])] = {
            **exchange,
            "request_payload": request_payload,
            "response_payload": json.loads(exchange["response_body"]),
        }

    assert {"server/discover", "tools/list", "tools/call"} <= set(by_method)
    for method, exchange in by_method.items():
        request_payload = exchange["request_payload"]
        request_headers = exchange["request_headers"]
        meta = request_payload["params"]["_meta"]

        assert exchange["response_status"] == 200
        assert request_headers["mcp-protocol-version"] == "2026-07-28"
        assert request_headers["mcp-method"] == method
        assert "mcp-session-id" not in request_headers
        assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
        assert isinstance(meta["io.modelcontextprotocol/clientCapabilities"], dict)
        assert meta["io.modelcontextprotocol/clientInfo"]["name"]

        result = exchange["response_payload"]["result"]
        assert result["resultType"] == "complete"
        assert result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == (
            "modern-wire-contract"
        )

    assert by_method["tools/call"]["request_headers"]["mcp-name"] == "echo"
    tools_result = by_method["tools/list"]["response_payload"]["result"]
    assert isinstance(tools_result["ttlMs"], int)
    assert tools_result["cacheScope"] in {"private", "public"}


@pytest.mark.asyncio
async def test_auto_client_falls_back_to_legacy_initialize() -> None:
    methods: list[str] = []

    async def legacy_endpoint(request: Request) -> Response:
        payload = json.loads((await request.body()).decode("utf-8"))
        method = str(payload.get("method") or "")
        methods.append(method)
        request_id = payload.get("id")

        if method == "server/discover":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"},
            })
        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "serverInfo": {"name": "legacy-test", "version": "1"},
                },
            })
        if method == "notifications/initialized":
            return Response(status_code=202)
        if method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": []},
            })
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Method not found"},
        })

    app = Starlette(routes=[Route("/mcp", legacy_endpoint, methods=["POST"])])
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app),
        base_url="http://legacy.test",
    ) as http_client:
        transport = streamable_http_client(
            "http://legacy.test/mcp",
            http_client=http_client,
            terminate_on_close=False,
        )
        async with Client(transport, mode="auto", cache=None) as client:
            tools = await client.list_tools()

            assert client.protocol_version == "2025-11-25"
            assert tools.tools == []

    assert methods[:3] == [
        "server/discover",
        "initialize",
        "notifications/initialized",
    ]
    assert methods[-1] == "tools/list"


@pytest.mark.asyncio
async def test_proc_bridge_preserves_dual_era_stateless_mcp_wire() -> None:
    server = KDCubeMCPServer("bridged-dual-era-wire")

    @server.tool()
    async def echo(value: str) -> str:
        return value

    async def bridged_endpoint(request: Request) -> Response:
        inner_app = integrations._coerce_bundle_mcp_asgi_app(
            server,
            transport="streamable-http",
        )
        return await integrations._dispatch_bundle_mcp_request(
            request=request,
            mcp_app=inner_app,
            transport="streamable-http",
            mcp_path="",
        )

    outer_app = Starlette(routes=[Route("/mcp", bridged_endpoint, methods=["POST", "GET", "DELETE"])])
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=outer_app),
        base_url="http://bridge.test",
    ) as http_client:
        for mode, expected_version in (
            ("auto", "2026-07-28"),
            ("legacy", "2025-11-25"),
        ):
            transport = streamable_http_client(
                "http://bridge.test/mcp",
                http_client=http_client,
                terminate_on_close=False,
            )
            async with Client(transport, mode=mode, cache=None) as client:
                tools = await client.list_tools()
                result = await client.call_tool("echo", {"value": mode})

                assert client.protocol_version == expected_version
                assert [tool.name for tool in tools.tools] == ["echo"]
                assert [block.text for block in result.content] == [mode]
