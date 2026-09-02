# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Authenticated streamable-HTTP MCP fixture for local acceptance tests."""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from starlette.responses import JSONResponse

from kdcube_ai_app.apps.chat.sdk.runtime.mcp.server import KDCubeMCPServer


def _fixture_version() -> str:
    return str(os.getenv("REMOTE_MCP_FIXTURE_VERSION") or "1").strip() or "1"


def build_app() -> Any:
    version = _fixture_version()
    bearer = str(os.getenv("REMOTE_MCP_FIXTURE_BEARER") or "").strip()
    call_counts = {"search": 0, "delete": 0}
    server = KDCubeMCPServer(
        "Connection Hub acceptance fixture",
        version=version,
        stateless_http=True,
        json_response=True,
    )

    @server.tool(
        name="search",
        description=(
            "Search fixture records"
            if version == "1"
            else "Search fixture records and include matching record details"
        ),
    )
    async def search(query: str) -> dict[str, Any]:
        call_counts["search"] += 1
        return {
            "ok": True,
            "tool": "search",
            "query": query,
            "fixture_version": version,
            "upstream_credential_verified": True,
            "upstream_call_count": call_counts["search"],
        }

    @server.tool(name="delete", description="Delete one fixture record")
    async def delete_record(record_id: str) -> dict[str, Any]:
        call_counts["delete"] += 1
        return {
            "ok": True,
            "tool": "delete",
            "record_id": record_id,
            "fixture_version": version,
            "upstream_credential_verified": True,
            "upstream_call_count": call_counts["delete"],
        }

    mcp_app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        host="0.0.0.0",
    )

    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "lifespan":
            await mcp_app(scope, receive, send)
            return
        if scope.get("type") == "http" and scope.get("path") == "/healthz":
            response = JSONResponse({"ok": True, "version": version})
            await response(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers") or ()
        }
        if not bearer or headers.get("authorization") != f"Bearer {bearer}":
            response = JSONResponse(
                {"ok": False, "error": "fixture_credential_required"},
                status_code=401,
            )
            await response(scope, receive, send)
            return
        await mcp_app(scope, receive, send)

    return app


def main() -> None:
    host = str(os.getenv("REMOTE_MCP_FIXTURE_HOST") or "0.0.0.0")
    port = int(os.getenv("REMOTE_MCP_FIXTURE_PORT") or "8765")
    uvicorn.run(build_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
