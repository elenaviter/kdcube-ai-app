# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube defaults for MCP SDK v2 server surfaces."""

from __future__ import annotations

from typing import Any, Optional

from mcp.server import MCPServer


class KDCubeMCPServer(MCPServer):
    """An MCP v2 server with distributed-serving HTTP defaults.

    Bundle MCP apps are rebuilt and dispatched independently, so their managed
    streamable-HTTP surface is stateless by default. The SDK v2 server itself
    accepts both the modern 2026-07-28 protocol and legacy initialize clients.
    """

    def __init__(
        self,
        *args: Any,
        stateless_http: bool = True,
        json_response: bool = False,
        streamable_http_path: str = "/mcp",
        **kwargs: Any,
    ) -> None:
        self.kdcube_stateless_http = bool(stateless_http)
        self.kdcube_json_response = bool(json_response)
        self.kdcube_streamable_http_path = str(streamable_http_path or "/mcp")
        super().__init__(*args, **kwargs)

    def streamable_http_app(
        self,
        *,
        streamable_http_path: Optional[str] = None,
        json_response: Optional[bool] = None,
        stateless_http: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        return super().streamable_http_app(
            streamable_http_path=(
                self.kdcube_streamable_http_path
                if streamable_http_path is None
                else streamable_http_path
            ),
            json_response=(
                self.kdcube_json_response
                if json_response is None
                else json_response
            ),
            stateless_http=(
                self.kdcube_stateless_http
                if stateless_http is None
                else stateless_http
            ),
            **kwargs,
        )

    def run(self, transport: str = "stdio", **kwargs: Any) -> None:
        if transport == "streamable-http":
            kwargs.setdefault("streamable_http_path", self.kdcube_streamable_http_path)
            kwargs.setdefault("json_response", self.kdcube_json_response)
            kwargs.setdefault("stateless_http", self.kdcube_stateless_http)
        return super().run(transport=transport, **kwargs)
