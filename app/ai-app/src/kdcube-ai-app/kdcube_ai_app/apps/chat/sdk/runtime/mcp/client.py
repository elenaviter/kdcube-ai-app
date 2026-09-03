# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Compatibility surface for MCP client primitives extracted to app-foundation."""

from app_foundation.mcp import (
    MCP_CLIENT_MODE_AUTO,
    mcp_tool_schema,
    normalize_mcp_tool_result,
    open_mcp_client,
)


__all__ = [
    "MCP_CLIENT_MODE_AUTO",
    "mcp_tool_schema",
    "normalize_mcp_tool_result",
    "open_mcp_client",
]
