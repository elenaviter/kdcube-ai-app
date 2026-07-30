# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter
#
# chat/sdk/runtime/mcp/mcp_adapter.py
#
# MCP adapter interface and tool schema contracts.

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import logging
import os

from kdcube_ai_app.apps.chat.sdk.runtime.mcp.client import (
    mcp_tool_schema,
    normalize_mcp_tool_result,
    open_mcp_client,
)

logger = logging.getLogger(__name__)


@dataclass
class MCPServerSpec:
    server_id: str
    display_name: str
    transport: str = "stdio"  # stdio | sse | streamable-http | http
    endpoint: str = ""        # URL for http/sse transports
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    auth_profile: Optional[Dict[str, Any]] = None
    protocol_mode: str = "auto"
    read_timeout_seconds: Optional[float] = None


@dataclass
class MCPToolSchema:
    id: str
    name: str
    description: str
    params_schema: Dict[str, Any]
    returns_schema: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class MCPAdapter(Protocol):
    """
    Minimal MCP adapter contract.
    Implementations are responsible for:
      - connecting to an MCP server
      - listing tools + schemas
      - executing tools remotely
    """

    server: MCPServerSpec

    async def list_tools(self) -> List[MCPToolSchema]:
        ...

    async def call_tool(
        self,
        tool_id: str,
        params: Dict[str, Any],
        *,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...


class PythonSDKMCPAdapter:
    """
    MCP adapter using the official Python SDK (mcp package).
    Supports stdio, sse and streamable-http transports.
    """

    def __init__(self, server: MCPServerSpec):
        self.server = server

    async def list_tools(self) -> List[MCPToolSchema]:
        logger.info("MCP adapter list_tools: server=%s transport=%s opening session", self.server.server_id, self.server.transport)
        async with self._session() as session:
            logger.info("MCP adapter list_tools: server=%s session ready, sending ListToolsRequest", self.server.server_id)
            resp = await session.list_tools()
            tools = [self._tool_from_sdk(t) for t in (getattr(resp, "tools", []) or [])]
            logger.info("MCP adapter list_tools: server=%s got %d tools", self.server.server_id, len(tools))
            return tools

    async def call_tool(
        self,
        tool_id: str,
        params: Dict[str, Any],
        *,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("MCP adapter call_tool: server=%s tool=%s opening session", self.server.server_id, tool_id)
        async with self._session() as session:
            logger.info("MCP adapter call_tool: server=%s tool=%s session ready, calling", self.server.server_id, tool_id)
            result = await session.call_tool(tool_id, params or {})
            logger.info("MCP adapter call_tool: server=%s tool=%s call completed", self.server.server_id, tool_id)
            return normalize_mcp_tool_result(result)

    def _tool_from_sdk(self, tool: Any) -> MCPToolSchema:
        normalized = mcp_tool_schema(tool)
        tool_id = normalized["name"]
        return MCPToolSchema(
            id=str(tool_id),
            name=str(tool_id),
            description=str(normalized["description"]),
            params_schema=normalized["input_schema"],
            returns_schema=normalized["output_schema"],
            tags=None,
        )

    async def _auth_headers(self) -> Dict[str, str]:
        auth = self.server.auth_profile or {}
        if not isinstance(auth, dict):
            return {}
        auth_type = (auth.get("type") or "").strip().lower()
        if auth_type in {"oauth_gui", "oauth-gui", "interactive"}:
            return {}
        env_key = auth.get("env")
        secret_key = auth.get("secret")  # dot-path key for get_secret()
        header = auth.get("header")
        # Resolve token: get_secret() first (supports bundle secrets), then env fallback
        token = None
        if secret_key:
            try:
                from kdcube_ai_app.apps.chat.sdk.config import get_secret
                token = await get_secret(secret_key)
            except Exception:
                pass
        if not token and env_key:
            try:
                from kdcube_ai_app.apps.chat.sdk.config import get_secret
                token = await get_secret(env_key)
            except Exception:
                token = os.environ.get(env_key)
        if not token:
            return {}
        if auth_type in {"bearer", "oauth"}:
            return {"Authorization": f"Bearer {token}"}
        if auth_type in {"api_key", "apikey", "key"}:
            return {str(header or "X-API-Key"): str(token)}
        if auth_type in {"header"} and header:
            return {str(header): str(token)}
        return {}

    def _session(self):
        return _adapter_client_context(self)


async def _resolve_secret_ref(value: str) -> str:
    """
    Resolve ``${secret:dot.path.key}`` references in env values via get_secret().
    If the value does not match the pattern, it is returned as-is.
    """
    if not isinstance(value, str) or not value.startswith("${secret:") or not value.endswith("}"):
        return value
    key = value[len("${secret:"):-1].strip()
    if not key:
        return value
    try:
        from kdcube_ai_app.apps.chat.sdk.config import get_secret
        resolved = await get_secret(key)
        return resolved or value
    except Exception:
        return value


async def _resolve_stdio_env(server: MCPServerSpec) -> dict | None:
    """
    Build environment dict for a stdio MCP subprocess.

    When *env* is ``None`` the child inherits the parent environment
    automatically (MCP SDK default).  When *env* is set, the SDK
    **replaces** the entire environment, so critical variables like
    ``PYTHONPATH`` and ``PATH`` would be lost.

    This helper merges the parent ``PYTHONPATH`` / ``PATH`` into the
    server-specific env so that ``python -m …`` invocations can resolve
    installed packages without hardcoding paths in the config.

    Env values matching ``${secret:dot.path.key}`` are resolved via
    get_secret(), enabling bundle-specific secrets from bundles.secrets.yaml.
    """
    env = server.env
    if env is None:
        return None

    env = dict(env)  # don't mutate the original

    # Resolve ${secret:...} references in env values
    for k, v in env.items():
        env[k] = await _resolve_secret_ref(v)

    # Inherit PYTHONPATH from the parent process so that
    # `python -m kdcube_ai_app.…` resolves without manual config.
    if "PYTHONPATH" not in env:
        parent_pp = os.environ.get("PYTHONPATH", "")
        if parent_pp:
            env["PYTHONPATH"] = parent_pp

    # Inherit PATH so that `python`, `npx`, etc. are discoverable.
    if "PATH" not in env:
        parent_path = os.environ.get("PATH", "")
        if parent_path:
            env["PATH"] = parent_path

    return env


@asynccontextmanager
async def _adapter_client_context(adapter: PythonSDKMCPAdapter):
    server = adapter.server
    async with open_mcp_client(
        transport=server.transport,
        endpoint=server.endpoint,
        command=server.command,
        args=server.args,
        env=await _resolve_stdio_env(server),
        headers=await adapter._auth_headers(),
        mode=server.protocol_mode,
        read_timeout_seconds=server.read_timeout_seconds,
    ) as client:
        yield client


class MCPToolsSubsystemLike(Protocol):
    async def execute_tool(
        self,
        *,
        alias: str,
        tool_name: str,
        params: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...


def mcp_tools_to_catalog(tools: List[MCPToolSchema]) -> List[Dict[str, Any]]:
    """
    Convert MCP tool schemas into the internal tool-catalog shape used by ReAct.
    """
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        out.append({
            "id": t.id,
            "doc": {
                "purpose": t.description or t.name,
                "args": t.params_schema or {},
                "returns": t.returns_schema or {},
            },
        })
    return out


async def execute_mcp_tool(
    *,
    tool_id: str,
    params: Dict[str, Any],
    mcp_subsystem: MCPToolsSubsystemLike,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute an MCP tool by tool_id using the MCPToolsSubsystem.
    tool_id format: mcp.<alias>.<tool_id...>
    """
    from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import parse_tool_id

    origin, provider, name = parse_tool_id(tool_id)
    if origin != "mcp" or not provider or not name:
        return {"error": f"Invalid MCP tool_id: {tool_id}"}
    if mcp_subsystem is None:
        return {"error": "MCP subsystem is not configured"}
    return await mcp_subsystem.execute_tool(alias=provider, tool_name=name, params=params, trace_id=trace_id)
