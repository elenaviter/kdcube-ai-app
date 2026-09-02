# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Official MCP SDK v2 client construction and result normalization."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from typing import Any, AsyncIterator, Mapping, Optional, Sequence


MCP_CLIENT_MODE_AUTO = "auto"


@asynccontextmanager
async def open_mcp_client(
    *,
    transport: str,
    endpoint: str = "",
    command: Optional[str] = None,
    args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    headers: Optional[Mapping[str, str]] = None,
    mode: str = MCP_CLIENT_MODE_AUTO,
    read_timeout_seconds: Optional[float] = None,
    follow_redirects: bool = True,
    http_transport: Any = None,
    trust_env: bool = True,
) -> AsyncIterator[Any]:
    """Open an MCP SDK v2 client with modern discovery and legacy fallback.

    ``mode="auto"`` first probes ``server/discover`` and falls back to the
    pre-2026 ``initialize`` handshake when the peer is legacy. HTTP headers
    are supplied to the transport HTTP client; they are never copied into MCP
    parameters or model-visible tool data.
    """

    from mcp.client import Client

    normalized = str(transport or "stdio").strip().lower()
    if normalized in {"stdio", "local"}:
        target = _stdio_transport(command=command, args=args, env=env)
    elif normalized == "sse":
        target = _sse_transport(endpoint=endpoint, headers=headers)
    elif normalized in {"streamable-http", "streamable_http", "http", ""}:
        target = _streamable_http_transport(
            endpoint=endpoint,
            headers=headers,
            follow_redirects=follow_redirects,
            http_transport=http_transport,
            trust_env=trust_env,
        )
    else:
        raise ValueError(f"Unsupported MCP transport: {transport!r}")

    async with Client(
        target,
        mode=mode,
        read_timeout_seconds=read_timeout_seconds,
    ) as client:
        yield client


def mcp_tool_schema(tool: Any) -> dict[str, Any]:
    """Return one SDK tool as a stable, JSON-schema-shaped mapping."""

    schema = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
        or {}
    )
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(mode="json", by_alias=True)
    if not isinstance(schema, dict):
        schema = {}
    return {
        "name": str(getattr(tool, "name", None) or getattr(tool, "id", None) or ""),
        "description": str(getattr(tool, "description", None) or ""),
        "input_schema": schema,
        "output_schema": _model_or_mapping(
            getattr(tool, "output_schema", None)
            or getattr(tool, "outputSchema", None)
        ),
    }


def normalize_mcp_tool_result(result: Any) -> Any:
    """Convert an SDK call result into the neutral value KDCube tools return."""

    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    content = [
        _content_block(block)
        for block in (getattr(result, "content", None) or [])
    ]
    json_payload = _single_json_text_payload(content)
    if json_payload is not None:
        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", None)
        if (
            isinstance(json_payload, Mapping)
            and is_error is not None
            and "is_error" not in json_payload
            and "isError" not in json_payload
        ):
            json_payload = {**json_payload, "is_error": bool(is_error)}
        return json_payload

    payload: dict[str, Any] = {"content": content}
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", None)
    if is_error is not None:
        payload["is_error"] = bool(is_error)
    return payload


def _single_json_text_payload(content: Sequence[Any]) -> Any:
    if len(content) != 1:
        return None
    block = content[0]
    if isinstance(block, Mapping):
        if str(block.get("type") or "") != "text":
            return None
        text = block.get("text")
    else:
        text = getattr(block, "text", None)
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _content_block(block: Any) -> Any:
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(block, Mapping):
        return dict(block)
    return {
        "type": getattr(block, "type", None),
        "text": getattr(block, "text", None),
    }


def _model_or_mapping(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)
    return dict(value) if isinstance(value, Mapping) else None


def _stdio_transport(
    *,
    command: Optional[str],
    args: Optional[Sequence[str]],
    env: Optional[Mapping[str, str]],
) -> Any:
    from mcp.client.stdio import StdioServerParameters, stdio_client

    return stdio_client(
        StdioServerParameters(
            command=str(command or ""),
            args=[str(item) for item in (args or ())],
            env=dict(env) if env is not None else None,
        )
    )


def _sse_transport(
    *,
    endpoint: str,
    headers: Optional[Mapping[str, str]],
) -> Any:
    from mcp.client.sse import sse_client

    if not endpoint:
        raise ValueError("MCP SSE transport requires an endpoint")
    return sse_client(url=endpoint, headers=dict(headers or {}))


@asynccontextmanager
async def _streamable_http_transport(
    *,
    endpoint: str,
    headers: Optional[Mapping[str, str]],
    follow_redirects: bool = True,
    http_transport: Any = None,
    trust_env: bool = True,
) -> AsyncIterator[Any]:
    import httpx2

    from mcp.client.streamable_http import streamable_http_client

    if not endpoint:
        raise ValueError("MCP streamable HTTP transport requires an endpoint")
    timeout = httpx2.Timeout(30.0, read=300.0)
    async with httpx2.AsyncClient(
        headers=dict(headers or {}),
        timeout=timeout,
        follow_redirects=bool(follow_redirects),
        transport=http_transport,
        trust_env=bool(trust_env),
    ) as http_client:
        async with streamable_http_client(
            endpoint,
            http_client=http_client,
        ) as streams:
            yield streams
