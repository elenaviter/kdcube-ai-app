# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Bind a KDCube-served MCP endpoint's tools as LangChain tools.

For any hosted LangGraph/LangChain agent: given a standard MCP server map —
``{server_id: {url, transport, headers}}`` — load its tools as LangChain
``BaseTool``s through KDCube's official-SDK adapter. Reusable by any bundle; the
per-user delegated bearer (if any) is already resolved into ``headers`` by
``solutions/connections/delegated_mcp.resolve_mcp_server_map`` — this module
knows nothing about delegated credentials, only the neutral server map.

Degrades cleanly: returns ``[]`` (with a logged hint) when the map is empty,
the MCP/LangChain dependencies are unavailable, or the endpoint is unreachable,
so the agent remains buildable with its plain tools regardless of MCP state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.runtime.mcp.client import (
    mcp_tool_schema,
    normalize_mcp_tool_result,
    open_mcp_client,
)

logger = logging.getLogger(__name__)


def mcp_adapters_available() -> bool:
    """Whether the official MCP SDK and LangChain tool API are importable."""
    try:
        from langchain_core.tools import StructuredTool  # noqa: F401
        from mcp.client import Client  # noqa: F401
        return True
    except Exception:
        return False


async def load_mcp_tools_from_server_map(
    server_map: Dict[str, Dict[str, Any]],
    *,
    error_sink: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Load LangChain tools from a resolved MCP server map. ``[]`` on any
    absence/failure — never raises, so a graph build never fails over an
    optional MCP tool source.

    ``error_sink``: when a dict is passed, a load failure records the raw
    exception under ``error_sink["_load_error"]``. Loading CONNECTS to each
    server, so a consent/auth denial (a KDCube `@mcp` 403) surfaces HERE, at
    load, before any tool call — the caller inspects the error and shapes a
    consent demand instead of silently dropping the tools."""
    if not server_map:
        return []
    if not mcp_adapters_available():
        logger.warning(
            "frameworks.langchain.mcp: MCP SDK or LangChain tool support is not "
            "installed; skipping MCP tools."
        )
        return []
    from langchain_core.tools import StructuredTool

    tools: List[Any] = []
    server_errors: Dict[str, BaseException] = {}
    for server_id, entry in server_map.items():
        try:
            async with _open_server_entry(entry) as client:
                listed = await client.list_tools()
                for raw_tool in getattr(listed, "tools", None) or []:
                    normalized = mcp_tool_schema(raw_tool)
                    tool_name = normalized["name"]
                    if not tool_name:
                        continue

                    async def _call_mcp_tool(
                        _server_entry: Mapping[str, Any] = dict(entry),
                        _tool_name: str = tool_name,
                        **params: Any,
                    ) -> Any:
                        async with _open_server_entry(_server_entry) as call_client:
                            result = await call_client.call_tool(_tool_name, params or {})
                            return normalize_mcp_tool_result(result)

                    tools.append(StructuredTool(
                        name=tool_name,
                        description=normalized["description"] or tool_name,
                        args_schema=normalized["input_schema"],
                        coroutine=_call_mcp_tool,
                        metadata={"mcp_server_id": server_id},
                    ))
        except Exception as exc:  # noqa: BLE001 - one optional server must not break the graph
            server_errors[server_id] = exc
            logger.warning(
                "frameworks.langchain.mcp: MCP tool load failed for %s (%s); "
                "continuing without that server.",
                server_id,
                exc,
            )

    if error_sink is not None and server_errors:
        error_sink["_server_errors"] = server_errors
        error_sink["_load_error"] = next(iter(server_errors.values()))

    logger.info(
        "frameworks.langchain.mcp: loaded %d MCP tool(s) from %d/%d server(s).",
        len(tools), len(server_map) - len(server_errors), len(server_map),
    )
    # Chat-side post-processing, applied once so every MCP consumer inherits
    # consent banners and file-card handling from self-describing results.
    try:
        from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.mcp_result import (
            bind_chat_result_handling,
        )

        tools = bind_chat_result_handling(tools)
    except Exception:  # pragma: no cover - never fail a load over the wrapper
        logger.info("frameworks.langchain.mcp: chat result-handling bind skipped", exc_info=True)
    return tools


async def load_mcp_server_instructions(
    server_map: Dict[str, Dict[str, Any]],
    *,
    timeout_s: float = 6.0,
) -> Dict[str, str]:
    """Fetch each MCP server's instructions from its negotiated connection.

    An MCP server may publish usage instructions during protocol negotiation,
    the operating guide MCP-native clients (e.g. Claude's connectors) surface
    to their model. Tool schemas do not carry that server-level guide, so this
    helper reads it from the negotiated KDCube client session. It is best-effort:
    ``{server_id: instructions}`` for servers that publish any; failures and
    absences are skipped silently — never raises."""
    out: Dict[str, str] = {}
    if not server_map:
        return out
    try:
        import asyncio

        from mcp.client import Client  # noqa: F401
    except Exception:
        return out
    for server_id, entry in server_map.items():
        try:
            async with asyncio.timeout(timeout_s):
                async with _open_server_entry(entry) as client:
                    instructions = str(client.instructions or "").strip()
                    if instructions:
                        out[server_id] = instructions
        except Exception:
            logger.info(
                "frameworks.langchain.mcp: no server instructions from %r (non-fatal).",
                server_id,
            )
    return out


def _open_server_entry(entry: Mapping[str, Any]):
    transport = str(entry.get("transport") or "streamable_http")
    return open_mcp_client(
        transport=transport,
        endpoint=str(entry.get("url") or entry.get("endpoint") or ""),
        command=(str(entry.get("command")) if entry.get("command") else None),
        args=entry.get("args") or (),
        env=entry.get("env") if isinstance(entry.get("env"), Mapping) else None,
        headers=(
            entry.get("headers")
            if isinstance(entry.get("headers"), Mapping)
            else None
        ),
        mode=str(entry.get("protocol_mode") or "auto"),
        read_timeout_seconds=(
            float(entry["read_timeout_seconds"])
            if entry.get("read_timeout_seconds") is not None
            else None
        ),
    )


def _iter_exc_chain(error: Any):
    seen: set = set()
    stack = [error]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        for nxt in (getattr(e, "__cause__", None), getattr(e, "__context__", None)):
            if nxt is not None:
                stack.append(nxt)
        for sub in getattr(e, "exceptions", None) or ():  # ExceptionGroup / TaskGroup
            stack.append(sub)


def load_error_looks_like_denial(error: Any) -> bool:
    """Whether an MCP load error carries an auth/consent denial (403/401)."""
    if error is None:
        return False
    text = " ".join(str(x) for x in _iter_exc_chain(error)).lower()
    return "403" in text or "forbidden" in text or "401" in text or "unauthorized" in text
