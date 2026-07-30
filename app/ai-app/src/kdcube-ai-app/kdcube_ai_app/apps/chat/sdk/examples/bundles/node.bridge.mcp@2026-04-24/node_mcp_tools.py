from __future__ import annotations

from typing import Any, Callable


def build_node_bridge_mcp_app(*, name: str, bridge_provider: Callable[[], Any]):
    try:
        from kdcube_ai_app.apps.chat.sdk.runtime.mcp.server import KDCubeMCPServer
    except Exception as e:  # pragma: no cover - runtime dependency
        raise ImportError("mcp server SDK is not installed") from e

    mcp = KDCubeMCPServer(name)

    @mcp.tool(
        name="node_status",
        description="Read health and metadata from the bundle-local Node sidecar backend.",
    )
    async def _node_status() -> dict[str, Any]:
        response = await bridge_provider().get_json("/api/projects/status")
        return {
            "ok": response.ok,
            "status": response.status,
            "data": response.data,
            "error": response.error,
        }

    @mcp.tool(
        name="search_projects",
        description="Search sample project data through the bundle-local Node sidecar backend.",
    )
    async def _search_projects(query: str = "") -> dict[str, Any]:
        response = await bridge_provider().post_json(
            "/api/projects/search",
            body={"query": str(query or "")},
        )
        return {
            "ok": response.ok,
            "status": response.status,
            "data": response.data,
            "error": response.error,
        }

    return mcp
