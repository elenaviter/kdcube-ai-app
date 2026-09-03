from app_foundation import mcp as foundation_mcp

from kdcube_ai_app.apps.chat.sdk.runtime.mcp import client as compatibility


def test_kdcube_mcp_client_surface_reexports_app_foundation_contract() -> None:
    assert compatibility.MCP_CLIENT_MODE_AUTO == foundation_mcp.MCP_CLIENT_MODE_AUTO
    assert compatibility.open_mcp_client is foundation_mcp.open_mcp_client
    assert compatibility.mcp_tool_schema is foundation_mcp.mcp_tool_schema
    assert (
        compatibility.normalize_mcp_tool_result
        is foundation_mcp.normalize_mcp_tool_result
    )
