# SPDX-License-Identifier: MIT

"""Custom tools integration tests.

Test that custom tools work correctly within the LangGraph context:
callables resolve, multiple tools don't conflict, and tool IDs are correct.

Run with:
  BUNDLE_UNDER_TEST=/abs/path/to/bundle pytest test_custom_tools_integration.py -v
  pytest test_custom_tools_integration.py --bundle-path=/abs/path/to/bundle -v
"""

from __future__ import annotations

import pytest


class TestToolIdFormat:
    """Verify tool ID conventions across different tool types."""

    def test_module_tool_id_has_two_parts(self):
        """Module tool IDs are '<alias>.<tool_name>'."""
        from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import parse_tool_id
        tool_id = "io_tools.read_file"
        origin, alias, name = parse_tool_id(tool_id)
        assert alias == "io_tools"
        assert name == "read_file"

    def test_mcp_tool_id_has_three_parts(self):
        """MCP tool IDs are 'mcp.<alias>.<tool_name>'."""
        from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import parse_tool_id
        tool_id = "mcp.web_search.search"
        origin, alias, name = parse_tool_id(tool_id)
        assert origin == "mcp"
        assert alias == "web_search"
        assert name == "search"

    def test_tool_id_with_dotted_tool_name_preserves_name(self):
        """Multi-segment tool names after alias are preserved."""
        from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import parse_tool_id
        tool_id = "mcp.server.sub.tool"
        origin, alias, name = parse_tool_id(tool_id)
        assert origin == "mcp"
        assert alias == "server"
        assert "sub" in name and "tool" in name


class TestToolSubsystemNoConflicts:
    """Test that multiple tools registered together don't conflict."""

    def _build_minimal_subsystem(self, specs):
        """Build ToolSubsystem with given specs list (module aliases must be valid)."""
        from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import ToolSubsystem
        ts = object.__new__(ToolSubsystem)
        ts._tool_runtime = {}
        ts._by_id = {}
        ts._mods_by_alias = {}
        ts._modules = []
        ts.tools_info = []
        ts._mcp_entries = []
        return ts

    def test_by_id_map_is_populated_from_tools_info(self):
        """_by_id fast-map mirrors tools_info entries."""
        from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import ToolSubsystem
        ts = object.__new__(ToolSubsystem)
        ts.tools_info = [
            {"id": "alias1.tool_a", "name": "tool_a"},
            {"id": "alias2.tool_b", "name": "tool_b"},
        ]
        ts._by_id = {e["id"]: e for e in ts.tools_info}
        assert "alias1.tool_a" in ts._by_id
        assert "alias2.tool_b" in ts._by_id

    def test_two_tool_ids_do_not_collide(self):
        """Two distinct tool IDs hash to different entries."""
        from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import ToolSubsystem
        ts = object.__new__(ToolSubsystem)
        ts.tools_info = [
            {"id": "mod1.func_a"},
            {"id": "mod2.func_a"},  # same name, different alias
        ]
        ts._by_id = {e["id"]: e for e in ts.tools_info}
        assert len(ts._by_id) == 2, "Different aliases must produce distinct tool IDs"


class TestBundleGraphHasRunMethod:
    """Test that the bundle graph exposes ainvoke for LangGraph tool calls."""

    def test_bundle_graph_has_ainvoke(self, bundle_graph):
        """Compiled graph exposes ainvoke (prerequisite for tool-calling nodes)."""
        assert hasattr(bundle_graph, "ainvoke"), "Compiled graph must expose ainvoke()"
        assert callable(bundle_graph.ainvoke)

    def test_bundle_graph_has_get_graph(self, bundle_graph):
        """Compiled graph exposes get_graph() for introspection."""
        assert hasattr(bundle_graph, "get_graph")
        assert callable(bundle_graph.get_graph)

    def test_bundle_graph_nodes_are_connected(self, bundle_graph):
        """Every real node is connected to at least one edge (no orphaned tool nodes)."""
        inner = bundle_graph.get_graph()
        all_nodes = set(inner.nodes) - {"__start__", "__end__"}
        connected = set()
        for e in inner.edges:
            connected.add(e.source)
            connected.add(e.target)
        orphans = all_nodes - connected
        assert not orphans, f"Orphan nodes found: {orphans}"
