# SPDX-License-Identifier: MIT

"""Graph tests for bundles (Type 3).

Test that the LangGraph works correctly.
Tests work with any bundle selected by folder.

Run with:
  BUNDLE_UNDER_TEST=/abs/path/to/bundle pytest test_graph.py -v
  pytest test_graph.py --bundle-path=/abs/path/to/bundle -v
"""

from __future__ import annotations

import time

class TestBundleGraph:
    """Test that the LangGraph compiles and is structurally correct."""

    def test_build_graph_returns_compiled_state_graph(self, bundle, bundle_graph):
        """_build_graph() returns a compiled StateGraph (not None)."""
        assert hasattr(bundle, "_build_graph"), "Bundle must implement _build_graph()"
        assert callable(bundle._build_graph)

        assert bundle_graph is not None

    def test_compiled_graph_stored_on_bundle(self, bundle, bundle_graph):
        """Bundle stores the compiled graph as self.graph after __init__."""
        del bundle_graph
        assert hasattr(bundle, "graph"), (
            "Bundle must store compiled graph as self.graph in __init__"
        )
        assert bundle.graph is not None

    def test_graph_has_nodes(self, bundle_graph):
        """Compiled graph has at least one node."""
        # LangGraph compiled graphs expose nodes via .nodes or get_graph()
        inner = bundle_graph.get_graph()
        assert len(inner.nodes) > 0, "Graph must have at least one node"

    def test_graph_has_edges(self, bundle_graph):
        """Compiled graph has edges connecting nodes."""
        inner = bundle_graph.get_graph()
        assert len(inner.edges) > 0, "Graph must have at least one edge"

    def test_graph_starts_from_start_node(self, bundle_graph):
        """Graph has an edge from __start__ to the first real node."""
        inner = bundle_graph.get_graph()
        edge_sources = {e.source for e in inner.edges}
        assert "__start__" in edge_sources, (
            "Graph must have an edge originating from START (__start__)"
        )

    def test_graph_ends_at_end_node(self, bundle_graph):
        """Graph has an edge to __end__."""
        inner = bundle_graph.get_graph()
        edge_targets = {e.target for e in inner.edges}
        assert "__end__" in edge_targets, (
            "Graph must have an edge leading to END (__end__)"
        )

    def test_graph_no_orphan_nodes(self, bundle_graph):
        """Every node is reachable (connected to at least one edge)."""
        inner = bundle_graph.get_graph()

        all_nodes = {n for n in inner.nodes}
        connected_nodes = set()
        for e in inner.edges:
            connected_nodes.add(e.source)
            connected_nodes.add(e.target)

        # Remove __start__ and __end__ which may not be in nodes dict
        real_nodes = all_nodes - {"__start__", "__end__"}
        orphans = real_nodes - connected_nodes
        assert not orphans, f"Orphan nodes found (not connected to any edge): {orphans}"

    def test_build_graph_does_not_raise(self, bundle_graph):
        """_build_graph() completes without raising an exception."""
        assert bundle_graph is not None

    def test_build_graph_is_fast(self, bundle, bundle_graph):
        """Graph compilation completes in a reasonable time (< 5 seconds)."""
        del bundle_graph
        start = time.time()
        bundle._build_graph()
        elapsed = time.time() - start
        assert elapsed < 5.0, (
            f"_build_graph() took {elapsed:.2f}s — expected < 5s"
        )

    def test_graph_supports_ainvoke(self, bundle_graph):
        """Compiled graph has an ainvoke method (async invocation)."""
        assert hasattr(bundle_graph, "ainvoke"), "Compiled graph must expose ainvoke()"
        assert callable(bundle_graph.ainvoke)

    def test_multiple_graph_builds_are_independent(self, bundle, bundle_graph):
        """Each call to _build_graph() returns a fresh, independent graph."""
        del bundle_graph
        graph1 = bundle._build_graph()
        graph2 = bundle._build_graph()
        # Should be different objects (not the same cached instance)
        assert graph1 is not graph2
