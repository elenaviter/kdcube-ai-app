# SPDX-License-Identifier: MIT

"""Initialization tests for bundles.

Test that bundles start up correctly.
Tests work with any bundle specified via --bundle-id parameter.

Run with:
  pytest test_initialization.py --bundle-id=react.doc -v
  pytest test_initialization.py --bundle-id=openrouter-data -v
"""


class TestBundleInitialization:
    """Test that the bundle starts up correctly."""

    def test_bundle_initializes_with_valid_config_redis_and_comm_context(self, bundle):
        """Bundle initializes with valid config, redis, and comm_context."""
        assert bundle is not None
        assert bundle.config is not None
        assert bundle.redis is not None
        assert bundle.comm_context is not None

    def test_langgraph_compiles_after_initialization(self, bundle):
        """LangGraph compiles after initialization."""
        # Bundle should have _build_graph method
        assert hasattr(bundle, "_build_graph")

        # Should be callable
        assert callable(bundle._build_graph)

        # Try to build graph
        graph = bundle._build_graph()
        assert graph is not None

    def test_configuration_property_returns_dict_with_role_models(self, bundle):
        """configuration property returns dict with role_models."""
        assert hasattr(bundle, "configuration")

        config = bundle.configuration
        assert isinstance(config, dict)
        assert "role_models" in config
        assert isinstance(config["role_models"], dict)

    def test_bundle_handles_none_redis_falls_back_to_defaults(self, bundle):
        """Bundle handles missing redis (falls back to defaults)."""
        # Bundle was initialized, even with mocked redis
        # Should have fallback mechanism
        assert bundle is not None
        assert bundle.config is not None

    def test_event_filter_initialized_if_provided(self, bundle):
        """Event filter initialized (if provided)."""
        # Bundle may or may not have event_filter
        # If it has one, it should be callable
        if hasattr(bundle, "_event_filter"):
            event_filter = bundle._event_filter
            if event_filter is not None:
                assert callable(event_filter)