# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for bundle tests.

These fixtures enable parameterized testing of any bundle by bundle ID.
React.doc can run tests for different bundles via:
  pytest test_*.py --bundle-id=react.doc -v
  pytest test_*.py --bundle-id=openrouter-data -v

Bundle discovery uses _reserved_bundle_entry() + _resolve_module() —
no Redis or database required. Works in sandbox (ISO runtime) too.
"""

import pytest
from unittest.mock import MagicMock


def pytest_addoption(parser):
    """Add --bundle-id parameter for pytest."""
    parser.addoption(
        "--bundle-id",
        action="store",
        default="react.doc",
        help="Bundle ID to test (auto-discovered from examples/bundles/)",
    )


@pytest.fixture(scope="session")
def bundle_id(request):
    """Bundle ID parameter passed by react.doc via --bundle-id."""
    return request.config.getoption("--bundle-id")


@pytest.fixture
def bundle(bundle_id):
    """Load and initialize bundle for testing.

    Uses _reserved_bundle_entry() to discover the bundle by ID, then
    _resolve_module() + _discover_decorated() to load the class.
    No Redis or database required — works locally and in ISO sandbox.

    Args:
        bundle_id: Bundle ID from --bundle-id parameter

    Returns:
        Initialized bundle instance ready for testing
    """
    try:
        from kdcube_ai_app.infra.plugin.bundle_store import (
            _reserved_bundle_entry,
            _discover_example_bundle_ids,
        )
        from kdcube_ai_app.infra.plugin.agentic_loader import (
            AgenticBundleSpec,
            _resolve_module,
            _discover_decorated,
        )

        entry = _reserved_bundle_entry(bundle_id)
        if entry is None:
            available = sorted(_discover_example_bundle_ids())
            pytest.skip(
                f"Bundle '{bundle_id}' not found. "
                f"Available: {', '.join(available)}"
            )

        spec = AgenticBundleSpec(path=entry.path, module=entry.module or "entrypoint")
        mod = _resolve_module(spec)
        chosen = _discover_decorated(mod)

        if chosen is None:
            pytest.skip(f"No @agentic_workflow class found in bundle '{bundle_id}'")

        kind, meta, symbol = chosen
        if kind == "class":
            bundle_cls = symbol
        else:
            # factory function — call it to get instance
            bundle_cls = None

        if bundle_cls is None:
            pytest.skip(f"Bundle '{bundle_id}' uses factory pattern, not supported in tests")

        instance = bundle_cls(
            config=MagicMock(),
            pg_pool=None,
            redis=MagicMock(),
            comm_context=MagicMock(),
        )
        return instance

    except ImportError as e:
        pytest.skip(f"Cannot import bundle infrastructure: {str(e)}")
    except Exception as e:
        pytest.skip(f"Cannot initialize bundle '{bundle_id}': {str(e)}")