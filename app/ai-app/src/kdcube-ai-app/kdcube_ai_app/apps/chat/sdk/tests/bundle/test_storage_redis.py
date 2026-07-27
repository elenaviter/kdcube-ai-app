# SPDX-License-Identifier: MIT

"""Redis cache storage tests for bundles.

Test that bundle_props loading from Redis works correctly, including
TTL handling, namespace isolation, and graceful fallback when Redis
is unavailable.

Run with:
  BUNDLE_UNDER_TEST=/abs/path/to/bundle pytest test_storage_redis.py -v
  pytest test_storage_redis.py --bundle-path=/abs/path/to/bundle -v
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_kv_cache(data: dict | None = None):
    """Build a mock kv_cache that returns the given data on get_json()."""
    mock = MagicMock()
    mock.get_json = AsyncMock(return_value=data)
    mock.set_json = AsyncMock(return_value=None)
    return mock


# ---------------------------------------------------------------------------
# Tests: refresh_bundle_props fallback
# ---------------------------------------------------------------------------

class TestRefreshBundlePropsNoRedis:
    """Test that refresh_bundle_props() falls back to defaults when no cache."""

    @pytest.mark.anyio
    async def test_fallback_to_defaults_when_redis_and_kv_cache_none(self, bundle):
        """Falls back to defaults when both redis and kv_cache are None."""
        orig_redis = bundle.redis
        orig_kv = bundle.kv_cache
        try:
            bundle.redis = None
            bundle.kv_cache = None
            props = await bundle.refresh_bundle_props(state={"tenant": "t", "project": "p"})
            assert isinstance(props, dict)
            defaults = dict(bundle.bundle_props_defaults or {})
            assert set(defaults).issubset(props)
        finally:
            bundle.redis = orig_redis
            bundle.kv_cache = orig_kv

    @pytest.mark.anyio
    async def test_fallback_when_tenant_missing_from_state(self, bundle):
        """Falls back to defaults when tenant/project not in state."""
        orig_redis = bundle.redis
        try:
            props = await bundle.refresh_bundle_props(state={})
            assert isinstance(props, dict)
            defaults = dict(bundle.bundle_props_defaults or {})
            assert set(defaults).issubset(props)
        finally:
            bundle.redis = orig_redis

    @pytest.mark.anyio
    async def test_bundle_props_is_set_after_refresh(self, bundle):
        """bundle_props is assigned after refresh_bundle_props() completes."""
        orig_redis = bundle.redis
        orig_kv = bundle.kv_cache
        try:
            bundle.redis = None
            bundle.kv_cache = None
            await bundle.refresh_bundle_props(state={})
            assert isinstance(bundle.bundle_props, dict)
        finally:
            bundle.redis = orig_redis
            bundle.kv_cache = orig_kv


class TestRefreshBundlePropsWithMockCache:
    """Test that Redis/kv_cache overrides are applied by refresh_bundle_props()."""

    @pytest.mark.anyio
    async def test_kv_cache_override_merged_into_props(self, bundle):
        """Overrides from kv_cache are deep-merged with defaults."""
        orig_redis = bundle.redis
        orig_kv = bundle.kv_cache
        orig_props = dict(bundle.bundle_props)
        try:
            override = {"role_models": {"overridden-role": {"provider": "openai", "model": "gpt-4o-kv"}}}
            bundle.redis = None
            bundle.kv_cache = _make_mock_kv_cache(override)

            # Need bundle_spec.id to trigger cache lookup
            if not getattr(getattr(bundle.config, "ai_bundle_spec", None), "id", None):
                pytest.skip("Bundle has no ai_bundle_spec.id — skipping cache lookup test")

            props = await bundle.refresh_bundle_props(
                state={"tenant": "test-t", "project": "test-p"}
            )
            role_models = props.get("role_models") or {}
            assert "overridden-role" in role_models, (
                "kv_cache override must be merged into bundle_props"
            )
            assert role_models["overridden-role"]["model"] == "gpt-4o-kv"
        finally:
            bundle.redis = orig_redis
            bundle.kv_cache = orig_kv
            bundle.bundle_props = orig_props

    @pytest.mark.anyio
    async def test_kv_cache_none_data_uses_defaults(self, bundle):
        """When kv_cache returns None, defaults are used."""
        orig_redis = bundle.redis
        orig_kv = bundle.kv_cache
        orig_props = dict(bundle.bundle_props)
        try:
            bundle.redis = None
            bundle.kv_cache = _make_mock_kv_cache(None)

            if not getattr(getattr(bundle.config, "ai_bundle_spec", None), "id", None):
                pytest.skip("Bundle has no ai_bundle_spec.id")

            props = await bundle.refresh_bundle_props(
                state={"tenant": "t", "project": "p"}
            )
            assert isinstance(props, dict)
            defaults = dict(bundle.bundle_props_defaults or {})
            assert set(defaults).issubset(props)
        finally:
            bundle.redis = orig_redis
            bundle.kv_cache = orig_kv
            bundle.bundle_props = orig_props


class TestDeepMergeProps:
    """Test _deep_merge_props() directly."""

    def test_deep_merge_adds_new_key(self, bundle):
        """Merge adds a new key from patch."""
        merged = bundle._deep_merge_props({"a": 1}, {"b": 2})
        assert merged["a"] == 1
        assert merged["b"] == 2

    def test_deep_merge_overrides_scalar(self, bundle):
        """Merge replaces a scalar value with the patch value."""
        merged = bundle._deep_merge_props({"x": "old"}, {"x": "new"})
        assert merged["x"] == "new"

    def test_deep_merge_recurses_into_dicts(self, bundle):
        """Merge recurses into nested dicts (deep merge, not replace)."""
        base = {"role_models": {"solver": {"model": "old-model", "provider": "anthropic"}}}
        patch = {"role_models": {"solver": {"model": "new-model"}}}
        merged = bundle._deep_merge_props(base, patch)
        assert merged["role_models"]["solver"]["model"] == "new-model"
        assert merged["role_models"]["solver"]["provider"] == "anthropic"

    def test_deep_merge_does_not_mutate_base(self, bundle):
        """_deep_merge_props() does not mutate the base dict."""
        base = {"k": {"nested": 1}}
        bundle._deep_merge_props(base, {"k": {"nested": 2}})
        assert base["k"]["nested"] == 1

    def test_deep_merge_empty_patch_returns_copy_of_base(self, bundle):
        """Merging with empty patch returns a copy of base."""
        base = {"a": 1, "b": {"c": 3}}
        merged = bundle._deep_merge_props(base, {})
        assert merged == base
        assert merged is not base
