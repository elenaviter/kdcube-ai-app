# SPDX-License-Identifier: MIT

"""Configuration tests for bundles (Type 2).

Test that bundle settings work correctly.
Tests work with any bundle selected by folder.

Run with:
  BUNDLE_UNDER_TEST=/abs/path/to/bundle pytest test_configuration.py -v
  pytest test_configuration.py --bundle-path=/abs/path/to/bundle -v
"""

from __future__ import annotations

import pytest


class TestBundleConfiguration:
    """Test that bundle settings work correctly."""

    def test_default_role_models_applied_from_code(self, bundle):
        """Declared model roles contain a model slug."""
        config = bundle.configuration
        if "role_models" not in config:
            pytest.skip("App declares no model roles")
        role_models = config.get("role_models") or {}
        assert role_models, "Declared role_models must not be empty"
        for role_key, role_val in role_models.items():
            assert isinstance(role_val, dict), (
                f"role_models[{role_key!r}] must be a dict, got {type(role_val)}"
            )
            assert "model" in role_val, (
                f"role_models[{role_key!r}] must have a 'model' key"
            )

    def test_bundle_props_initialized_from_configuration(self, bundle):
        """bundle_props is set from configuration defaults at init time."""
        assert isinstance(bundle.bundle_props, dict)
        defaults = dict(bundle.bundle_props_defaults or {})
        assert set(defaults).issubset(bundle.bundle_props)

    def test_bundle_prop_returns_correct_model_for_existing_role(self, bundle):
        """bundle_prop() resolves a nested config path correctly."""
        config = bundle.configuration
        role_models = config.get("role_models") or {}

        # Pick the first role that has no dot in its name so path navigation works
        flat_role = next(
            (k for k in role_models if "." not in k),
            next(iter(role_models), None),
        )
        if flat_role is None:
            pytest.skip("No role_models defined in configuration")

        # bundle_prop uses dot-separated path navigation
        model_val = bundle.bundle_prop(f"role_models.{flat_role}.model")
        expected = role_models[flat_role]["model"]
        assert model_val == expected, (
            f"bundle_prop('role_models.{flat_role}.model') returned {model_val!r}, "
            f"expected {expected!r}"
        )

    def test_missing_config_path_returns_none_no_key_error(self, bundle):
        """Missing config paths return None (no KeyError)."""
        result = bundle.bundle_prop("nonexistent.deeply.nested.path")
        assert result is None

    def test_missing_config_path_returns_custom_default(self, bundle):
        """bundle_prop default parameter is returned for missing paths."""
        sentinel = object()
        result = bundle.bundle_prop("nonexistent.key", default=sentinel)
        assert result is sentinel

    def test_external_config_override_respected(self, bundle):
        """Directly set bundle_props overrides are reflected by bundle_prop()."""
        original_props = dict(bundle.bundle_props)
        try:
            bundle.bundle_props = {
                "role_models": {
                    "test-role": {"provider": "test", "model": "test-model-xyz"}
                }
            }
            result = bundle.bundle_prop("role_models.test-role.model")
            assert result == "test-model-xyz"
        finally:
            bundle.bundle_props = original_props

    @pytest.mark.anyio
    async def test_refresh_bundle_props_falls_back_to_defaults_when_no_cache(self, bundle):
        """refresh_bundle_props() returns defaults when no Redis or kv_cache available."""
        original_redis = bundle.redis
        original_kv_cache = bundle.kv_cache
        try:
            bundle.redis = None
            bundle.kv_cache = None

            state = {
                "tenant": "test-tenant",
                "project": "test-project",
            }
            props = await bundle.refresh_bundle_props(state=state)

            assert isinstance(props, dict)
            defaults = dict(bundle.bundle_props_defaults or {})
            assert set(defaults).issubset(props)
        finally:
            bundle.redis = original_redis
            bundle.kv_cache = original_kv_cache

    @pytest.mark.anyio
    async def test_refresh_bundle_props_without_tenant_project_uses_defaults(self, bundle):
        """refresh_bundle_props() uses defaults when tenant/project not in state."""
        original_redis = bundle.redis
        try:
            # Even with redis present, missing tenant/project falls back to defaults
            state = {}
            props = await bundle.refresh_bundle_props(state=state)

            assert isinstance(props, dict)
            defaults = dict(bundle.bundle_props_defaults or {})
            assert set(defaults).issubset(props)
        finally:
            bundle.redis = original_redis

    @pytest.mark.anyio
    async def test_redis_overrides_take_precedence_over_defaults(self, bundle):
        """Bundle props set via deep merge override defaults."""
        original_props = dict(bundle.bundle_props)
        try:
            defaults = dict(bundle.bundle_props_defaults or {})
            override = {
                "role_models": {
                    "override-role": {
                        "provider": "openai",
                        "model": "gpt-4o-override",
                    }
                }
            }
            # Simulate what refresh_bundle_props does when Redis returns overrides
            merged = bundle._deep_merge_props(defaults, override)
            bundle.bundle_props = merged

            # The override role must be present
            assert bundle.bundle_prop("role_models.override-role.model") == "gpt-4o-override"
            # Original roles must still be present in bundle_props (deep merge, not replace)
            original_roles = set(defaults.get("role_models") or {})
            current_roles = set((bundle.bundle_props.get("role_models") or {}).keys())
            assert original_roles.issubset(current_roles), (
                f"Deep merge dropped roles: {original_roles - current_roles}"
            )
        finally:
            bundle.bundle_props = original_props

    def test_configuration_property_returns_dict(self, bundle):
        """configuration property returns a plain dict."""
        config = bundle.configuration
        assert isinstance(config, dict)

    def test_declared_role_models_is_a_mapping(self, bundle):
        """role_models is a mapping when the app declares it."""
        config = bundle.configuration
        if "role_models" in config:
            assert isinstance(config["role_models"], dict)
