# SPDX-License-Identifier: MIT

"""Model routing tests for bundles (Type 9).

Test that model selection (default / config / Redis override) works correctly.
Tests work with any bundle selected by folder.

Run with:
  BUNDLE_UNDER_TEST=/abs/path/to/bundle pytest test_model_routing.py -v
  pytest test_model_routing.py --bundle-path=/abs/path/to/bundle -v
"""

from __future__ import annotations

import pytest


class TestModelRouting:
    """Test that model selection works correctly."""

    def test_default_model_present_in_role_models(self, bundle):
        """An app that declares model roles gives each one a model slug."""
        config = bundle.configuration
        if "role_models" not in config:
            pytest.skip("App declares no model roles")
        role_models = config.get("role_models") or {}
        assert role_models, "Declared role_models must not be empty"
        for role, spec in role_models.items():
            assert spec.get("model"), f"role_models[{role!r}] must have a non-empty 'model' key"

    def test_config_role_models_applied_at_init(self, bundle):
        """Config role_models are applied to bundle.config during initialization."""
        config_role_models = (bundle.configuration or {}).get("role_models") or {}
        for role in config_role_models:
            assert role in (bundle.config.role_models or {}), (
                f"Role '{role}' from configuration not applied to config.role_models"
            )

    def test_bundle_props_override_changes_model_slug(self, bundle):
        """Directly overriding bundle_props changes the model returned by bundle_prop()."""
        original = dict(bundle.bundle_props)
        try:
            bundle.bundle_props = {
                "role_models": {
                    "test-override-role": {"provider": "openai", "model": "gpt-4o-override-xyz"}
                }
            }
            result = bundle.bundle_prop("role_models.test-override-role.model")
            assert result == "gpt-4o-override-xyz"
        finally:
            bundle.bundle_props = original

    def test_config_set_role_models_updates_role_models(self, bundle):
        """config.set_role_models() updates the role_models dict."""
        original = dict(bundle.config.role_models or {})
        try:
            new_models = dict(original)
            new_models["__test_routing_role__"] = {"provider": "anthropic", "model": "claude-haiku-4-5"}
            bundle.config.set_role_models(new_models)
            assert bundle.config.role_models.get("__test_routing_role__", {}).get("model") == "claude-haiku-4-5"
        finally:
            bundle.config.set_role_models(original)

    def test_deep_merge_preserves_existing_roles(self, bundle):
        """_deep_merge_props() adds new role without removing existing roles."""
        original = dict(bundle.bundle_props)
        try:
            defaults = dict(bundle.bundle_props_defaults or {})
            override = {"role_models": {"new-role": {"provider": "openai", "model": "gpt-4o"}}}
            merged = bundle._deep_merge_props(defaults, override)

            # All original roles must still be present
            orig_roles = set((defaults.get("role_models") or {}).keys())
            merged_roles = set((merged.get("role_models") or {}).keys())
            assert orig_roles.issubset(merged_roles), (
                f"Deep merge dropped roles: {orig_roles - merged_roles}"
            )
            # New role must be added
            assert "new-role" in merged_roles
        finally:
            bundle.bundle_props = original

    def test_bundle_has_models_service_router(self, bundle):
        """bundle.models_service has a router attribute for model dispatch."""
        svc = bundle.models_service
        assert hasattr(svc, "router"), "ModelServiceBase must expose a router"

    def test_config_role_models_is_dict(self, bundle):
        """config.role_models is always a dict (never None after init)."""
        assert isinstance(bundle.config.role_models, dict)
