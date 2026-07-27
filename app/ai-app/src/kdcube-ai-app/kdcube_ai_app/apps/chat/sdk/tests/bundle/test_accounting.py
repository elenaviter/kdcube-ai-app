# SPDX-License-Identifier: MIT

"""Accounting tests for bundles (Type 7).

Test that LLM usage tracking works correctly.
Tests work with any bundle selected by folder.

Run with:
  BUNDLE_UNDER_TEST=/abs/path/to/bundle pytest test_accounting.py -v
  pytest test_accounting.py --bundle-path=/abs/path/to/bundle -v
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# AccountingContext tests
# ---------------------------------------------------------------------------

class TestAccountingContext:
    """Test that AccountingContext captures and stores events correctly."""

    def test_accounting_context_can_be_created(self):
        """AccountingContext initializes without errors."""
        from kdcube_ai_app.infra.accounting import AccountingContext
        ctx = AccountingContext()
        assert ctx is not None

    def test_accounting_context_event_cache_starts_empty(self):
        """Fresh AccountingContext has no cached events."""
        from kdcube_ai_app.infra.accounting import AccountingContext
        ctx = AccountingContext()
        assert ctx.get_cached_events() == []

    def test_accounting_context_cache_event_stores_event(self):
        """cache_event() appends to the event cache."""
        from kdcube_ai_app.infra.accounting import AccountingContext, AccountingEvent
        ctx = AccountingContext()
        ev = AccountingEvent(
            service_type="llm",
            provider="anthropic",
            model_or_service="claude-sonnet-4-6",
        )
        ctx.cache_event(ev)
        cached = ctx.get_cached_events()
        assert len(cached) == 1
        assert cached[0] is ev

    def test_accounting_context_clear_cache_empties_events(self):
        """clear_cache() removes all cached events."""
        from kdcube_ai_app.infra.accounting import AccountingContext, AccountingEvent
        ctx = AccountingContext()
        ev = AccountingEvent(service_type="llm", provider="openai", model_or_service="gpt-4o")
        ctx.cache_event(ev)
        ctx.clear_cache()
        assert ctx.get_cached_events() == []

    def test_accounting_context_update_stores_key_values(self):
        """update() stores arbitrary key-value pairs in the context."""
        from kdcube_ai_app.infra.accounting import AccountingContext
        ctx = AccountingContext()
        ctx.update(user_id="u1", session_id="s1", component="solver")
        d = ctx.to_dict()
        assert d["user_id"] == "u1"
        assert d["session_id"] == "s1"
        assert d["component"] == "solver"

    def test_accounting_context_user_id_property(self):
        """user_id property reads from internal context dict."""
        from kdcube_ai_app.infra.accounting import AccountingContext
        ctx = AccountingContext()
        ctx.user_id = "test-user"
        assert ctx.user_id == "test-user"

    def test_accounting_context_session_id_property(self):
        """session_id property reads from internal context dict."""
        from kdcube_ai_app.infra.accounting import AccountingContext
        ctx = AccountingContext()
        ctx.session_id = "sess-123"
        assert ctx.session_id == "sess-123"


# ---------------------------------------------------------------------------
# AccountingEvent tests
# ---------------------------------------------------------------------------

class TestAccountingEvent:
    """Test AccountingEvent structure."""

    def test_accounting_event_can_be_created_with_minimal_params(self):
        """AccountingEvent initializes with service_type, provider, model."""
        from kdcube_ai_app.infra.accounting import AccountingEvent
        ev = AccountingEvent(
            service_type="llm",
            provider="anthropic",
            model_or_service="claude-sonnet-4-6",
        )
        assert ev.service_type == "llm"
        assert ev.provider == "anthropic"
        assert ev.model_or_service == "claude-sonnet-4-6"

    def test_accounting_event_to_dict_contains_required_fields(self):
        """AccountingEvent.to_dict() includes all required top-level keys."""
        from kdcube_ai_app.infra.accounting import AccountingEvent
        ev = AccountingEvent(
            service_type="llm",
            provider="openai",
            model_or_service="gpt-4o",
        )
        d = ev.to_dict()
        assert isinstance(d, dict)
        for required in ("service_type", "provider", "model_or_service"):
            assert required in d, f"Missing key in AccountingEvent.to_dict(): {required}"


# ---------------------------------------------------------------------------
# Usage normalization tests
# ---------------------------------------------------------------------------

class TestUsageNormalization:
    """Test that token usage is normalized to the expected structure."""

    def test_norm_usage_dict_returns_expected_keys(self):
        """_norm_usage_dict normalizes provider-specific fields to standard keys."""
        from kdcube_ai_app.infra.service_hub.inventory import _norm_usage_dict
        raw = {"input_tokens": 100, "output_tokens": 50}
        norm = _norm_usage_dict(raw)
        assert "prompt_tokens" in norm
        assert "completion_tokens" in norm
        assert "total_tokens" in norm

    def test_norm_usage_dict_sums_total_tokens(self):
        """total_tokens = prompt_tokens + completion_tokens."""
        from kdcube_ai_app.infra.service_hub.inventory import _norm_usage_dict
        raw = {"input_tokens": 120, "output_tokens": 80}
        norm = _norm_usage_dict(raw)
        assert norm["total_tokens"] == norm["prompt_tokens"] + norm["completion_tokens"]

    def test_norm_usage_dict_handles_openai_style_keys(self):
        """Handles OpenAI-style prompt_tokens / completion_tokens keys."""
        from kdcube_ai_app.infra.service_hub.inventory import _norm_usage_dict
        raw = {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}
        norm = _norm_usage_dict(raw)
        assert norm["prompt_tokens"] == 200
        assert norm["completion_tokens"] == 100
        assert norm["total_tokens"] == 300

    def test_norm_usage_dict_handles_empty_input(self):
        """Empty usage dict returns zero-valued structure."""
        from kdcube_ai_app.infra.service_hub.inventory import _norm_usage_dict
        norm = _norm_usage_dict({})
        assert isinstance(norm, dict)
        assert norm.get("total_tokens", 0) >= 0


# ---------------------------------------------------------------------------
# EconomicsLimitException tests
# ---------------------------------------------------------------------------

class TestEconomicsLimitException:
    """Test that EconomicsLimitException has the expected attributes."""

    def test_economics_limit_exception_has_code_attribute(self):
        """EconomicsLimitException stores the error code."""
        from kdcube_ai_app.apps.chat.sdk.infra.economics.policy import EconomicsLimitException
        exc = EconomicsLimitException("quota exceeded", code="quota")
        assert exc.code == "quota"
        assert str(exc) == "quota exceeded"

    def test_economics_limit_exception_has_data_attribute(self):
        """EconomicsLimitException stores optional data dict."""
        from kdcube_ai_app.apps.chat.sdk.infra.economics.policy import EconomicsLimitException
        exc = EconomicsLimitException("over budget", code="over_budget", data={"remaining": 0})
        assert exc.data["remaining"] == 0

    def test_economics_limit_exception_data_defaults_to_empty_dict(self):
        """EconomicsLimitException.data defaults to {} when not provided."""
        from kdcube_ai_app.apps.chat.sdk.infra.economics.policy import EconomicsLimitException
        exc = EconomicsLimitException("limit", code="limit")
        assert exc.data == {}

    def test_economics_limit_exception_is_runtime_error(self):
        """EconomicsLimitException is a subclass of RuntimeError."""
        from kdcube_ai_app.apps.chat.sdk.infra.economics.policy import EconomicsLimitException
        exc = EconomicsLimitException("err", code="c")
        assert isinstance(exc, RuntimeError)


# ---------------------------------------------------------------------------
# Bundle accounting integration (via bundle fixture)
# ---------------------------------------------------------------------------

class TestBundleAccounting:
    """Test that the bundle exposes accounting-related infrastructure."""

    def test_bundle_has_models_service(self, bundle):
        """Bundle exposes models_service for LLM calls."""
        assert hasattr(bundle, "models_service"), (
            "Bundle must expose models_service for LLM accounting"
        )
        assert bundle.models_service is not None

    def test_bundle_has_config_with_role_models(self, bundle):
        """Declared LLM roles remain available for accounting."""
        declared = (bundle.configuration or {}).get("role_models")
        if declared is None:
            pytest.skip("App declares no model roles")
        assert hasattr(bundle.config, "role_models")
        role_models = bundle.config.role_models or {}
        assert role_models, "Declared role_models must not be empty"

    def test_price_table_is_accessible(self):
        """price_table() returns a dict-like pricing configuration."""
        try:
            from kdcube_ai_app.infra.accounting.usage import price_table
            table = price_table()
            assert table is not None
        except ImportError:
            pytest.skip("price_table not available in this environment")

    def test_anthropic_price_lookup_supports_aliases(self):
        from kdcube_ai_app.infra.accounting.usage import _find_llm_price

        sonnet = _find_llm_price("anthropic", "sonnet")
        opus = _find_llm_price("anthropic", "opus")
        haiku = _find_llm_price("anthropic", "haiku")

        assert sonnet is not None
        assert sonnet["model"] == "claude-sonnet-4-6"
        assert opus is not None
        assert opus["model"] == "claude-opus-4-6"
        assert haiku is not None
        assert haiku["model"] == "claude-haiku-4-5-20251001"
