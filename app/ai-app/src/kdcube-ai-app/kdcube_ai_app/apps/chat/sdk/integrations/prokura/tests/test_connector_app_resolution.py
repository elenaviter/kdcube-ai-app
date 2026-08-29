# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The guarded service decides the connector app (operator rulings 2026-07-26):
its declaration carried in the portable context (bundle_call_context), or
empty = provider-wide. No hardcoded defaults, never a user pick, and the
declaration survives runtime boundaries because it rides the guaranteed
cross-runtime context."""

from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import (
    bind_current_bundle_call_context,
)
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.connector_app_resolution import (
    resolve_connector_app_id,
    set_service_connector_apps,
)


def test_service_declaration_resolves():
    with bind_current_bundle_call_context({}):
        set_service_connector_apps({"slack": "slack-demo", "google": "gmail"})
        assert resolve_connector_app_id("slack") == "slack-demo"
        assert resolve_connector_app_id("google") == "gmail"


def test_undeclared_provider_is_provider_wide():
    with bind_current_bundle_call_context({}):
        set_service_connector_apps({"slack": "slack-demo"})
        assert resolve_connector_app_id("acme-crm") == ""


def test_no_declaration_is_provider_wide():
    with bind_current_bundle_call_context({}):
        assert resolve_connector_app_id("slack") == ""


def test_blank_entries_are_ignored():
    with bind_current_bundle_call_context({}):
        set_service_connector_apps({" ": "x", "slack": "  "})
        assert resolve_connector_app_id("slack") == ""


def test_resolve_never_crashes_when_context_unset():
    """The detached agent runtime may present no bound call context. Resolve
    must read defensively and return "" (provider-wide), never raise - a raise
    here was swallowed at the agent gate and wrongly fell back to the
    agent-grant view."""
    # No bound bundle_call_context at all (get returns {} by contract).
    assert resolve_connector_app_id("slack") == ""
