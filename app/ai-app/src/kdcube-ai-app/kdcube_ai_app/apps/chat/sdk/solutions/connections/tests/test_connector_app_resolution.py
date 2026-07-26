# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The guarded service decides the connector app (operator rulings 2026-07-26):
its bound declaration, or empty = provider-wide. No hardcoded defaults, never
a user pick."""

from kdcube_ai_app.apps.chat.sdk.solutions.connections.connector_app_resolution import (
    resolve_connector_app_id,
    set_service_connector_apps,
)


def test_service_declaration_resolves():
    set_service_connector_apps({"slack": "slack-demo", "google": "gmail"})
    try:
        assert resolve_connector_app_id("slack") == "slack-demo"
        assert resolve_connector_app_id("google") == "gmail"
    finally:
        set_service_connector_apps(None)


def test_undeclared_provider_is_provider_wide():
    set_service_connector_apps({"slack": "slack-demo"})
    try:
        assert resolve_connector_app_id("acme-crm") == ""
    finally:
        set_service_connector_apps(None)


def test_no_declaration_is_provider_wide():
    set_service_connector_apps(None)
    assert resolve_connector_app_id("slack") == ""


def test_blank_entries_are_ignored():
    set_service_connector_apps({" ": "x", "slack": "  "})
    try:
        assert resolve_connector_app_id("slack") == ""
    finally:
        set_service_connector_apps(None)
