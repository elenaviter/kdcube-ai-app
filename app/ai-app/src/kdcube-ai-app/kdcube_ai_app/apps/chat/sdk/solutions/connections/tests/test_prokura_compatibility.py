# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The transitional KDCube namespace must not duplicate Prokura state."""

from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("legacy_name", "prokura_name"),
    [
        (
            "kdcube_ai_app.apps.chat.sdk.solutions.connections.authority_registry",
            "prokura.authority_registry",
        ),
        (
            "kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.cards.model",
            "prokura.delegated_credentials.cards.model",
        ),
        (
            "kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.catalog.models",
            "prokura.delegated_credentials.catalog.models",
        ),
        (
            "kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.models",
            "prokura.delegated_to_kdcube.models",
        ),
        (
            "kdcube_ai_app.apps.chat.sdk.solutions.connections.hub.edges",
            "prokura.hub.edges",
        ),
    ],
)
def test_legacy_leaf_import_is_the_prokura_module(legacy_name: str, prokura_name: str):
    assert import_module(legacy_name) is import_module(prokura_name)
