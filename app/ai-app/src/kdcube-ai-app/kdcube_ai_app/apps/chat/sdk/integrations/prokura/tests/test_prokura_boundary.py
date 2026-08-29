# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Portable Prokura policy stays distinct from KDCube host bindings."""

from __future__ import annotations

from prokura.hub.edges import ConnectionEdgeStore as ProkuraConnectionEdgeStore

from kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_credentials.automation_access import (
    AutomationAccessService,
)
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.delegated_credentials.cards.persistence import (
    DurableCardPersistence,
)
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.hub import ConnectionEdgeStore
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.hub.provider_impl import (
    ConnectionHubProvider,
)


def test_portable_contract_is_reexported_without_a_second_implementation():
    assert ConnectionEdgeStore is ProkuraConnectionEdgeStore


def test_kdcube_ports_are_explicit_host_types():
    assert AutomationAccessService.__module__.startswith(
        "kdcube_ai_app.apps.chat.sdk.integrations.prokura."
    )
    assert DurableCardPersistence.__module__.startswith(
        "kdcube_ai_app.apps.chat.sdk.integrations.prokura."
    )
    assert ConnectionHubProvider.__module__.startswith(
        "kdcube_ai_app.apps.chat.sdk.integrations.prokura."
    )
