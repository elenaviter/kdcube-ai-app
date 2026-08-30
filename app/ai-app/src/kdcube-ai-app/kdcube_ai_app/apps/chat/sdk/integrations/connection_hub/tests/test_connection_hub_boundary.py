# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Portable Connection Hub policy stays distinct from KDCube host bindings."""

from __future__ import annotations

from connection_hub.hub.edges import ConnectionEdgeStore as PortableConnectionEdgeStore

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.automation_access import (
    AutomationAccessService,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.cards.persistence import (
    DurableCardPersistence,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.hub import ConnectionEdgeStore
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.hub.provider_impl import (
    ConnectionHubProvider,
)


def test_portable_contract_is_reexported_without_a_second_implementation():
    assert ConnectionEdgeStore is PortableConnectionEdgeStore


def test_kdcube_ports_are_explicit_host_types():
    assert AutomationAccessService.__module__.startswith(
        "kdcube_ai_app.apps.chat.sdk.integrations.connection_hub."
    )
    assert DurableCardPersistence.__module__.startswith(
        "kdcube_ai_app.apps.chat.sdk.integrations.connection_hub."
    )
    assert ConnectionHubProvider.__module__.startswith(
        "kdcube_ai_app.apps.chat.sdk.integrations.connection_hub."
    )
