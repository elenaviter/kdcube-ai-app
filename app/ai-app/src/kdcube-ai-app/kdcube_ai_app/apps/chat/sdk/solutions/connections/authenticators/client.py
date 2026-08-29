# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""KDCube app-operation adapter for Prokura request authentication."""

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.infra.bundle_operations import call_bundle_operation
from prokura.authenticators.client import (
    DEFAULT_CONNECTION_HUB_BUNDLE_ID,
    REQUEST_AUTHENTICATE_OPERATION,
    ConnectionHubAuthenticatorsClient as ProkuraAuthenticatorsClient,
)


class ConnectionHubAuthenticatorsClient(ProkuraAuthenticatorsClient):
    """Authenticator client bound to KDCube's app-operation bridge."""

    def __init__(
        self,
        *,
        connection_hub_bundle_id: str = DEFAULT_CONNECTION_HUB_BUNDLE_ID,
        tenant: str | None = None,
        project: str | None = None,
    ) -> None:
        super().__init__(
            connection_hub_bundle_id=connection_hub_bundle_id,
            tenant=tenant,
            project=project,
            operation_caller=call_bundle_operation,
        )


__all__ = [
    "ConnectionHubAuthenticatorsClient",
    "DEFAULT_CONNECTION_HUB_BUNDLE_ID",
    "REQUEST_AUTHENTICATE_OPERATION",
]
