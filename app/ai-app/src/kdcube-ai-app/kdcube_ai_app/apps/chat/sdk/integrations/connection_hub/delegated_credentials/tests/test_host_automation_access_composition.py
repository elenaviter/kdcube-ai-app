# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Host composition must preserve every portable delegated-access port."""

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.automation_access import (
    AutomationAccessService,
)


def test_host_forwards_dynamic_resource_overlay_provider() -> None:
    async def overlay(owner_subject: str):
        return {"owner_subject": owner_subject}

    service = AutomationAccessService(
        redis=object(),
        tenant="tenant",
        project="project",
        config=object(),
        grant_store=object(),
        resource_overlay_provider=overlay,
    )

    assert service._resource_overlay_provider is overlay
