# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from connection_hub.authority_projection import (
    authority_has_platform_privilege,
    project_execution_authority,
)


def test_project_execution_authority_charges_projected_platform_user():
    projection = project_execution_authority(
        {
            "actor_user_id": "telegram_100200300",
            "platform_user_id": "a1b2c3d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
            "platform_roles": ["kdcube:role:super-admin"],
            "platform_permissions": ["memories:read"],
        },
        fallback_user_id="telegram_100200300",
    )

    assert projection.actor_user_id == "telegram_100200300"
    assert projection.economics_user_id == "a1b2c3d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d"
    assert projection.roles == ("kdcube:role:super-admin",)
    assert projection.permissions == ("memories:read",)
    assert projection.budget_bypass is True


def test_project_execution_authority_keeps_unlinked_actor_local_and_roleless():
    projection = project_execution_authority(
        {},
        actor_user_id="telegram_100200300",
        fallback_user_id="telegram_100200300",
    )

    assert projection.actor_user_id == "telegram_100200300"
    assert projection.economics_user_id == "telegram_100200300"
    assert projection.roles == ()
    assert projection.budget_bypass is None


def test_authority_has_platform_privilege_uses_central_platform_role_set():
    assert authority_has_platform_privilege(["kdcube:role:admin"])
    assert not authority_has_platform_privilege(["kdcube:role:registered"])
