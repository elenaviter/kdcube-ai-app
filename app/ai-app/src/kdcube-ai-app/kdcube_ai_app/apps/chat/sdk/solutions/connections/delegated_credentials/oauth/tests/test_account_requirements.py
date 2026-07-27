# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The consent page must surface which connected accounts a requested scope
needs, and let the operator connect/upgrade in place — not dead-end when a
required provider is unconnected. These tests pin that resolution + render."""
from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.oauth.account_requirements import (
    accounts_needed_for_scopes,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.oauth.consent import (
    _render_accounts_needed_panel,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.models import (
    DelegatedToKdcubeConfig,
)


def _config() -> DelegatedToKdcubeConfig:
    return DelegatedToKdcubeConfig.from_config(
        {
            "providers": {
                "google": {
                    "label": "Google",
                    "enabled": True,
                    "connector_apps": {
                        "gmail": {
                            "label": "Gmail",
                            "enabled": True,
                            "allowed_claims": ["gmail:read", "gmail:send", "sheets:read", "sheets:write"],
                        }
                    },
                    "claims": {
                        "gmail:read": {}, "gmail:send": {},
                        "sheets:read": {}, "sheets:write": {},
                    },
                },
                "slack": {
                    "label": "Slack",
                    "enabled": True,
                    "connector_apps": {
                        "slack-demo": {"enabled": True, "allowed_claims": ["slack:search"]}
                    },
                    "claims": {"slack:search": {}},
                },
            }
        }
    )


def _connect_url(provider_id, connector_app_id, claims):
    return f"hub://{provider_id}/{connector_app_id}?claims={','.join(claims)}"


def test_needed_provider_with_no_account_is_not_connected_and_carries_connect_url():
    result = accounts_needed_for_scopes(
        ["sheets:read", "sheets:write", "slack:search"],
        config=_config(),
        connected_accounts=[
            {"provider_id": "slack", "account_id": "T1", "label": "elena @ NestLogic", "claims": ["slack:search"]},
        ],
        connect_url_builder=_connect_url,
    )
    by_provider = {row.provider_id: row for row in result.providers}

    google = by_provider["google"]
    assert google.status() == "not_connected"
    assert google.needed_claims == ("sheets:read", "sheets:write")
    assert google.missing_claims == ("sheets:read", "sheets:write")
    # connector app is broker-selected: the one whose allowed_claims cover the need
    assert google.connector_app_id == "gmail"
    assert google.connect_url == "hub://google/gmail?claims=sheets:read,sheets:write"

    slack = by_provider["slack"]
    assert slack.status() == "connected"
    assert slack.missing_claims == ()
    assert slack.connect_url == ""  # nothing to connect
    assert result.has_gap is True


def test_connected_but_missing_claim_is_needs_access():
    # Google account connected for gmail only; sheets scopes still missing.
    result = accounts_needed_for_scopes(
        ["sheets:read", "sheets:write"],
        config=_config(),
        connected_accounts=[
            {"provider_id": "google", "account_id": "G1", "label": "Elena Viter", "claims": ["gmail:read"]},
        ],
        connect_url_builder=_connect_url,
    )
    google = result.providers[0]
    assert google.status() == "needs_access"
    assert google.connected is True
    assert google.missing_claims == ("sheets:read", "sheets:write")
    assert google.connect_url == "hub://google/gmail?claims=sheets:read,sheets:write"


def test_fully_satisfied_has_no_gap_and_no_connect_url():
    result = accounts_needed_for_scopes(
        ["sheets:read"],
        config=_config(),
        connected_accounts=[
            {"provider_id": "google", "account_id": "G1", "label": "Elena Viter", "claims": ["gmail:read", "sheets:read"]},
        ],
        connect_url_builder=_connect_url,
    )
    assert result.has_gap is False
    assert result.providers[0].status() == "connected"
    assert result.providers[0].connect_url == ""


def test_door_claims_without_a_provider_vocab_are_unresolved_not_invented():
    # `mail:read` is a door claim owned by no provider's claim vocabulary here;
    # it must not be invented into a provider row.
    result = accounts_needed_for_scopes(
        ["mail:read", "sheets:read"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
    )
    assert [row.provider_id for row in result.providers] == ["google"]
    assert result.unresolved_claims == ("mail:read",)


def test_door_claim_maps_to_provider_and_merges_into_its_row():
    # mail:read is a door claim (no provider owns it in the vocabulary); the
    # door mapping resolves it to google/gmail:read, which must MERGE into the
    # same Google row as sheets:read rather than becoming unresolved.
    result = accounts_needed_for_scopes(
        ["mail:read", "sheets:read"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
        door_claim_providers={"mail:read": [("google", ["gmail:read"])]},
    )
    assert [row.provider_id for row in result.providers] == ["google"]
    google = result.providers[0]
    assert google.needed_claims == ("gmail:read", "sheets:read")
    assert result.unresolved_claims == ()
    # least-privilege connect covers both door-backed and direct claims
    assert google.connect_url == "hub://google/gmail?claims=gmail:read,sheets:read"


def test_door_claim_provider_not_in_config_is_ignored():
    result = accounts_needed_for_scopes(
        ["mail:read"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
        door_claim_providers={"mail:read": [("nonesuch", ["x:read"])]},
    )
    assert result.providers == ()
    # token was matched by the door map (consumed), so it is not "unresolved"
    assert result.unresolved_claims == ()


def test_missing_config_returns_no_providers():
    result = accounts_needed_for_scopes(["sheets:read"], config=None)
    assert result.providers == ()
    assert result.unresolved_claims == ("sheets:read",)


def test_panel_renders_connect_affordance_for_unconnected_provider():
    result = accounts_needed_for_scopes(
        ["sheets:read", "sheets:write"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
    )
    html = _render_accounts_needed_panel(result, __import__("html").escape)
    assert "Accounts this connection needs" in html
    assert "Google" in html
    assert "not connected" in html
    assert "Connect Google" in html
    assert "hub://google/gmail?claims=sheets:read,sheets:write" in html


def test_panel_is_empty_when_no_provider_requirements():
    from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.oauth.account_requirements import (
        AccountRequirements,
    )

    html = _render_accounts_needed_panel(AccountRequirements(providers=(), unresolved_claims=("mail:read",)), __import__("html").escape)
    assert html == ""
