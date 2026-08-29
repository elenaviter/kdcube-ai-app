# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The consent page must surface which connected accounts a requested scope
needs, and let the operator connect/upgrade in place — not dead-end when a
required provider is unconnected. These tests pin that resolution + render."""
from __future__ import annotations

from prokura.delegated_credentials.oauth.account_requirements import (
    accounts_needed_for_scopes,
)
from prokura.delegated_credentials.oauth.consent import (
    _render_accounts_needed_panel,
)
from prokura.delegated_to_kdcube.models import (
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
                "email": {
                    "label": "iCloud Mail",
                    "enabled": True,
                    "connector_apps": {
                        "app_password": {"enabled": True, "allowed_claims": ["email:read", "email:send"]}
                    },
                    "claims": {"email:read": {}, "email:send": {}},
                },
            }
        }
    )


# mail:read/mail:send are door claims that (soon) resolve to more than one
# provider; any ONE satisfies. Single-provider mail keeps [("google", ...)].
MAIL_DOORS = {
    "mail:read": [("google", ["gmail:read"]), ("email", ["email:read"])],
    "mail:send": [("google", ["gmail:send"]), ("email", ["email:send"])],
}


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


def test_door_claim_folds_into_a_hard_required_provider():
    # sheets:read hard-requires Google; mail:read (any-of Google/iCloud) is then
    # covered by that same Google connect, so it FOLDS into the Google row and
    # produces NO separate choice — one connect covers both.
    result = accounts_needed_for_scopes(
        ["mail:read", "sheets:read"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
        door_claim_providers=MAIL_DOORS,
    )
    assert [row.provider_id for row in result.providers] == ["google"]
    assert result.choices == ()  # folded, not offered as a choice
    google = result.providers[0]
    # hard claim first, folded door claim appended
    assert google.needed_claims == ("sheets:read", "gmail:read")
    assert result.unresolved_claims == ()
    assert google.connect_url == "hub://google/gmail?claims=sheets:read,gmail:read"


def test_multi_provider_door_claim_with_no_hard_need_is_a_single_choice():
    # No hard requirement forces a provider, and no mail account is connected:
    # the operator must connect ONE of Google/iCloud — a single choice with two
    # options, NOT two mandatory provider rows.
    result = accounts_needed_for_scopes(
        ["mail:read"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
        door_claim_providers=MAIL_DOORS,
    )
    assert result.providers == ()
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.label == "Mail"
    assert [o.provider_id for o in choice.options] == ["google", "email"]
    assert choice.options[0].connect_url == "hub://google/gmail?claims=gmail:read"
    assert choice.options[1].connect_url == "hub://email/app_password?claims=email:read"
    assert result.has_gap is True


def test_multi_provider_door_claim_satisfied_by_one_connected_option():
    # iCloud already connected and holding email:read satisfies mail:read; no
    # choice, no row, no gap — the operator is NOT nagged to also connect Google.
    result = accounts_needed_for_scopes(
        ["mail:read"],
        config=_config(),
        connected_accounts=[
            {"provider_id": "email", "account_id": "IC1", "label": "me@icloud.com", "claims": ["email:read"]},
        ],
        connect_url_builder=_connect_url,
        door_claim_providers=MAIL_DOORS,
    )
    assert result.providers == ()
    assert result.choices == ()
    assert result.has_gap is False


def test_same_provider_door_claims_coalesce_into_one_choice():
    # mail:read + mail:send offer the same providers; they must coalesce into a
    # single "Mail" choice carrying both provider claims, not two choices.
    result = accounts_needed_for_scopes(
        ["mail:read", "mail:send"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
        door_claim_providers=MAIL_DOORS,
    )
    assert len(result.choices) == 1
    google = result.choices[0].options[0]
    assert google.claims == ("gmail:read", "gmail:send")
    assert google.connect_url == "hub://google/gmail?claims=gmail:read,gmail:send"


def test_door_claim_provider_not_in_config_is_ignored():
    result = accounts_needed_for_scopes(
        ["mail:read"],
        config=_config(),
        connected_accounts=[],
        connect_url_builder=_connect_url,
        door_claim_providers={"mail:read": [("nonesuch", ["x:read"])]},
    )
    assert result.providers == ()
    assert result.choices == ()
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
    from prokura.delegated_credentials.oauth.account_requirements import (
        AccountRequirements,
    )

    html = _render_accounts_needed_panel(AccountRequirements(providers=(), unresolved_claims=("mail:read",)), __import__("html").escape)
    assert html == ""


def _need_row(*, needed, satisfied, missing, label="KDCube Demo Reader"):
    from prokura.delegated_credentials.oauth.account_requirements import (
        ConnectedAccountView, ProviderAccountRequirement,
    )
    return ProviderAccountRequirement(
        provider_id="google", provider_label="Google", connector_app_id="gmail",
        needed_claims=tuple(needed),
        accounts=(ConnectedAccountView(account_id="acc-1", label=label, held_claims=tuple(satisfied)),),
        satisfied_claims=tuple(satisfied), missing_claims=tuple(missing),
        connect_url="https://example.test/hub",
    )


def test_needs_row_does_not_reprint_the_claim_list_it_just_showed():
    """A connected account covering NONE of the requirement made the row state
    the same gap twice - "needs X - connected as Y, still needs X". The row
    already names X; the detail only adds that an account exists."""
    from prokura.delegated_credentials.oauth.consent import (
        _render_account_need_row,
    )
    html = _render_account_need_row(
        _need_row(needed=["sheets:read", "sheets:write"], satisfied=[],
                  missing=["sheets:read", "sheets:write"]),
        lambda s: s,
    )
    assert "still needs" not in html
    assert "connected as KDCube Demo Reader" in html
    assert html.count("sheets:write") == 1        # named once, not twice
    assert "Approve access for Google" in html    # the fix is still actionable


def test_needs_row_keeps_still_needs_when_the_account_covers_part_of_it():
    """A PARTIAL gap is what "still needs" is for - there the shorter list is
    new information, so it must survive."""
    from prokura.delegated_credentials.oauth.consent import (
        _render_account_need_row,
    )
    html = _render_account_need_row(
        _need_row(needed=["sheets:read", "sheets:write"], satisfied=["sheets:read"],
                  missing=["sheets:write"]),
        lambda s: s,
    )
    assert "still needs" in html
    assert html.count("sheets:read") == 1         # only in the needs list
