from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.boundary_policy import (
    NamespaceBoundaryPolicy,
)


def _policy() -> NamespaceBoundaryPolicy:
    return NamespaceBoundaryPolicy.from_config(
        "slack",
        {
            "authority_id": "delegated_client",
            "tools": {
                "action": {
                    "operation": "object.action",
                    "label": "Slack action",
                    "grants": ["legacy:action"],
                    "operations": {
                        "object.action.post_message": {
                            "label": "Post message",
                            "grants": ["slack:post"],
                        }
                    },
                }
            },
        },
    )


def test_exact_bounded_action_policy_wins_over_parent_tool_policy() -> None:
    policy = _policy()

    assert policy.operation_configured(
        tool_name="action", operation="object.action.post_message"
    )
    assert policy.grants_for(
        tool_name="action", operation="object.action.post_message"
    ) == ("slack:post",)
    assert policy.label_for(
        tool_name="action", operation="object.action.post_message"
    ) == "Post message"


def test_unknown_bounded_action_is_closed_when_exact_catalog_exists() -> None:
    policy = _policy()

    assert not policy.operation_configured(
        tool_name="action", operation="object.action.upload_file"
    )


def test_legacy_parent_action_policy_remains_a_fallback() -> None:
    policy = NamespaceBoundaryPolicy.from_config(
        "mail",
        {
            "tools": {
                "action": {
                    "operation": "object.action",
                    "operations": {
                        "object.action": {"grants": ["mail:legacy"]}
                    },
                }
            }
        },
    )

    assert policy.operation_configured(
        tool_name="action", operation="object.action.send"
    )
    assert policy.grants_for(
        tool_name="action", operation="object.action.send"
    ) == ("mail:legacy",)


def test_parent_does_not_open_unknown_action_beside_exact_catalog() -> None:
    policy = NamespaceBoundaryPolicy.from_config(
        "mail",
        {
            "tools": {
                "action": {
                    "operation": "object.action",
                    "operations": {
                        "object.action": {"grants": ["mail:legacy"]},
                        "object.action.send": {"grants": ["mail:send"]},
                    },
                }
            }
        },
    )

    assert policy.operation_configured(
        tool_name="action", operation="object.action.send"
    )
    assert not policy.operation_configured(
        tool_name="action", operation="object.action.forward"
    )
