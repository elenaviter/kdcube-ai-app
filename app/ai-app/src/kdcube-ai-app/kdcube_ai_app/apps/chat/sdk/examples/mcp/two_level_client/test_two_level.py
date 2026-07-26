# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# Verifies the reference classifier against the real KDCube door envelope
# shapes, as literal dicts - no live door required.

from kdcube_ai_app.apps.chat.sdk.examples.mcp.two_level_client.two_level import (
    classify_tool_result,
)

# --------------------------------------------------------------------------
# The four envelope shapes, transcribed from the door's own builders:
#   consent_denial.agent_grant_consent_denial          (level 1)
#   consent_denial.connect_first_denial_for_identity   (level 2, connect_required)
#   preflight.connected_account_consent_payload wrapped
#     by ConnectedAccountCredential.error_envelope      (level 2, account_required)
#   a plain success payload                             (level 0, ok)
# --------------------------------------------------------------------------

LEVEL_1 = {
    "ok": False,
    "error": "delegated_consent_required",            # NOTE: a bare string at gate 1
    "message": "'productivity_slack_search' on 'productivity' requires additional delegated consent.",
    "namespace": "productivity",
    "tool": "productivity_slack_search",
    "operation": "search",
    "required_grants": ["slack:search"],
    "missing_grants": ["slack:search"],
    "available_grants": [],
    "code": "connections.consent_needed",             # top-level code for delegated-client callers
    "connection_hub_url": "https://hub.example/api/integrations/.../connections_settings?tab=delegated_by_kdcube",
    "next_step": "The user extends this agent's grant with the missing access in Connection Hub.",
    "consent": {
        "kind": "delegated_agent_grant",
        "reason": "delegated_consent_required",
        "agent_client_id": "kdcube-agent:demo:assistant",
        "resource": "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/productivity*",
        "claims": ["slack:search"],
        "namespace": "productivity",
        "connection_hub_url": "https://hub.example/api/integrations/.../connections_settings?tab=delegated_by_kdcube",
    },
}

LEVEL_2_CONNECT = {
    "ok": False,
    "error": {
        "code": "needs_connected_account_consent",
        "message": "Connect a slack account in Connection Hub to continue.",
    },
    "reason": "connect_required",                     # top-level reason on the connect-first path
    "retry_hint": True,
    "namespace": "productivity_slack_search",
    "tool": "productivity_slack_search",
    "operation": "search",
    "provider_id": "slack",
    "required_grants": ["slack:search"],
    "missing_grants": ["slack:search"],
    "consent": {
        "kind": "delegated_to_kdcube.connected_account",
        "reason": "connect_required",
        "retry_hint": True,
        "provider_id": "slack",
        "connector_app_id": "slack-demo",
        "claims": ["slack:search"],
        "candidates": [],
        "url": "https://hub.example/connect/slack",
        "action_label": "Connect Slack",
    },
    "next_step": "Connect a slack account in Connection Hub first; the guided plan hands off to granting THIS agent.",
    "connection_hub_url": "https://hub.example/connect/slack",
}

LEVEL_2_ACCOUNT = {
    "ok": False,
    "error": {
        "code": "needs_connected_account_consent",
        "message": "Several Slack accounts match - choose one.",
    },
    "consent_required": True,
    "consent": {
        "kind": "delegated_to_kdcube.connected_account",
        "reason": "account_required",                 # reason lives under consent on this path
        "retry_hint": True,
        "provider_id": "slack",
        "connector_app_id": "slack-demo",
        "claims": ["slack:search"],
        "account_id": "",
        "candidates": [
            {"account_id": "acc_acme", "label": "Acme Workspace", "workspace": "acme", "status": "connected"},
            {"account_id": "acc_personal", "label": "Personal", "workspace": "personal", "status": "connected"},
        ],
        "url": "https://hub.example/pick/slack",
        "action_label": "Choose account",
    },
}

SUCCESS = {
    "ok": True,
    "ret": {"results": [{"channel": "#planning", "text": "Q3 planning kickoff"}]},
}


def test_level_1_caller_grant():
    out = classify_tool_result(LEVEL_1)
    assert out.ok is False
    assert out.level == 1
    # `code` names the denial itself; the door ships it as the bare `error`
    # string at gate 1 (the generic top-level `connections.consent_needed`
    # code is secondary).
    assert out.code == "delegated_consent_required"
    assert out.reason == ""                     # reason is a level-2 concept
    assert out.retry_hint is False              # a grant miss is never fixed by a blind resend
    assert out.claims == ["slack:search"]
    assert out.connection_hub_url.startswith("https://hub.example/")
    assert "grant" in out.next_action.lower()
    assert "slack:search" in out.next_action


def test_level_2_connect_required():
    out = classify_tool_result(LEVEL_2_CONNECT)
    assert out.ok is False
    assert out.level == 2
    assert out.code == "needs_connected_account_consent"
    assert out.reason == "connect_required"
    assert out.retry_hint is True
    assert out.provider_id == "slack"
    assert out.resend_with_account_id is False
    assert out.connection_hub_url == "https://hub.example/connect/slack"
    assert "connect" in out.next_action.lower()
    assert "slack" in out.next_action.lower()


def test_level_2_account_required():
    out = classify_tool_result(LEVEL_2_ACCOUNT)
    assert out.ok is False
    assert out.level == 2
    assert out.reason == "account_required"
    assert out.retry_hint is True
    assert out.resend_with_account_id is True
    assert out.candidate_account_ids() == ["acc_acme", "acc_personal"]
    # The fix is to resend with an account_id, naming a real candidate.
    assert "account_id=acc_acme" in out.next_action


def test_success():
    out = classify_tool_result(SUCCESS)
    assert out.ok is True
    assert out.level == 0
    assert out.code == "ok"
    assert out.next_action == ""


def test_malformed_degrades_to_level_0():
    # A non-mapping never raises; it degrades to an unrecognised level-0 verdict.
    out = classify_tool_result("not a dict")  # type: ignore[arg-type]
    assert out.ok is False
    assert out.level == 0
    assert out.code == "unrecognized"
    assert out.next_action

    # A partial/unknown denial also degrades cleanly, keeping any message.
    partial = {"ok": False, "error": {"code": "some_other_error", "message": "boom"}}
    out2 = classify_tool_result(partial)
    assert out2.level == 0
    assert out2.ok is False
    assert "boom" in out2.next_action


def test_agent_grant_required_is_level_2():
    # An account is connected but THIS caller is unbound (default-closed): a
    # level-2 reason under consent, resolved on the grant card (not a resend).
    envelope = {
        "ok": False,
        "error": {"code": "needs_connected_account_consent", "message": "Approve this caller on the account."},
        "consent": {
            "kind": "delegated_to_kdcube.connected_account",
            "reason": "agent_grant_required",
            "retry_hint": True,
            "provider_id": "google",
            "claims": ["gmail:read"],
            "candidates": [{"account_id": "acc_mail", "email": "user@example.com"}],
            "url": "https://hub.example/grant-card",
            "action_label": "Grant on account",
        },
    }
    out = classify_tool_result(envelope)
    assert out.level == 2
    assert out.reason == "agent_grant_required"
    assert out.resend_with_account_id is False        # fixed on the grant card, not by resend
    assert "gmail:read" in out.next_action
