# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The named-services door's consent denial carries the per-agent grant path.

Regression: a hosted agent's op on an ungranted namespace (mail via a bearer
holding only slack grants) returned a bare `delegated_consent_required` error —
no agent identity, no resource, no grant action — so the caller's chat surface
could not raise the scoped consent banner and the Connection Hub landing showed
no pending claims. The denial now carries the full consent block for
`kdcube-agent:*` callers; other client families keep the reconnect guidance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kdcube_ai_app.apps.chat.sdk.runtime.dynamic_module_loader import load_dynamic_module_for_path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def _bridge_module():
    _name, module = load_dynamic_module_for_path(
        BUNDLE_ROOT / "services" / "named_services" / "bridge.py"
    )
    return module


class _Policy:
    namespace = "mail"

    def tool_configured(self, tool_name):
        return True

    def operation_configured(self, *, tool_name, operation):
        return True

    def grants_for(self, *, tool_name, operation):
        return ["mail:read"]

    def authority_for(self, *, tool_name, operation):
        return ""


def _request(client_id: str):
    return SimpleNamespace(state=SimpleNamespace(delegated_credential={
        "credential": {"attrs": {"grants": ["named_services:use", "slack:read"]}},
        "grant_record": {
            "client_id": client_id,
            "grants": [],
            "resource_grants": {
                "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/named_services*": [
                    "named_services:use", "slack:read",
                ],
            },
        },
    }))


def _bridge(m, request):
    return m.NamedServicesMcpBridge(config={}, tenant="t", project="p", request=request)


async def test_agent_caller_denial_carries_the_consent_block():
    m = _bridge_module()
    client = "kdcube-agent:ported-langgraph-agents@2026-07-13:lg-react"
    bridge = _bridge(m, _request(client))

    denial = await bridge._authorize(_Policy(), "object.search", tool_name="search")

    assert denial["error"] == "delegated_consent_required"
    assert denial["missing_grants"] == ["mail:read"]
    assert denial["code"] == "connections.consent_needed"
    consent = denial["consent"]
    assert consent["kind"] == "delegated_agent_grant"
    assert consent["agent_client_id"] == client
    assert consent["resource"].endswith("named_services*")
    assert consent["claims"] == ["mail:read"]
    assert consent["tool_name"] == "mail"
    assert consent["grant"]["operation"] == "delegated_agent_grant_create"
    assert consent["grant"]["payload"]["claims"] == ["mail:read"]
    assert "Connection Hub" in denial["next_step"]


async def test_external_caller_gets_identity_block_and_reconnect_fallback():
    # An external delegated client (Claude Code) is part of the SAME universal
    # contract: its denial carries the consent block naming the client and the
    # missing claims. Without a configured public base URL there is no hub deep
    # link, so the reconnect guidance stays as the fallback next step.
    m = _bridge_module()
    bridge = _bridge(m, _request("claude"))

    denial = await bridge._authorize(_Policy(), "object.search", tool_name="search")

    assert denial["error"] == "delegated_consent_required"
    consent = denial["consent"]
    assert consent["agent_client_id"] == "claude"
    assert consent["claims"] == ["mail:read"]
    assert "grant" not in consent          # one-click grant is hosted-agent only
    assert "Reconnect" in denial["next_step"]


def _request_via_subject():
    """The LIVE projection: grant_record without client_id; the delegate
    subject on the credential is the only agent identity, and the credential
    attrs carry the granted resource."""
    return SimpleNamespace(state=SimpleNamespace(delegated_credential={
        "credential": {
            "sub": "integration:kdcube-agent:ported-langgraph-agents@2026-07-13:lg-react:user-1",
            "attrs": {
                "grants": ["named_services:use"],
                "resource": "*/kdcube-services@1-0/public/mcp/named_services*",
            },
        },
        "grant_record": {"grants": []},
    }))


async def test_agent_identity_falls_back_to_the_delegate_subject():
    # Regression (live 2026-07-19): the projected grant_record carried no
    # client_id, so the denial went out bare (block={}) and no banner rose.
    m = _bridge_module()
    bridge = _bridge(m, _request_via_subject())

    denial = await bridge._authorize(_Policy(), "object.search", tool_name="search")

    assert denial["missing_grants"] == ["mail:read"]
    consent = denial["consent"]
    assert consent["agent_client_id"] == "kdcube-agent:ported-langgraph-agents@2026-07-13:lg-react"
    assert consent["resource"] == "*/kdcube-services@1-0/public/mcp/named_services*"
    assert consent["grant"]["payload"]["claims"] == ["mail:read"]
