# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The one canonical credential reader covers BOTH envelope shapes.

This test is the single place the nested-field drift is guarded: an agent
bearer carries a resource_grants map in credential.attrs; an OAuth client
carries a single attrs.resource; both are read identically flat. When a new
delegated shape appears, extend this test, not five scattered readers."""

from __future__ import annotations

from types import SimpleNamespace

from connection_hub.delegated_credentials.credential_view import (
    DelegatedCredentialView,
    delegated_credential_view,
)

DOOR = "*/kdcube-services@1-0/public/mcp/named_services*"


def _req(delegated):
    return SimpleNamespace(state=SimpleNamespace(delegated_credential=delegated))


def test_agent_bearer_resource_from_credential_resource_grants():
    # The live agent shape: no single attrs.resource; the door resource lives
    # in credential.attrs.resource_grants. (The regression that broke the
    # lg-react mail banner read the wrong nesting and got resource=''.)
    view = delegated_credential_view(_req({
        "credential": {
            "sub": "integration:kdcube-agent:app@v1:main:user-1",
            "attrs": {
                "grants": ["named_services:use", "slack:read"],
                "resource_grants": {DOOR: ["named_services:use", "slack:read"]},
            },
        },
        "grant_record": {"grants": []},
    }))
    assert view.resource == DOOR
    assert view.resources == (DOOR.rstrip("/"),)
    assert view.grants == frozenset({"named_services:use", "slack:read"})
    assert view.client_id == "kdcube-agent:app@v1:main"
    assert view.is_agent and view.agent_client_id == "kdcube-agent:app@v1:main"
    assert view.grants_for_resource(
        "https://h/api/integrations/bundles/t/p/kdcube-services@1-0/public/mcp/named_services"
    ) == {"named_services:use", "slack:read"}


def test_oauth_client_single_resource_string():
    # An OAuth external client carries a single attrs.resource + scopes; the
    # view synthesizes resource_grants so downstream logic is shape-agnostic.
    view = delegated_credential_view(_req({
        "credential": {
            "attrs": {
                "client_id": "claude",
                "resource": "https://h/api/records",
                "scopes": ["records:read"],
            },
        },
        "grant_record": {"client_id": "claude", "registry_access_id": "oauth-abc"},
    }))
    assert view.resource == "https://h/api/records"
    assert view.grants == frozenset({"records:read"})
    assert view.client_id == "claude"
    assert not view.is_agent and view.agent_client_id == ""
    assert view.registry_access_id == "oauth-abc"


def test_grants_read_from_grant_record_embedded_credential():
    # Grant facts can hide in grant_record.credential.attrs — the door's auth
    # set must still include them.
    view = delegated_credential_view(_req({
        "credential": {"attrs": {}},
        "grant_record": {
            "client_id": "kdcube-agent:app@v1:main",
            "credential": {"attrs": {"scopes": ["memories:read"], "resource_grants": {DOOR: ["memories:read"]}}},
        },
    }))
    assert "memories:read" in view.grants
    assert view.resource == DOOR


def test_absent_credential_is_empty_not_present():
    view = delegated_credential_view(SimpleNamespace(state=SimpleNamespace()))
    assert not view.present and view.resource == "" and view.grants == frozenset()


def test_from_envelope_matches_from_request():
    # The guard reads an envelope directly; the bridge reads a request. Same
    # facts, same view.
    class _Env:
        def to_dict(self):
            # A real credential carries scopes AND resource_grants (both set by
            # build_delegated_client_credential); .grants reads the scope set.
            return {"attrs": {"scopes": ["slack:read"], "resource_grants": {DOOR: ["slack:read"]}}}
    view = DelegatedCredentialView.from_envelope(_Env())
    assert view.resource == DOOR
    assert view.grants == frozenset({"slack:read"})
    assert view.grants_for_resource("https://h/kdcube-services@1-0/public/mcp/named_services") == {"slack:read"}
