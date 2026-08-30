# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""LinkedIn delegated-to-KDCube adapter contract."""

from __future__ import annotations

import base64
import json

import pytest

from connection_hub.delegated_to_kdcube import providers  # noqa: F401
from connection_hub.delegated_to_kdcube.adapters import (
    resolve_adapter,
)

CLAIM_MAP = {
    "linkedin:profile": {"provider_scopes": ["openid", "profile", "email"]},
    "linkedin:post": {"provider_scopes": ["w_member_social"]},
}


def _id_token(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


@pytest.fixture()
def adapter():
    return resolve_adapter("linkedin.oauth_member")


def test_adapter_is_registered_with_oauth_endpoints(adapter):
    assert adapter.oauth_enabled
    assert adapter.authorize_url == "https://www.linkedin.com/oauth/v2/authorization"
    assert adapter.token_url == "https://www.linkedin.com/oauth/v2/accessToken"


def test_identity_scopes_are_requested_even_for_a_write_only_claim(adapter):
    scopes = adapter.provider_scopes_for_claims(["linkedin:post"], CLAIM_MAP)
    assert "openid" in scopes and "profile" in scopes
    assert "w_member_social" in scopes


def test_identity_scopes_are_not_duplicated(adapter):
    scopes = adapter.provider_scopes_for_claims(
        ["linkedin:profile", "linkedin:post"], CLAIM_MAP
    )
    assert scopes.count("openid") == 1
    assert scopes.count("profile") == 1


@pytest.mark.asyncio
async def test_subject_is_read_from_the_id_token(adapter):
    profile = await adapter.normalize_profile(
        {
            "access_token": "T",
            "id_token": _id_token(
                {
                    "sub": "dE5aOhH-ap",
                    "email": "jane@example.com",
                    "given_name": "Jane",
                    "family_name": "Smith",
                }
            ),
        }
    )
    assert profile["external_subject"] == "dE5aOhH-ap"
    assert profile["email"] == "jane@example.com"
    assert profile["display_name"] == "Jane Smith"


@pytest.mark.asyncio
async def test_missing_id_token_yields_an_empty_subject(adapter):
    profile = await adapter.normalize_profile({"access_token": "T"})
    assert profile["external_subject"] == ""


def test_credential_without_refresh_token_is_not_refreshable(adapter):
    assert adapter.credential_refreshable({"access_token": "T"}) is False
    assert adapter.credential_refreshable({"access_token": "T", "refresh_token": "R"}) is True


# --- Organization lane (Community Management API connector) ----------------

ORG_CLAIM_MAP = {
    **CLAIM_MAP,
    "linkedin:org:post": {"provider_scopes": ["w_organization_social"]},
    "linkedin:org:read": {"provider_scopes": ["r_organization_social"]},
}


def test_org_only_claims_request_no_identity_scopes(adapter):
    """A CMA-only app has no Sign In product: LinkedIn rejects openid/profile.
    An org-only selection must therefore go out with exactly the org scopes."""
    scopes = adapter.provider_scopes_for_claims(
        ["linkedin:org:post", "linkedin:org:read"], ORG_CLAIM_MAP
    )
    assert sorted(scopes) == ["r_organization_social", "w_organization_social"]


def test_mixed_claims_still_force_identity_scopes(adapter):
    scopes = adapter.provider_scopes_for_claims(
        ["linkedin:post", "linkedin:org:read"], ORG_CLAIM_MAP
    )
    assert "openid" in scopes and "profile" in scopes


class _Response:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _OrgHttpClient:
    """userinfo fails (no openid on the token); organizationAcls answers."""

    def __init__(self):
        self.requested: list[str] = []

    async def get(self, url, **kwargs):
        self.requested.append(url)
        if url.endswith("/userinfo"):
            return _Response(403, {"error": "insufficient_scope"})
        if "organizationAcls" in url:
            assert kwargs["headers"]["LinkedIn-Version"], "versioned endpoint needs a version header"
            return _Response(
                200,
                {
                    "elements": [
                        {
                            "organization": "urn:li:organization:5123456",
                            "role": "ADMINISTRATOR",
                            "state": "APPROVED",
                            "roleAssignee": "urn:li:person:dE5aOhH-ap",
                        }
                    ]
                },
            )
        if "/organizations/" in url:
            return _Response(200, {"localizedName": "KDCube"})
        return _Response(404, {})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_org_token_identity_resolves_through_organization_acls(adapter, monkeypatch):
    """No openid -> no userinfo, no id_token. The org lane identifies through
    the organizations the member administers: subject = the member URN,
    workspace = the organization URN (one connected account per organization)."""
    import httpx

    from connection_hub.delegated_to_kdcube.providers import (
        linkedin as adapter_module,
    )

    client = _OrgHttpClient()
    monkeypatch.setattr(adapter_module.httpx, "AsyncClient", lambda **kwargs: client)
    profile = await adapter.fetch_profile(access_token="ORG_TOKEN", token={"access_token": "ORG_TOKEN"})
    assert profile["external_subject"] == "urn:li:person:dE5aOhH-ap"
    assert profile["workspace"] == "urn:li:organization:5123456"
    assert profile["workspace_label"] == "KDCube"
    assert profile["display_name"] == "KDCube"


@pytest.mark.asyncio
async def test_member_token_identity_is_unchanged_by_the_org_fallback(adapter, monkeypatch):
    """userinfo answering means the member path resolves exactly as before —
    the org fallback never runs."""
    from connection_hub.delegated_to_kdcube.providers import (
        linkedin as adapter_module,
    )

    class _MemberClient(_OrgHttpClient):
        async def get(self, url, **kwargs):
            self.requested.append(url)
            if url.endswith("/userinfo"):
                return _Response(200, {"sub": "dE5aOhH-ap", "email": "jane@example.com", "name": "Jane Smith"})
            raise AssertionError(f"unexpected call: {url}")

    client = _MemberClient()
    monkeypatch.setattr(adapter_module.httpx, "AsyncClient", lambda **kwargs: client)
    profile = await adapter.fetch_profile(access_token="MEMBER_TOKEN")
    assert profile["external_subject"] == "dE5aOhH-ap"
    assert profile.get("workspace") is None or profile.get("workspace") == ""
