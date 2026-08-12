# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""LinkedIn member adapter registration for delegated to KDCube."""

from __future__ import annotations

import base64
import copy
import json
from typing import Any, Mapping

import httpx

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.adapters import (
    DelegatedToKdcubeAdapter,
    adapter,
)

LINKEDIN_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
LINKEDIN_ORGANIZATION_ACLS_URL = "https://api.linkedin.com/rest/organizationAcls"
LINKEDIN_ORGANIZATIONS_URL = "https://api.linkedin.com/rest/organizations"

# The organization lane's scopes (Community Management API app). A connect
# whose resolved scopes are ONLY these must not force the OIDC identity
# scopes: a CMA-only application has no Sign In product, and LinkedIn rejects
# scopes no enabled product grants.
LINKEDIN_ORG_SCOPES = frozenset({"r_organization_social", "w_organization_social"})

# Versioned /rest calls need a LinkedIn-Version header. Deployments override
# via the provider's adapter_config.api_version; this default tracks the
# integration's shipped default.
DEFAULT_LINKEDIN_API_VERSION = "202601"


def _decode_id_token_claims(id_token: str) -> dict[str, Any]:
    parts = str(id_token or "").split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        parsed = json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8"))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _identity_from_claims(claims: Mapping[str, Any]) -> dict[str, Any]:
    subject = str(claims.get("sub") or "").strip()
    email = str(claims.get("email") or "").strip()
    given = str(claims.get("given_name") or "").strip()
    family = str(claims.get("family_name") or "").strip()
    name = str(claims.get("name") or "").strip() or " ".join(part for part in (given, family) if part)
    return {
        "external_subject": subject,
        "email": email,
        "display_name": name or email or subject,
    }


@adapter("linkedin.oauth_member")
class LinkedInMemberAdapter(DelegatedToKdcubeAdapter):
    """LinkedIn member OAuth.

    ``external_subject`` is the OIDC ``sub``; authorship is
    ``urn:li:person:{external_subject}``.

    LinkedIn issues refresh tokens only to approved applications. Without one
    the base class reports the credential as non-refreshable.
    """

    label = "LinkedIn"
    kind = "oauth2"
    authorize_url = "https://www.linkedin.com/oauth/v2/authorization"
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    oauth_default_scopes = ("openid", "profile", "email")

    # Identity scopes every MEMBER LinkedIn connection needs, whatever was
    # ticked. Org-only connects (see provider_scopes_for_claims) skip them.
    REQUIRED_IDENTITY_SCOPES = ("openid", "profile")

    # Set by bind(); carries adapter_config (api_version) for the org-identity
    # /rest calls. The registered adapter is a singleton, so bind returns a
    # bound COPY rather than mutating shared state.
    provider: Any = None

    def bind(self, *, provider: Any = None, connector_app: Any = None) -> "LinkedInMemberAdapter":
        del connector_app
        if provider is None:
            return self
        bound = copy.copy(self)
        bound.provider = provider
        return bound

    def provider_scopes_for_claims(self, claims: list, claim_map: dict) -> list:
        # `sub` is delivered only at connect time, via id_token or userinfo.
        # Identity scopes are added to every MEMBER request regardless of
        # claims. An org-only selection (Community Management API connector)
        # must go out untouched: that app has no Sign In product, and LinkedIn
        # rejects scopes no enabled product grants.
        scopes = super().provider_scopes_for_claims(claims, claim_map)
        if scopes and set(scopes) <= LINKEDIN_ORG_SCOPES:
            return scopes
        missing = [scope for scope in self.REQUIRED_IDENTITY_SCOPES if scope not in scopes]
        return [*missing, *scopes]

    def _api_version(self) -> str:
        config = getattr(self, "provider", None)
        adapter_config = dict(getattr(config, "adapter_config", None) or {})
        return str(adapter_config.get("api_version") or "").strip() or DEFAULT_LINKEDIN_API_VERSION

    async def _fetch_org_identity(self, client: httpx.AsyncClient, access_token: str) -> dict[str, Any]:
        """Identity for a token with no OIDC scopes: the organization lane.

        organizationAcls names the member (roleAssignee) and the organizations
        they administer — the only identity read a CMA-only token can perform.
        The first APPROVED ADMINISTRATOR organization becomes the account's
        workspace; its localizedName (best effort) becomes the label.
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "LinkedIn-Version": self._api_version(),
            "X-Restli-Protocol-Version": "2.0.0",
        }
        response = await client.get(
            LINKEDIN_ORGANIZATION_ACLS_URL,
            params={"q": "roleAssignee", "role": "ADMINISTRATOR", "state": "APPROVED"},
            headers=headers,
        )
        if response.status_code >= 400:
            return {}
        try:
            body = response.json()
        except Exception:
            return {}
        elements = list((body or {}).get("elements") or []) if isinstance(body, Mapping) else []
        rows = [dict(item or {}) for item in elements]
        rows = [row for row in rows if str(row.get("organization") or "").strip()]
        if not rows:
            return {}
        first = sorted(rows, key=lambda row: str(row.get("organization")))[0]
        organization = str(first.get("organization") or "").strip()
        subject = str(first.get("roleAssignee") or "").strip()
        label = ""
        org_id = organization.rsplit(":", 1)[-1]
        try:
            org_response = await client.get(
                f"{LINKEDIN_ORGANIZATIONS_URL}/{org_id}",
                headers=headers,
            )
            if org_response.status_code < 400:
                org_body = org_response.json()
                if isinstance(org_body, Mapping):
                    label = str(org_body.get("localizedName") or "").strip()
        except Exception:
            label = ""
        return {
            "external_subject": subject or organization,
            "email": "",
            "display_name": label or organization,
            "workspace": organization,
            "workspace_label": label,
        }

    async def fetch_profile(self, *, access_token: str, token: dict | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    LINKEDIN_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                try:
                    data = response.json()
                except Exception:
                    data = {}
                if not isinstance(data, Mapping) or response.status_code >= 400:
                    detail = ""
                    if isinstance(data, Mapping):
                        detail = str(data.get("error_description") or data.get("message") or data.get("error") or "")
                    # A w_member_social-only token cannot read userinfo.
                    fallback = _identity_from_claims(_decode_id_token_claims(str((token or {}).get("id_token") or "")))
                    if fallback.get("external_subject"):
                        return fallback
                    # An org-lane token (no OIDC scopes at all) identifies
                    # through the organizations it administers.
                    org_identity = await self._fetch_org_identity(client, access_token)
                    if org_identity.get("external_subject"):
                        return org_identity
                    raise RuntimeError(f"LinkedIn userinfo failed: {detail or 'unknown error'}")
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LinkedIn userinfo request failed: {exc}") from exc
        return _identity_from_claims(data)

    async def normalize_profile(self, credential: dict) -> dict:
        # The token response carries no subject; it is in the id_token when
        # openid was granted.
        data = dict(credential or {})
        claims = {**_decode_id_token_claims(str(data.get("id_token") or "")), **data}
        return _identity_from_claims(claims)


__all__ = ["LinkedInMemberAdapter"]
