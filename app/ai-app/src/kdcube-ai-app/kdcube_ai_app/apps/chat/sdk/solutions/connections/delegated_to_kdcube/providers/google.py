# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Google/Gmail adapter registration for delegated to KDCube."""

from __future__ import annotations

import httpx

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.adapters import (
    DelegatedToKdcubeAdapter,
    adapter,
)

GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


@adapter("google.oauth")
class GoogleOAuthAdapter(DelegatedToKdcubeAdapter):
    label = "Google"
    kind = "oauth2"
    authorize_url = "https://accounts.google.com/o/oauth2/v2/auth"
    token_url = "https://oauth2.googleapis.com/token"
    oauth_default_scopes = ("openid", "email", "profile")

    def authorize_extra_params(self) -> dict:
        # No include_granted_scopes: KDCube's own claim selection (and the
        # add/replace union on re-connect) is the single source of truth for
        # the requested scopes. Google's incremental carry-forward would grant
        # scopes beyond what the user ticked here, so the consent screen and
        # the issued token must reflect exactly this request.
        return {
            "access_type": "offline",
            "prompt": "consent",
        }

    async def fetch_profile(self, *, access_token: str, token: dict | None = None) -> dict:
        del token
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Google userinfo request failed: {exc}") from exc
        try:
            data = response.json()
        except Exception:
            data = {}
        if not isinstance(data, dict) or response.status_code >= 400:
            detail = ""
            if isinstance(data, dict):
                detail = str(data.get("error_description") or data.get("error") or "")
            raise RuntimeError(f"Google userinfo failed: {detail or 'unknown error'}")
        return {
            "external_subject": str(data.get("sub") or "").strip(),
            "email": str(data.get("email") or "").strip(),
            "display_name": str(data.get("name") or data.get("email") or "").strip(),
        }

    async def normalize_profile(self, credential: dict) -> dict:
        return {
            "external_subject": str(credential.get("sub") or credential.get("external_subject") or "").strip(),
            "email": str(credential.get("email") or "").strip(),
            "display_name": str(credential.get("name") or credential.get("email") or "").strip(),
        }


__all__ = ["GoogleOAuthAdapter"]
