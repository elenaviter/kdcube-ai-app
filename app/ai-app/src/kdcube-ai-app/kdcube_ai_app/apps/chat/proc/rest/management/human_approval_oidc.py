# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Cognito and Google fresh-authentication adapters."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
from connection_hub.connection_edges import request_origin
from fastapi import Request
from kdcube_ai_app.apps.chat.proc.rest.management.config import (
    CognitoFreshAuthenticationProvider,
    HumanApprovalConfig,
)
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval import (
    FRESH_AUTHENTICATION,
    BrowserAdminSession,
    HumanApprovalChallenge,
    HumanApprovalContext,
    HumanApprovalError,
    HumanApprovalEvidence,
    HumanApprovalOutcome,
    HumanApprovalPhase,
    resolve_browser_admin_session,
)
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval_store import (
    HumanProofRecord,
    OidcChallengeRecord,
    RedisHumanApprovalStore,
)
from kdcube_ai_app.apps.chat.sdk.config import get_settings

GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
OIDC_CALLBACK_PATH = "/api/integrations/management/v1/human-approval/oidc/callback"
_CLOCK_SKEW_SECONDS = 30


def _origin(request: Request) -> str:
    origin = request_origin(request).rstrip("/")
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (
            parsed.scheme == "http"
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}
        )
    ):
        raise HumanApprovalError(
            "human_approval_public_origin_invalid",
            status_code=503,
        )
    return origin


def human_approval_config(request: Request) -> HumanApprovalConfig:
    override = getattr(request.app.state, "human_approval_config", None)
    if override is not None:
        override.validate()
        return override
    try:
        return HumanApprovalConfig.from_settings(get_settings())
    except (TypeError, ValueError):
        raise HumanApprovalError(
            "human_approval_configuration_invalid",
            status_code=503,
        ) from None


def human_approval_store(
    request: Request,
    *,
    config: HumanApprovalConfig | None = None,
) -> RedisHumanApprovalStore:
    override = getattr(request.app.state, "human_approval_store", None)
    if override is not None:
        return override
    redis = getattr(request.app.state, "redis_async", None)
    if redis is None:
        raise HumanApprovalError(
            "human_approval_store_unavailable",
            status_code=503,
        )
    settings = get_settings()
    tenant = str(getattr(settings, "TENANT", None) or "").strip()
    project = str(getattr(settings, "PROJECT", None) or "").strip()
    if not tenant or not project:
        raise HumanApprovalError(
            "human_approval_coordinates_unavailable",
            status_code=503,
        )
    selected = config or human_approval_config(request)
    return RedisHumanApprovalStore(
        redis,
        tenant=tenant,
        project=project,
        ttl_seconds=selected.challenge_ttl_seconds,
    )


def _pkce_challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )


def _audience(claims: Mapping[str, Any]) -> set[str]:
    value = claims.get("aud")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item or "").strip() for item in value if str(item or "").strip()}
    return set()


def _select_cognito_provider(
    config: HumanApprovalConfig,
    admin: BrowserAdminSession,
) -> CognitoFreshAuthenticationProvider:
    providers = list(config.cognito_providers)
    hint = dict(admin.identity_hint or {})
    issuer = str(hint.get("iss") or "").strip()
    client_ids = _audience(hint)
    hinted_client = str(hint.get("client_id") or "").strip()
    if hinted_client:
        client_ids.add(hinted_client)
    matches = [
        provider
        for provider in providers
        if (not issuer or secrets.compare_digest(provider.issuer, issuer))
        and (not client_ids or provider.app_client_id in client_ids)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(providers) == 1 and not issuer and not client_ids:
        return providers[0]
    raise HumanApprovalError(
        "human_approval_cognito_provider_unresolved",
        status_code=409,
    )


def _fresh_provider(
    config: HumanApprovalConfig,
    admin: BrowserAdminSession,
) -> tuple[str, str]:
    selected = config.fresh_authentication_provider
    if selected == "auto":
        hint = dict(admin.identity_hint or {})
        issuer = str(hint.get("iss") or "").strip()
        provider = str(hint.get("provider") or "").strip().lower()
        if (
            provider == "google"
            or issuer in {"accounts.google.com", "https://accounts.google.com"}
            or admin.subject.startswith("google:")
        ):
            selected = "google"
        elif issuer.startswith("https://cognito-idp.") or provider == "cognito":
            selected = "cognito"
        elif bool(config.cognito_providers) != bool(config.google.client_id):
            selected = "cognito" if config.cognito_providers else "google"
        else:
            raise HumanApprovalError(
                "human_approval_fresh_authentication_provider_ambiguous",
                status_code=409,
            )
    if (
        selected == "google"
        and config.google.client_id
        and not admin.subject.startswith("google:")
    ):
        raise HumanApprovalError(
            "human_approval_google_subject_unavailable",
            status_code=409,
        )
    if selected == "cognito":
        if not config.cognito_managed_login:
            raise HumanApprovalError(
                "human_approval_cognito_managed_login_required",
                status_code=409,
            )
        provider = _select_cognito_provider(config, admin)
        return "cognito", provider.alias
    if selected == "google" and config.google.client_id:
        return "google", "google"
    raise HumanApprovalError(
        "human_approval_fresh_authentication_unavailable",
        status_code=409,
    )


def _validate_fresh_claims(
    claims: Mapping[str, Any],
    *,
    challenge: OidcChallengeRecord,
    provider_subject: str,
    require_after_challenge: bool = False,
    clock: Any = time.time,
) -> int:
    nonce = str(claims.get("nonce") or "").strip()
    subject = str(claims.get("sub") or "").strip()
    if not secrets.compare_digest(nonce, challenge.nonce):
        raise HumanApprovalError(
            "human_approval_oidc_nonce_invalid",
            status_code=403,
        )
    if not secrets.compare_digest(subject, provider_subject):
        raise HumanApprovalError(
            "human_approval_oidc_subject_changed",
            status_code=403,
        )
    try:
        authenticated_at = int(claims.get("auth_time"))
    except (TypeError, ValueError):
        raise HumanApprovalError(
            "human_approval_auth_time_required",
            status_code=409,
        ) from None
    now = int(clock())
    if (
        authenticated_at <= 0
        or authenticated_at > now + _CLOCK_SKEW_SECONDS
        or now - authenticated_at > challenge.context.max_evidence_age_seconds
        or (
            require_after_challenge
            and authenticated_at < challenge.created_at - _CLOCK_SKEW_SECONDS
        )
    ):
        raise HumanApprovalError(
            "human_approval_authentication_not_fresh",
            status_code=409,
        )
    return authenticated_at


class OidcFreshAuthenticationVerifier:
    """Produce one-use fresh-authentication evidence from the active IdP."""

    async def evaluate(
        self,
        request: Request,
        *,
        context: HumanApprovalContext,
        phase: HumanApprovalPhase,
    ) -> HumanApprovalOutcome:
        admin = await resolve_browser_admin_session(request)
        config = human_approval_config(request)
        store = human_approval_store(request, config=config)
        try:
            proof = await store.proof(
                context=context,
                admin=admin,
                consume=phase == "commit",
            )
        except Exception:  # noqa: BLE001 - normalize storage boundary failures
            raise HumanApprovalError(
                "human_approval_store_unavailable",
                status_code=503,
            ) from None
        if proof is not None:
            return proof
        if phase == "commit":
            raise HumanApprovalError(
                "human_approval_restart_required",
                status_code=409,
            )

        provider, alias = _fresh_provider(config, admin)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64) if provider == "cognito" else ""
        redirect_uri = f"{_origin(request)}{OIDC_CALLBACK_PATH}"
        try:
            challenge = await store.create_oidc_challenge(
                provider=provider,
                provider_alias=alias,
                nonce=nonce,
                code_verifier=verifier,
                redirect_uri=redirect_uri,
                context=context,
                admin=admin,
            )
        except Exception:  # noqa: BLE001 - normalize storage boundary failures
            raise HumanApprovalError(
                "human_approval_store_unavailable",
                status_code=503,
            ) from None

        if provider == "cognito":
            selected = next(
                item for item in config.cognito_providers if item.alias == alias
            )
            authorization_url = (
                f"{selected.hosted_ui_domain}/oauth2/authorize?"
                + urlencode(
                    {
                        "client_id": selected.app_client_id,
                        "response_type": "code",
                        "scope": "openid",
                        "redirect_uri": redirect_uri,
                        "state": challenge.state,
                        "nonce": nonce,
                        "prompt": "login",
                        "code_challenge": _pkce_challenge(verifier),
                        "code_challenge_method": "S256",
                    }
                )
            )
            method = "cognito_managed_login"
        else:
            claims = _json_claim_request()
            authorization_url = (
                GOOGLE_AUTHORIZATION_ENDPOINT
                + "?"
                + urlencode(
                    {
                        "client_id": config.google.client_id,
                        "response_type": "id_token",
                        "response_mode": "form_post",
                        "scope": "openid email",
                        "redirect_uri": redirect_uri,
                        "state": challenge.state,
                        "nonce": nonce,
                        "prompt": "select_account",
                        "claims": claims,
                    }
                )
            )
            method = "google_oidc_auth_time"
        return HumanApprovalChallenge(
            authorization_url=authorization_url,
            method=method,
        )


def _json_claim_request() -> str:
    return json.dumps(
        {"id_token": {"auth_time": {"essential": True}}},
        sort_keys=True,
        separators=(",", ":"),
    )


async def _exchange_cognito_code(
    request: Request,
    *,
    provider: CognitoFreshAuthenticationProvider,
    challenge: OidcChallengeRecord,
    code: str,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    override = getattr(request.app.state, "human_approval_cognito_exchange", None)
    if override is not None:
        return await override(
            provider=provider,
            challenge=challenge,
            code=code,
        )
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=False,
        ) as client:
            response = await client.post(
                f"{provider.hosted_ui_domain}/oauth2/token",
                headers={"Accept": "application/json"},
                data={
                    "grant_type": "authorization_code",
                    "client_id": provider.app_client_id,
                    "code": code,
                    "redirect_uri": challenge.redirect_uri,
                    "code_verifier": challenge.code_verifier,
                },
            )
            response.raise_for_status()
            payload = response.json()
    except Exception:  # noqa: BLE001 - conceal provider transport details
        raise HumanApprovalError(
            "human_approval_cognito_exchange_failed",
            status_code=403,
        ) from None
    if not isinstance(payload, Mapping):
        raise HumanApprovalError(
            "human_approval_cognito_exchange_failed",
            status_code=403,
        )
    return payload


async def _verify_cognito_token(
    request: Request,
    *,
    provider: CognitoFreshAuthenticationProvider,
    token: str,
) -> Mapping[str, Any]:
    override = getattr(request.app.state, "human_approval_cognito_token_verifier", None)
    if override is not None:
        return await override(provider=provider, token=token)
    from kdcube_ai_app.auth.implementations.cognito import CognitoAuthManager

    try:
        manager = CognitoAuthManager.from_values(
            region=provider.region,
            pool_id=provider.user_pool_id,
            client_id=provider.app_client_id,
            hosted_ui=provider.hosted_ui_domain,
            provider_alias=provider.alias,
        )
        return await manager.verify_id_token(token)
    except Exception:  # noqa: BLE001 - conceal provider verifier details
        raise HumanApprovalError(
            "human_approval_cognito_token_invalid",
            status_code=403,
        ) from None


async def _verify_google_token(
    request: Request,
    *,
    token: str,
    config: HumanApprovalConfig,
) -> Mapping[str, Any]:
    override = getattr(request.app.state, "human_approval_google_token_verifier", None)
    if override is not None:
        return await override(token=token, config=config.google)
    from kdcube_ai_app.apps.chat.sdk.integrations.google import oidc as google_oidc

    try:
        return await asyncio.to_thread(
            google_oidc.verify_google_id_token,
            token,
            client_id=config.google.client_id,
            jwks_url=config.google.jwks_url,
        )
    except Exception:  # noqa: BLE001 - conceal provider verifier details
        raise HumanApprovalError(
            "human_approval_google_token_invalid",
            status_code=403,
        ) from None


async def complete_oidc_callback(
    request: Request,
    *,
    state: str,
    code: str = "",
    id_token: str = "",
    response_issuer: str = "",
) -> str:
    """Verify an IdP callback, discard its token, and retain bounded proof."""

    if not 32 <= len(state) <= 512:
        raise HumanApprovalError("human_approval_state_invalid", status_code=400)
    config = human_approval_config(request)
    store = human_approval_store(request, config=config)
    try:
        challenge = await store.oidc_challenge(state)
    except Exception:  # noqa: BLE001 - normalize storage boundary failures
        raise HumanApprovalError(
            "human_approval_store_unavailable",
            status_code=503,
        ) from None
    if challenge is None:
        raise HumanApprovalError(
            "human_approval_state_invalid",
            status_code=403,
        )

    if challenge.provider == "cognito":
        if not code or len(code) > 4096 or id_token:
            raise HumanApprovalError(
                "human_approval_oidc_response_invalid",
                status_code=400,
            )
        provider = next(
            (
                item
                for item in config.cognito_providers
                if item.alias == challenge.provider_alias
            ),
            None,
        )
        if provider is None:
            raise HumanApprovalError(
                "human_approval_cognito_provider_unresolved",
                status_code=409,
            )
        if response_issuer and not secrets.compare_digest(
            response_issuer,
            provider.issuer,
        ):
            raise HumanApprovalError(
                "human_approval_oidc_issuer_mismatch",
                status_code=403,
            )
        tokens = await _exchange_cognito_code(
            request,
            provider=provider,
            challenge=challenge,
            code=code,
            timeout_seconds=config.http_timeout_seconds,
        )
        token = str(tokens.get("id_token") or "").strip()
        if not token or len(token) > 65536:
            raise HumanApprovalError(
                "human_approval_cognito_token_invalid",
                status_code=403,
            )
        claims = await _verify_cognito_token(
            request,
            provider=provider,
            token=token,
        )
        provider_subject = challenge.admin.subject
        method = "cognito_managed_login"
    elif challenge.provider == "google":
        if not id_token or len(id_token) > 65536 or code:
            raise HumanApprovalError(
                "human_approval_oidc_response_invalid",
                status_code=400,
            )
        if not secrets.compare_digest(
            response_issuer,
            "https://accounts.google.com",
        ):
            raise HumanApprovalError(
                "human_approval_oidc_issuer_mismatch",
                status_code=403,
            )
        claims = await _verify_google_token(
            request,
            token=id_token,
            config=config,
        )
        expected = challenge.admin.subject
        if not expected.startswith("google:"):
            raise HumanApprovalError(
                "human_approval_google_subject_unavailable",
                status_code=409,
            )
        provider_subject = expected.removeprefix("google:")
        method = "google_oidc_auth_time"
    else:
        raise HumanApprovalError(
            "human_approval_oidc_response_invalid",
            status_code=400,
        )

    verified_at = _validate_fresh_claims(
        claims,
        challenge=challenge,
        provider_subject=provider_subject,
        require_after_challenge=challenge.provider == "cognito",
    )
    try:
        consumed = await store.consume_oidc_challenge(challenge)
        if not consumed:
            raise HumanApprovalError(
                "human_approval_state_consumed",
                status_code=409,
            )
        created = await store.put_proof(
            HumanProofRecord(
                evidence=HumanApprovalEvidence(
                    subject=challenge.admin.subject,
                    assurance=FRESH_AUTHENTICATION,
                    method=method,
                    request_digest=challenge.context.request_digest,
                    verified_at=verified_at,
                ),
                context=challenge.context,
                admin=challenge.admin,
            )
        )
    except HumanApprovalError:
        raise
    except Exception:  # noqa: BLE001 - normalize storage boundary failures
        raise HumanApprovalError(
            "human_approval_store_unavailable",
            status_code=503,
        ) from None
    if not created:
        raise HumanApprovalError(
            "human_approval_state_consumed",
            status_code=409,
        )
    return challenge.context.return_url


__all__ = [
    "GOOGLE_AUTHORIZATION_ENDPOINT",
    "OIDC_CALLBACK_PATH",
    "OidcFreshAuthenticationVerifier",
    "complete_oidc_callback",
    "human_approval_config",
    "human_approval_store",
]
