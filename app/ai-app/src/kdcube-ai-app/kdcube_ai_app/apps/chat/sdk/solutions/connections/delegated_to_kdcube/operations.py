# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Connection Hub operations for delegated to KDCube.

This module contains the application-independent operation layer. Bundle
entrypoints should expose thin routes over these methods instead of reading
Connection Hub user props/secrets directly.
"""

from __future__ import annotations

import logging

import inspect
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.adapters import resolve_adapter
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.broker import DelegatedToKdcubeBroker
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.models import (
    CONNECTION_HUB_BUNDLE_ID,
    ConnectedAccount,
    IntegrationProvider,
    DelegatedToKdcubeConfig,
    as_dict,
    as_str,
    as_str_list,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.oauth import (
    OAuthStateStore,
    consume_oauth_state,
    create_oauth_state,
    state_digest,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.store import (
    DelegatedToKdcubeStore,
    credential_id_for,
)


SECRET_INPUT_KEYS = {
    "access_token",
    "refresh_token",
    "id_token",
    "app_password",
    "password",
    "api_key",
    "token",
    "secret",
}


def _clean_claims(provider: IntegrationProvider, connector_app_id: str, raw: Any) -> tuple[str, ...]:
    # Fail CLOSED: an empty selection stays empty (the caller rejects it), never
    # defaults to the connector app's whole ceiling. A connect must request only
    # the claims the user actually ticked - a dropped/empty selection must not
    # silently request read+send when the user chose read.
    requested = set(as_str_list(raw))
    known = set(provider.claims)
    selected = tuple(sorted(item for item in requested if item in known))
    unknown = sorted(item for item in requested if item not in known)
    if unknown:
        raise ValueError(f"unknown provider claim: {', '.join(unknown)}")
    app = provider.connector_apps.get(as_str(connector_app_id))
    if app and app.allowed_claims:
        outside = sorted(item for item in selected if item not in set(app.allowed_claims))
        if outside:
            raise ValueError(f"claim is outside connector app ceiling: {', '.join(outside)}")
    return selected


def _credential_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = as_dict(payload.get("credential"))
    for key in SECRET_INPUT_KEYS:
        value = payload.get(key)
        if as_str(value):
            raw[key] = value
    return {str(key): value for key, value in raw.items() if value not in (None, "")}


def _build_url(base_url: str, params: Mapping[str, Any]) -> str:
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{urlencode({k: v for k, v in params.items() if v not in (None, '')})}"


async def _resolve_client_secret(
    resolver: Callable[..., Any],
    *,
    provider_id: str,
    connector_app_id: str,
    connector_app: Any,
) -> str:
    value = resolver(provider_id=provider_id, connector_app_id=connector_app_id, connector_app=connector_app)
    if inspect.isawaitable(value):
        value = await value
    return as_str(value)


LOGGER = logging.getLogger("kdcube.connections.delegated_to_kdcube")


class DelegatedToKdcubeOperations:
    """Operation facade used by Connection Hub routes and tests."""

    def __init__(
        self,
        *,
        config: DelegatedToKdcubeConfig,
        store: DelegatedToKdcubeStore,
        consent_granted_notifier: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        # Called (best-effort) after a consent persists, with the granted
        # facts — the hub binds it to author `connections.consent.granted`
        # conversation events for pending demands (see consent_demand.py).
        self.consent_granted_notifier = consent_granted_notifier

    async def _notify_consent_granted(
        self,
        *,
        provider_id: str,
        connector_app_id: str,
        claims: tuple[str, ...],
        account_id: str,
    ) -> None:
        if self.consent_granted_notifier is None:
            return
        try:
            LOGGER.info(
                "[delegated.ops] consent notifier start user=%s provider=%s connector=%s account=%s claims=%s",
                self.store.user_id,
                provider_id,
                connector_app_id,
                account_id,
                ",".join(claims),
            )
            result = self.consent_granted_notifier(
                provider_id=provider_id,
                connector_app_id=connector_app_id,
                claims=list(claims),
                account_id=account_id,
            )
            if hasattr(result, "__await__"):
                await result
            LOGGER.info(
                "[delegated.ops] consent notifier done user=%s provider=%s connector=%s account=%s",
                self.store.user_id,
                provider_id,
                connector_app_id,
                account_id,
            )
        except Exception:
            LOGGER.warning("[delegated.consent] granted notifier failed (best-effort)", exc_info=True)

    async def catalog(self, *, provider_id: str = "") -> dict[str, Any]:
        accounts = await self.store.list_accounts(provider_id=provider_id)
        LOGGER.info(
            "[delegated.ops] catalog user=%s provider=%s enabled=%s accounts=%s",
            self.store.user_id,
            provider_id or "*",
            self.config.enabled,
            len(accounts),
        )
        return {
            "ok": True,
            "enabled": self.config.enabled,
            "providers": self.config.to_dict(include_client_ids=True)["providers"],
            "accounts": [await self._public_account(account) for account in accounts],
        }

    async def _public_account(self, account: ConnectedAccount) -> dict[str, Any]:
        data = account.public_dict()
        credential = await self.store.get_credential(account.credential_id)
        health = self._credential_health(account, credential)
        data.update(health)
        # A persisted live rejection outranks timestamp-derived health: a
        # provider may refuse a token whose stored expiry still looks valid.
        metadata = dict(account.metadata or {})
        persisted = as_str(metadata.get("credential_status"))
        if persisted in {"reconnect_required", "missing", "revoked"}:
            data["credential_status"] = persisted
            data["reconnect_required"] = persisted != "revoked"
            data["credential_message"] = (
                "The provider rejected the stored credential. Reconnect this account."
                if persisted == "reconnect_required"
                else data.get("credential_message") or ""
            )
        for key in ("credential_status_at", "last_error", "last_error_at"):
            value = metadata.get(key)
            if value:
                data[key] = value
        return data

    def _credential_health(self, account: ConnectedAccount, credential: dict[str, Any]) -> dict[str, Any]:
        if not credential:
            return {
                "credential_status": "missing",
                "credential_kind": "missing",
                "credential_refreshable": False,
                "credential_expires_at": 0,
                "reconnect_required": True,
                "credential_message": "Credential is missing. Reconnect this account.",
            }
        provider = self.config.provider(account.provider_id)
        adapter = None
        if provider and provider.adapter:
            try:
                resolved = resolve_adapter(provider.adapter)
                connector_app = provider.connector_apps.get(account.connector_app_id)
                adapter = resolved.bind(provider=provider, connector_app=connector_app)
            except Exception:
                adapter = None
        oauth = bool(credential.get("oauth") or credential.get("access_token"))
        expires_at = 0
        refreshable = False
        if adapter is not None:
            expires_at = adapter.credential_expires_at(credential)
            refreshable = adapter.credential_refreshable(credential)
        else:
            try:
                expires_at = int(credential.get("expires_at") or 0)
            except Exception:
                expires_at = 0
            refreshable = bool(str(credential.get("refresh_token") or "").strip())
        if not oauth:
            return {
                "credential_status": "active",
                "credential_kind": "static_secret",
                "credential_refreshable": False,
                "credential_expires_at": 0,
                "reconnect_required": False,
                "credential_message": "Credential is stored.",
            }
        if not expires_at:
            if refreshable:
                return {
                    "credential_status": "refreshable",
                    "credential_kind": "oauth",
                    "credential_refreshable": True,
                    "credential_expires_at": 0,
                    "reconnect_required": False,
                    "credential_message": "OAuth credential has a refresh token; KDCube will refresh it on next use.",
                }
            return {
                "credential_status": "active",
                "credential_kind": "oauth",
                "credential_refreshable": refreshable,
                "credential_expires_at": 0,
                "reconnect_required": False,
                "credential_message": "OAuth credential is stored.",
            }
        now = int(time.time())
        if expires_at <= now and not refreshable:
            return {
                "credential_status": "reconnect_required",
                "credential_kind": "oauth",
                "credential_refreshable": False,
                "credential_expires_at": expires_at,
                "reconnect_required": True,
                "credential_message": "OAuth access expired and no refresh token is stored. Reconnect this account.",
            }
        if expires_at <= now:
            return {
                "credential_status": "refreshable",
                "credential_kind": "oauth",
                "credential_refreshable": True,
                "credential_expires_at": expires_at,
                "reconnect_required": False,
                "credential_message": "OAuth access expired; KDCube will refresh it on next use.",
            }
        if expires_at <= now + 300:
            return {
                "credential_status": "expires_soon",
                "credential_kind": "oauth",
                "credential_refreshable": refreshable,
                "credential_expires_at": expires_at,
                "reconnect_required": False,
                "credential_message": "OAuth access expires soon.",
            }
        return {
            "credential_status": "active",
            "credential_kind": "oauth",
            "credential_refreshable": refreshable,
            "credential_expires_at": expires_at,
            "reconnect_required": False,
            "credential_message": "OAuth credential is active.",
        }

    async def connect_credential(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = as_str(payload.get("provider_id"))
        connector_app_id = as_str(payload.get("connector_app_id"))
        provider = self.config.provider(provider_id)
        LOGGER.info(
            "[delegated.ops] connect credential start user=%s provider=%s connector=%s account=%s requested_claims=%s",
            self.store.user_id,
            provider_id,
            connector_app_id,
            as_str(payload.get("account_id")) or "<new>",
            len(as_str_list(payload.get("claims"))),
        )
        if not self.config.enabled:
            raise ValueError("delegated-to-KDCube connections are not enabled")
        if provider is None or not provider.enabled:
            raise ValueError(f"provider is not enabled: {provider_id or '<missing>'}")
        if not connector_app_id:
            raise ValueError("connector app is required")
        if connector_app_id not in provider.connector_apps:
            raise ValueError(f"connector app is not configured: {connector_app_id}")
        credential = _credential_from_payload(payload)
        claims = _clean_claims(provider, connector_app_id, payload.get("claims"))
        if not claims:
            raise ValueError("at least one provider claim is required")
        if not credential:
            raise ValueError("credential is required")
        LOGGER.info(
            "[delegated.ops] connect credential claims resolved user=%s provider=%s connector=%s claims=%s",
            self.store.user_id,
            provider_id,
            connector_app_id,
            ",".join(claims),
        )

        account = ConnectedAccount(
            account_id=as_str(payload.get("account_id")),
            provider_id=provider_id,
            connector_app_id=connector_app_id,
            external_subject=as_str(payload.get("external_subject")),
            display_name=as_str(payload.get("display_name") or payload.get("label")),
            email=as_str(payload.get("email")),
            workspace=as_str(payload.get("workspace") or payload.get("team") or payload.get("tenant")),
            claims=claims,
            credential_id=as_str(payload.get("credential_id")),
            metadata=as_dict(payload.get("metadata")),
        )
        stored = await self.store.upsert_account(account)
        credential_id = stored.credential_id or credential_id_for(stored.account_id)
        credential_with_metadata = {
            "provider_id": provider_id,
            "connector_app_id": connector_app_id,
            "claims": list(claims),
            **credential,
        }
        await self.store.set_credential(credential_id, credential_with_metadata)
        LOGGER.info(
            "[delegated.ops] connect credential persisted user=%s provider=%s connector=%s account=%s credential=%s claims=%s",
            self.store.user_id,
            provider_id,
            connector_app_id,
            stored.account_id,
            credential_id,
            ",".join(stored.claims),
        )
        if credential_id != stored.credential_id:
            stored = await self.store.upsert_account(
                ConnectedAccount(
                    account_id=stored.account_id,
                    provider_id=stored.provider_id,
                    connector_app_id=stored.connector_app_id,
                    external_subject=stored.external_subject,
                    display_name=stored.display_name,
                    email=stored.email,
                    workspace=stored.workspace,
                    claims=stored.claims,
                    credential_id=credential_id,
                    status=stored.status,
                    connected_at=stored.connected_at,
                    metadata=stored.metadata,
                )
            )
        # The grant is an authored conversation event: notify (best-effort)
        # so pending demands in chat conversations learn the consent landed.
        await self._notify_consent_granted(
            provider_id=provider_id,
            connector_app_id=connector_app_id,
            claims=claims,
            account_id=stored.account_id,
        )
        return {"ok": True, "account": stored.public_dict()}

    async def start_oauth(
        self,
        payload: Mapping[str, Any],
        *,
        user_id: str,
        callback_url: str,
        state_store: OAuthStateStore,
        state_secret: str,
        ttl_seconds: int = 900,
        source: str = "connection_hub_widget",
    ) -> dict[str, Any]:
        provider_id = as_str(payload.get("provider_id"))
        connector_app_id = as_str(payload.get("connector_app_id"))
        provider = self.config.provider(provider_id)
        LOGGER.info(
            "[delegated.oauth] start request user=%s provider=%s connector=%s account=%s requested_claims=%s source=%s",
            user_id,
            provider_id,
            connector_app_id,
            as_str(payload.get("account_id")) or "<new>",
            len(as_str_list(payload.get("claims"))),
            source,
        )
        if not self.config.enabled:
            raise ValueError("delegated-to-KDCube connections are not enabled")
        if provider is None or not provider.enabled:
            raise ValueError(f"provider is not enabled: {provider_id or '<missing>'}")
        if not connector_app_id:
            raise ValueError("connector app is required")
        connector_app = provider.connector_apps.get(connector_app_id)
        if connector_app is None or not connector_app.enabled:
            raise ValueError(f"connector app is not enabled: {connector_app_id}")
        if not connector_app.client_id:
            raise ValueError(f"OAuth client id is not configured for connector app: {connector_app_id}")
        adapter = resolve_adapter(provider.adapter).bind(provider=provider, connector_app=connector_app)
        if not adapter.oauth_enabled:
            raise ValueError(f"provider adapter does not support OAuth: {provider.adapter}")
        claims = _clean_claims(provider, connector_app_id, payload.get("claims"))
        if not claims:
            raise ValueError("at least one provider claim is required")
        # Re-approving an existing account ADDS to its held claims by default:
        # the authorize request carries scopes for the union, and the callback
        # records the union — an incremental consent never drops access the
        # user already approved. claims_mode="replace" makes the requested set
        # authoritative (the hub's manage form: unticking removes a claim).
        claims_mode = as_str(payload.get("claims_mode") or "add").lower()
        account_id = as_str(payload.get("account_id"))
        if account_id and claims_mode != "replace":
            existing = await self.store.get_account(account_id)
            if (
                existing is not None
                and existing.provider_id == provider_id
                and (not existing.connector_app_id or existing.connector_app_id == connector_app_id)
            ):
                app_ceiling = set(connector_app.allowed_claims or ())
                held = tuple(
                    item
                    for item in (existing.claims or ())
                    if item in set(provider.claims) and (not app_ceiling or item in app_ceiling)
                )
                claims = tuple(sorted({*claims, *held}))
        LOGGER.info(
            "[delegated.oauth] start claims resolved user=%s provider=%s connector=%s claims=%s mode=%s",
            user_id,
            provider_id,
            connector_app_id,
            ",".join(claims),
            claims_mode,
        )
        redirect_uri = connector_app.redirect_uri or as_str(callback_url)
        if not redirect_uri:
            raise ValueError("OAuth callback URL is not available")
        scopes = adapter.provider_scopes_for_claims(list(claims), provider.claims)
        state = await create_oauth_state(
            state_store,
            secret=state_secret,
            user_id=user_id,
            provider_id=provider_id,
            connector_app_id=connector_app_id,
            claims=claims,
            account_id=account_id,
            claims_mode=claims_mode,
            return_hint=as_str(payload.get("return_hint") or payload.get("return_to")),
            source=source,
            ttl_seconds=ttl_seconds,
        )
        params = {
            "response_type": "code",
            "client_id": connector_app.client_id,
            "redirect_uri": redirect_uri,
            "state": state["state"],
        }
        scope_param = adapter.authorize_scope_param()
        if scopes:
            params[scope_param] = " ".join(scopes)
        for key, value in adapter.authorize_extra_params().items():
            params.setdefault(key, value)
        LOGGER.info(
            "[delegated.oauth] start ok user=%s provider=%s connector=%s state_id=%s scope_count=%s requested_scopes=[%s] redirect_uri=%s",
            user_id,
            provider_id,
            connector_app_id,
            state["state_id"],
            len(scopes),
            " ".join(scopes),
            redirect_uri,
        )
        return {
            "ok": True,
            "provider_id": provider_id,
            "connector_app_id": connector_app_id,
            "authorize_url": _build_url(adapter.authorize_url, params),
            "state_id": state["state_id"],
            "redirect_uri": redirect_uri,
            "claims": list(claims),
            "provider_scopes": scopes,
        }

    async def complete_oauth(
        self,
        *,
        code: str,
        state: str,
        callback_url: str,
        state_store: OAuthStateStore,
        state_secret: str,
        client_secret_resolver: Callable[..., Any],
    ) -> dict[str, Any]:
        LOGGER.info(
            "[delegated.oauth] callback begin user=%s state=%s code_present=%s",
            self.store.user_id,
            state_digest(state),
            bool(code),
        )
        payload = await consume_oauth_state(state_store, state=state, secret=state_secret)
        user_id = as_str(payload.get("user_id"))
        if user_id != self.store.user_id:
            raise ValueError("OAuth state user does not match integration store")
        provider_id = as_str(payload.get("provider_id"))
        connector_app_id = as_str(payload.get("connector_app_id"))
        account_id = as_str(payload.get("account_id"))
        claims_mode = as_str(payload.get("claims_mode") or "add").lower() or "add"
        LOGGER.info(
            "[delegated.oauth] state consumed user=%s provider=%s connector=%s account=%s claims=%s mode=%s state=%s",
            user_id,
            provider_id,
            connector_app_id,
            account_id or "<new>",
            ",".join(as_str_list(payload.get("claims"))),
            claims_mode,
            state_digest(state),
        )
        provider = self.config.provider(provider_id)
        if provider is None or not provider.enabled:
            raise ValueError(f"provider is not enabled: {provider_id or '<missing>'}")
        connector_app = provider.connector_apps.get(connector_app_id)
        if connector_app is None or not connector_app.enabled:
            raise ValueError(f"connector app is not enabled: {connector_app_id}")
        adapter = resolve_adapter(provider.adapter).bind(provider=provider, connector_app=connector_app)
        client_secret = await _resolve_client_secret(
            client_secret_resolver,
            provider_id=provider_id,
            connector_app_id=connector_app_id,
            connector_app=connector_app,
        )
        redirect_uri = connector_app.redirect_uri or as_str(callback_url)
        token = await adapter.exchange_code(
            code=code,
            redirect_uri=redirect_uri,
            client_id=connector_app.client_id,
            client_secret=client_secret,
        )
        access_token = as_str(token.get("access_token"))
        if not access_token:
            raise ValueError("OAuth token response did not include access_token")
        # Google returns the ACTUALLY-granted scopes in the token response's
        # `scope`. It can be narrower than requested (granular consent, or a
        # scope not registered on the OAuth consent screen), which is exactly
        # how "insufficient authentication scopes" happens at call time while
        # the account still shows the claim. Log requested-vs-granted verbatim.
        LOGGER.info(
            "[delegated.oauth] token exchanged user=%s provider=%s connector=%s has_access_token=%s has_refresh_token=%s expires_in=%s granted_scope=[%s] requested_claims=[%s] state=%s",
            user_id,
            provider_id,
            connector_app_id,
            bool(access_token),
            bool(token.get("refresh_token")),
            token.get("expires_in") or token.get("expires_at") or "",
            as_str(token.get("scope")),
            ",".join(as_str_list(payload.get("claims"))),
            state_digest(state),
        )
        profile = await adapter.fetch_profile(access_token=access_token, token=token)
        if not profile:
            profile = await adapter.normalize_profile(token)
        LOGGER.info(
            "[delegated.oauth] profile resolved user=%s provider=%s connector=%s external_subject=%s workspace=%s display=%s state=%s",
            user_id,
            provider_id,
            connector_app_id,
            bool(profile.get("external_subject") or profile.get("sub") or profile.get("id")),
            bool(profile.get("workspace")),
            bool(profile.get("display_name") or profile.get("name") or profile.get("email")),
            state_digest(state),
        )
        claims = tuple(as_str_list(payload.get("claims")))
        credential = {
            "oauth": True,
            **token,
        }
        connected = await self.connect_credential(
            {
                "provider_id": provider_id,
                "connector_app_id": connector_app_id,
                "external_subject": profile.get("external_subject") or profile.get("sub") or profile.get("id"),
                "email": profile.get("email"),
                "display_name": profile.get("display_name") or profile.get("name"),
                "workspace": profile.get("workspace"),
                "claims": claims,
                "account_id": account_id,
                "claims_mode": claims_mode,
                "credential": credential,
                "metadata": {
                    "oauth_source": payload.get("source"),
                    "oauth_state_id": state_digest(state),
                    "workspace_label": profile.get("workspace_label"),
                },
            }
        )
        stored_account = connected.get("account") if isinstance(connected.get("account"), dict) else {}
        LOGGER.info(
            "[delegated.oauth] consent persisted: account=%s provider=%s connector=%s claims=%s user=%s",
            stored_account.get("account_id"),
            provider_id,
            connector_app_id,
            ",".join(stored_account.get("claims") or []),
            user_id,
        )
        return {
            "ok": True,
            "provider_id": provider_id,
            "connector_app_id": connector_app_id,
            "user_id": user_id,
            "account": connected.get("account"),
            "return_hint": as_str(payload.get("return_hint")),
        }

    async def disconnect(self, *, account_id: str) -> dict[str, Any]:
        removed = await self.store.disconnect_account(account_id)
        return {"ok": removed, "removed": removed, "account_id": as_str(account_id)}

    async def resolve(
        self,
        *,
        provider_id: str = "",
        claim: str = "",
        connector_app_id: str = "",
        account_id: str = "",
    ) -> dict[str, Any]:
        broker = DelegatedToKdcubeBroker(config=self.config, store=self.store)
        result = await broker.ensure_claim(
            provider_id=provider_id,
            connector_app_id=connector_app_id,
            claim=claim,
            account_id=account_id,
        )
        return result.to_dict(include_credential=False)


def operations_for_user(
    *,
    user_id: str,
    config: DelegatedToKdcubeConfig,
    bundle_id: str = CONNECTION_HUB_BUNDLE_ID,
    store: DelegatedToKdcubeStore | None = None,
    consent_granted_notifier: Callable[..., Any] | None = None,
) -> DelegatedToKdcubeOperations:
    resolved_store = store or DelegatedToKdcubeStore(user_id=user_id, bundle_id=bundle_id)
    return DelegatedToKdcubeOperations(
        config=config,
        store=resolved_store,
        consent_granted_notifier=consent_granted_notifier,
    )


__all__ = ["DelegatedToKdcubeOperations", "operations_for_user"]
