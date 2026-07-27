# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Which connected provider accounts a requested OAuth scope actually needs.

The consent page must not be a dead end: when an external client (Claude Code)
requests scopes whose claims are backed by a provider the operator has not
connected — or has connected but not for these claims — the page has to SAY so
and offer to connect/upgrade in place, rather than silently omitting the
requirement (the operator would approve a grant that cannot resolve, and the
first tool call would fail at the Delegated-to gate).

This module is a pure resolver over declared config: it maps each requested
claim to the provider whose claim vocabulary owns it, groups by provider, and
reports connected/missing status against the operator's real connected
accounts. Claims that are not provider-claim tokens (namespace/"door" claims
whose provider is chosen at first use) are returned separately as
``unresolved`` — they are legible in the capabilities list below and are not
invented into a provider here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube.models import (
    ConnectorApp,
    DelegatedToKdcubeConfig,
    IntegrationProvider,
)


@dataclass(frozen=True)
class ConnectedAccountView:
    """One connected account, from the consent handler's account list."""

    account_id: str
    label: str
    held_claims: tuple[str, ...]


@dataclass(frozen=True)
class ProviderAccountRequirement:
    """A provider the requested scope needs, with connect/upgrade status."""

    provider_id: str
    provider_label: str
    connector_app_id: str
    needed_claims: tuple[str, ...]
    accounts: tuple[ConnectedAccountView, ...]
    satisfied_claims: tuple[str, ...]
    missing_claims: tuple[str, ...]
    connect_url: str = ""

    @property
    def connected(self) -> bool:
        return bool(self.accounts)

    @property
    def fully_satisfied(self) -> bool:
        return self.connected and not self.missing_claims

    def status(self) -> str:
        """One of: not_connected | needs_access | connected."""
        if not self.connected:
            return "not_connected"
        if self.missing_claims:
            return "needs_access"
        return "connected"


@dataclass(frozen=True)
class AccountRequirements:
    providers: tuple[ProviderAccountRequirement, ...]
    unresolved_claims: tuple[str, ...]

    @property
    def has_gap(self) -> bool:
        return any(not p.fully_satisfied for p in self.providers)


def _norm_list(values: Iterable[Any]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _claim_to_providers(config: DelegatedToKdcubeConfig) -> dict[str, list[str]]:
    """Reverse index: provider-claim token -> provider ids that declare it."""
    index: dict[str, list[str]] = {}
    for provider_id, provider in (config.providers or {}).items():
        if not isinstance(provider, IntegrationProvider) or not provider.enabled:
            continue
        for claim in (provider.claims or {}):
            token = str(claim or "").strip()
            if not token:
                continue
            index.setdefault(token, [])
            if provider_id not in index[token]:
                index[token].append(str(provider_id))
    return index


def _pick_connector_app(provider: IntegrationProvider, needed: Sequence[str]) -> ConnectorApp | None:
    """The enabled connector app that can grant the needed claims.

    Broker-style: prefer an app whose ``allowed_claims`` covers ALL needed
    claims; else the one covering the most; else the first enabled app. This is
    the same "the app that holds the claims" selection the credential broker
    uses, so the connect deep-link targets an app that can actually grant them.
    """
    need = set(needed)
    best: ConnectorApp | None = None
    best_score = -1
    for app in (provider.connector_apps or {}).values():
        if not isinstance(app, ConnectorApp) or not app.enabled:
            continue
        allowed = set(app.allowed_claims or ())
        score = len(need & allowed)
        if need.issubset(allowed):
            return app
        if score > best_score:
            best = app
            best_score = score
    return best


def accounts_needed_for_scopes(
    scopes: Iterable[str],
    *,
    config: DelegatedToKdcubeConfig | None,
    connected_accounts: Iterable[Mapping[str, Any]] | None = None,
    connect_url_builder: Callable[[str, str, Sequence[str]], str] | None = None,
) -> AccountRequirements:
    """Resolve requested scopes to the provider accounts they need.

    ``connect_url_builder(provider_id, connector_app_id, claims)`` returns the
    Connection Hub deep-link that pre-selects the provider, connector app, and
    least-privilege claims for a connect/upgrade. When omitted, rows carry no
    URL (the page still renders the requirement, just without the button).
    """
    if config is None or not getattr(config, "providers", None):
        return AccountRequirements(providers=(), unresolved_claims=_norm_list(scopes))

    claim_index = _claim_to_providers(config)

    # Group requested claims by the provider that owns them, preserving order.
    needed_by_provider: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for scope in scopes or ():
        token = str(scope or "").strip()
        if not token:
            continue
        owners = claim_index.get(token)
        if not owners:
            if token not in unresolved:
                unresolved.append(token)
            continue
        for provider_id in owners:
            bucket = needed_by_provider.setdefault(provider_id, [])
            if token not in bucket:
                bucket.append(token)

    # Index the operator's connected accounts by provider.
    accounts_by_provider: dict[str, list[ConnectedAccountView]] = {}
    for account in connected_accounts or ():
        provider_id = str(account.get("provider_id") or "").strip()
        if not provider_id:
            continue
        accounts_by_provider.setdefault(provider_id, []).append(
            ConnectedAccountView(
                account_id=str(account.get("account_id") or "").strip(),
                label=str(account.get("label") or account.get("account_id") or "").strip(),
                held_claims=_norm_list(account.get("claims") or ()),
            )
        )

    rows: list[ProviderAccountRequirement] = []
    for provider_id, needed in needed_by_provider.items():
        provider = config.provider(provider_id)
        if provider is None:
            continue
        needed_claims = _norm_list(needed)
        accounts = tuple(accounts_by_provider.get(provider_id, ()))
        held: set[str] = set()
        for account in accounts:
            held.update(account.held_claims)
        satisfied = tuple(c for c in needed_claims if c in held)
        missing = tuple(c for c in needed_claims if c not in held)
        app = _pick_connector_app(provider, needed_claims)
        connector_app_id = app.connector_app_id if app is not None else ""
        connect_url = ""
        if connect_url_builder is not None and missing:
            connect_url = connect_url_builder(provider_id, connector_app_id, missing) or ""
        rows.append(
            ProviderAccountRequirement(
                provider_id=provider_id,
                provider_label=str(provider.label or provider_id),
                connector_app_id=connector_app_id,
                needed_claims=needed_claims,
                accounts=accounts,
                satisfied_claims=satisfied,
                missing_claims=missing,
                connect_url=connect_url,
            )
        )

    return AccountRequirements(providers=tuple(rows), unresolved_claims=tuple(unresolved))


__all__ = [
    "AccountRequirements",
    "ConnectedAccountView",
    "ProviderAccountRequirement",
    "accounts_needed_for_scopes",
]
