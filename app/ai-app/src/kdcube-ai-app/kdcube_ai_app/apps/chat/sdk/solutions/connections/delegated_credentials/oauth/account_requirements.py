# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Which connected provider accounts a requested OAuth scope actually needs.

The consent page must not be a dead end: when an external client (Claude Code)
requests scopes whose claims are backed by a provider the operator has not
connected — or has connected but not for these claims — the page has to SAY so
and offer to connect/upgrade in place, rather than silently omitting the
requirement (the operator would approve a grant that cannot resolve, and the
first tool call would fail at the Delegated-to gate).

A requested claim reaches a provider two ways:

* **Hard (AND)** — a provider-claim token in some provider's vocabulary
  (``sheets:read`` -> Google). Every provider a scope names this way is
  required; each becomes its own requirement row.
* **Any-of (OR)** — a provider-neutral "door" claim (``mail:read``) whose
  backing PROVIDER claim differs from the door token and which several providers
  can satisfy (Google gmail:read, iCloud email:read, …). Connecting **any one**
  option satisfies it; the operator must NOT be told to connect them all.

Door groups resolve to: satisfied (an option is already connected and holds its
claims); folded into a provider that is required anyway for a hard reason (one
connect covers both); or an any-of ``DoorClaimChoice`` rendered as "connect one
of". Claims backed by no provider at all are returned as ``unresolved`` — legible
in the capabilities list, never invented into a provider.
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
    """A provider the requested scope needs (hard requirement), with status."""

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
class DoorClaimOption:
    """One provider that can satisfy a door claim (an any-of alternative)."""

    provider_id: str
    provider_label: str
    connector_app_id: str
    claims: tuple[str, ...]
    connected: bool
    connect_url: str = ""


@dataclass(frozen=True)
class DoorClaimChoice:
    """A door claim (e.g. mail) with no backing account yet: any ONE option
    satisfies it. Rendered as a 'connect one of' block, so the operator is never
    told to connect every provider that could serve it."""

    label: str
    options: tuple[DoorClaimOption, ...]


@dataclass(frozen=True)
class AccountRequirements:
    providers: tuple[ProviderAccountRequirement, ...]
    unresolved_claims: tuple[str, ...]
    choices: tuple[DoorClaimChoice, ...] = ()

    @property
    def has_gap(self) -> bool:
        return bool(self.choices) or any(not p.fully_satisfied for p in self.providers)


def _norm_list(values: Iterable[Any]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _door_label(door_claim: str) -> str:
    family = str(door_claim or "").split(":", 1)[0].strip()
    return family[:1].upper() + family[1:] if family else "Account"


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
    door_claim_providers: Mapping[str, Sequence[tuple[str, Sequence[str]]]] | None = None,
) -> AccountRequirements:
    """Resolve requested scopes to the provider accounts they need.

    ``door_claim_providers`` maps a door claim to its any-of options
    ``[(provider_id, [provider_claims])]``. ``connect_url_builder(provider_id,
    connector_app_id, claims)`` returns the Connection Hub deep-link that
    pre-selects the provider, connector app, and least-privilege claims for a
    connect/upgrade. When omitted, rows/options carry no URL.
    """
    if config is None or not getattr(config, "providers", None):
        return AccountRequirements(providers=(), unresolved_claims=_norm_list(scopes))

    claim_index = _claim_to_providers(config)
    door_index = door_claim_providers or {}

    # Index the operator's connected accounts by provider.
    accounts_by_provider: dict[str, list[ConnectedAccountView]] = {}
    held_by_provider: dict[str, set[str]] = {}
    for account in connected_accounts or ():
        provider_id = str(account.get("provider_id") or "").strip()
        if not provider_id:
            continue
        view = ConnectedAccountView(
            account_id=str(account.get("account_id") or "").strip(),
            label=str(account.get("label") or account.get("account_id") or "").strip(),
            held_claims=_norm_list(account.get("claims") or ()),
        )
        accounts_by_provider.setdefault(provider_id, []).append(view)
        held_by_provider.setdefault(provider_id, set()).update(view.held_claims)

    # Split requested scopes into HARD provider-claim requirements (every named
    # provider is required) and door groups (any ONE option satisfies). Door
    # claims that offer the same set of providers (mail:read + mail:send) are
    # coalesced into one group so the operator sees one "Mail" choice.
    hard_needed: dict[str, list[str]] = {}
    door_groups: "dict[tuple[str, ...], dict]" = {}
    unresolved: list[str] = []

    def _add_hard(provider_id: str, provider_claim: str) -> None:
        bucket = hard_needed.setdefault(provider_id, [])
        if provider_claim and provider_claim not in bucket:
            bucket.append(provider_claim)

    for scope in scopes or ():
        token = str(scope or "").strip()
        if not token:
            continue
        owners = claim_index.get(token)
        if owners:
            for provider_id in owners:
                _add_hard(provider_id, token)
            continue
        raw_options = door_index.get(token)
        if raw_options:
            options: list[tuple[str, tuple[str, ...]]] = []
            for provider_id, provider_claims in raw_options:
                pid = str(provider_id or "").strip()
                if not pid or config.provider(pid) is None:
                    continue
                claims = tuple(str(c or "").strip() for c in (provider_claims or ()) if str(c or "").strip())
                options.append((pid, claims))
            if not options:
                continue
            key = tuple(pid for pid, _ in options)
            group = door_groups.get(key)
            if group is None:
                group = {"label": _door_label(token), "claims": {pid: [] for pid, _ in options}}
                door_groups[key] = group
            for pid, claims in options:
                for claim in claims:
                    if claim not in group["claims"][pid]:
                        group["claims"][pid].append(claim)
            continue
        if token not in unresolved:
            unresolved.append(token)

    # Resolve each door group: satisfied by a connected account, folded into a
    # provider already required for a hard reason (so one connect covers both),
    # else an any-of choice over the options (connect any one).
    choices: list[DoorClaimChoice] = []
    for key, group in door_groups.items():
        alternatives = [(pid, tuple(group["claims"][pid])) for pid in key]
        if any(
            accounts_by_provider.get(pid) and set(claims).issubset(held_by_provider.get(pid, set()))
            for pid, claims in alternatives
        ):
            continue  # already satisfied by a connected account
        folded = False
        for pid, claims in alternatives:
            if pid in hard_needed:
                for claim in claims:
                    _add_hard(pid, claim)
                folded = True
                break
        if folded:
            continue  # one connect for the hard provider covers this door claim too
        options_out: list[DoorClaimOption] = []
        for pid, claims in alternatives:
            provider = config.provider(pid)
            app = _pick_connector_app(provider, claims)
            connector_app_id = app.connector_app_id if app is not None else ""
            held = held_by_provider.get(pid, set())
            missing = tuple(c for c in claims if c not in held)
            connect_url = ""
            if connect_url_builder is not None:
                connect_url = connect_url_builder(pid, connector_app_id, missing or claims) or ""
            options_out.append(
                DoorClaimOption(
                    provider_id=pid,
                    provider_label=str(provider.label or pid),
                    connector_app_id=connector_app_id,
                    claims=tuple(claims),
                    connected=bool(accounts_by_provider.get(pid)),
                    connect_url=connect_url,
                )
            )
        choices.append(DoorClaimChoice(label=str(group["label"]), options=tuple(options_out)))

    # Build the HARD provider rows (each required), including folded door claims.
    rows: list[ProviderAccountRequirement] = []
    for provider_id, needed in hard_needed.items():
        provider = config.provider(provider_id)
        if provider is None:
            continue
        needed_claims = _norm_list(needed)
        accounts = tuple(accounts_by_provider.get(provider_id, ()))
        held = held_by_provider.get(provider_id, set())
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

    return AccountRequirements(
        providers=tuple(rows),
        unresolved_claims=tuple(unresolved),
        choices=tuple(choices),
    )


__all__ = [
    "AccountRequirements",
    "ConnectedAccountView",
    "DoorClaimChoice",
    "DoorClaimOption",
    "ProviderAccountRequirement",
    "accounts_needed_for_scopes",
]
