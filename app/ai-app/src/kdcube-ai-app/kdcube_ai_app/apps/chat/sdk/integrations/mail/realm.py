# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Mail as a REALM: the user's connected mail accounts across providers.

A platform user connects mail accounts of different providers (a Gmail
account, an iCloud mailbox) and Connection Hub binds each to a card with its
own claims. The tools that read or write mail used to ask the hub for Google
accounts only, so a fully connected, fully consented iCloud account was
invisible to them: one Google account matched, no ambiguity, silent pick.
This module is the realm view the mail tools resolve through instead:

- ``discover_mail_providers``: the realm's members for this deployment,
  read from the hub catalog by adapter family (Gmail via OAuth, every
  IMAP/SMTP instance), so no provider instance is ever named in code;
- ``list_mail_accounts``: every connected mail account the caller may use,
  across those members, filtered by the agent's per-account binding when the
  call runs on a delegated card;
- ``choose_mail_account``: the selection rule. An explicit ``account_id``
  wins; one eligible account is used; several eligible accounts come back as
  an ``account_required`` envelope with labeled candidates, so the agent asks
  the person instead of guessing a provider.

Provider transport stays in the provider packages (``google.gmail_tools``,
``email.icloud_tools``); this module only decides WHICH account a call is
about and which provider therefore answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connector_app_resolution import (
    resolve_connector_app_id,
)

LOGGER = logging.getLogger(__name__)

_SERVICE: Any = None
_INTEGRATIONS: dict[str, Any] = {}


def bind_service(svc: Any) -> None:
    global _SERVICE
    _SERVICE = svc


def bind_integrations(integrations: Mapping[str, Any] | None) -> None:
    global _INTEGRATIONS
    _INTEGRATIONS = dict(integrations or {})


def _clean(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class MailProviderSpec:
    """One provider INSTANCE's place in the realm, as discovered from the
    Connection Hub catalog: its hub identity, the claim each mail verb needs
    there, the transport that speaks to it, and the instance settings (hosts)
    that transport reads. Nothing in this module names a provider instance;
    the deployment's descriptor does.

    IMAP/SMTP instances have no separate compose claim; a draft is an APPEND
    into the Drafts mailbox, a write the send claim already authorizes."""

    key: str
    label: str
    provider_id: str
    connector_app_id: str
    transport: str  # "gmail" | "imap_smtp"
    read_claim: str
    send_claim: str
    draft_claim: str
    settings: dict[str, Any] = field(default_factory=dict)

    def claim_for(self, need: str) -> str:
        return {
            "read": self.read_claim,
            "send": self.send_claim,
            "draft": self.draft_claim,
        }[need]

    def requirement(self, need: str) -> dict[str, Any]:
        """The single-provider requirement the enforcement layer understands."""
        return {
            "provider_id": self.provider_id,
            "connector_app_id": self.connector_app_id,
            "claims": [self.claim_for(need)],
        }


GMAIL_ADAPTER_PREFIX = "google."
IMAP_SMTP_ADAPTER = "email.imap_smtp_app_password"
GMAIL_CLAIMS = {"read": "gmail:read", "send": "gmail:send", "draft": "gmail:compose"}
EMAIL_CLAIMS = {"read": "email:read", "send": "email:send", "draft": "email:send"}


def _spec_from_catalog(provider_id: str, entry: Mapping[str, Any]) -> MailProviderSpec | None:
    """A realm member from one catalog provider entry, or None when the
    provider is not a mail provider. Membership is decided by the ADAPTER
    family and the claims it publishes, never by the instance's name."""
    if not entry.get("enabled", True):
        return None
    adapter = _clean(entry.get("adapter"))
    claims = dict(entry.get("claims") or {})
    connector_apps = dict(entry.get("connector_apps") or {})
    connector_app_id = ""
    for app_id, app in connector_apps.items():
        if isinstance(app, Mapping) and app.get("enabled", True):
            connector_app_id = _clean(app_id)
            break
    label = _clean(entry.get("label")) or provider_id
    if adapter == IMAP_SMTP_ADAPTER and "email:read" in claims:
        return MailProviderSpec(
            key=provider_id,
            label=label,
            provider_id=provider_id,
            connector_app_id=connector_app_id or "app_password",
            transport="imap_smtp",
            read_claim=EMAIL_CLAIMS["read"],
            send_claim=EMAIL_CLAIMS["send"],
            draft_claim=EMAIL_CLAIMS["draft"],
            settings=dict(entry.get("adapter_config") or {}),
        )
    if adapter.startswith(GMAIL_ADAPTER_PREFIX) and "gmail:read" in claims:
        for app_id, app in connector_apps.items():
            allowed = set(app.get("allowed_claims") or []) if isinstance(app, Mapping) else set()
            if "gmail:read" in allowed:
                connector_app_id = _clean(app_id)
                break
        try:
            resolved = _clean(resolve_connector_app_id(provider_id))
        except Exception:  # noqa: BLE001 - resolution is a convenience, not a gate
            resolved = ""
        return MailProviderSpec(
            key="gmail" if provider_id == "google" else provider_id,
            label="Gmail" if provider_id == "google" else label,
            provider_id=provider_id,
            connector_app_id=resolved or connector_app_id or "gmail",
            transport="gmail",
            read_claim=GMAIL_CLAIMS["read"],
            send_claim=GMAIL_CLAIMS["send"],
            draft_claim=GMAIL_CLAIMS["draft"],
        )
    return None


async def discover_mail_providers() -> list[MailProviderSpec]:
    """The realm's members for THIS deployment, read from the hub catalog:
    every enabled provider instance whose adapter family and claims say
    'mail'. Adding a provider (Yahoo, a company IMAP server) is a descriptor
    block; this code never learns its name."""
    client = await _hub_client()
    if client is None:
        return []
    try:
        catalog = await client.catalog()
    except Exception as exc:  # noqa: BLE001 - no catalog means no realm, not a crash
        LOGGER.warning("[mail.realm] catalog unavailable: %s", exc)
        return []
    providers = catalog.get("providers") if isinstance(catalog, Mapping) else None
    out: list[MailProviderSpec] = []
    for provider_id, entry in dict(providers or {}).items():
        if not isinstance(entry, Mapping):
            continue
        spec = _spec_from_catalog(_clean(provider_id), entry)
        if spec is not None:
            out.append(spec)
    out.sort(key=lambda spec: (0 if spec.transport == "gmail" else 1))
    return out


def mail_requirement(specs: Iterable[MailProviderSpec], need: str) -> dict[str, Any]:
    """The tool gate for a mail verb: an ``any_of`` group over every realm
    member's own requirement. Declared to the enforcement layer at call time
    because the members are discovered, not known when the tool is written."""
    alternatives = [spec.requirement(need) for spec in specs]
    return {"any_of": alternatives} if len(alternatives) != 1 else alternatives[0]


@dataclass(frozen=True)
class MailAccount:
    account_id: str
    provider: MailProviderSpec
    email: str = ""
    display_name: str = ""
    status: str = ""
    claims: tuple[str, ...] = ()
    bound_claims: tuple[str, ...] | None = None  # None = no agent binding in play

    @property
    def label(self) -> str:
        return self.display_name or self.email or self.account_id

    def allows(self, need: str) -> bool:
        claim = self.provider.claim_for(need)
        if claim not in self.claims:
            return False
        if self.bound_claims is None:
            return True
        return "*" in self.bound_claims or claim in self.bound_claims

    def public_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "provider": self.provider.key,
            "provider_id": self.provider.provider_id,
            "provider_label": self.provider.label,
            "label": self.label,
            "email": self.email,
            "status": self.status,
            "claims": list(self.claims),
            "can_read": self.allows("read"),
            "can_send": self.allows("send"),
            "can_draft": self.allows("draft"),
        }


async def _hub_client() -> Any | None:
    """Connection Hub client for the calling user; None outside a bound scope
    (no user identity, or the module not bound to an entrypoint)."""
    from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import get_current_user_identity
    from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connection_edges import (
        DEFAULT_CONNECTION_HUB_BUNDLE_ID,
    )
    from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_to_kdcube import (
        DelegatedToKdcubeClient,
    )

    identity = get_current_user_identity() or {}
    user_id = _clean(identity.get("user_id"))
    if not user_id or _SERVICE is None:
        return None
    return await DelegatedToKdcubeClient.from_connection_hub(
        _SERVICE,
        user_id=user_id,
        tenant=_clean(identity.get("tenant_id")),
        project=_clean(identity.get("project_id")),
        connection_hub_bundle_id=DEFAULT_CONNECTION_HUB_BUNDLE_ID,
    )


def _binding_for(provider_id: str, account_id: str) -> tuple[str, ...] | None:
    """The agent's claim binding for one account: None when no delegated
    identity is bound (the user's own tools are unrestricted), an empty tuple
    when the agent is bound to nothing on this account (default-closed)."""
    try:
        from connection_hub.agent_account_scope import account_claim_scope_for
    except Exception:  # noqa: BLE001 - scope module absent = no agent binding
        return None
    scope = account_claim_scope_for(provider_id)
    if scope is None:
        return None
    for key in (account_id, "*"):
        if key in scope:
            return tuple(scope[key])
    return ()


async def list_mail_accounts(
    *, specs: Iterable[MailProviderSpec] | None = None
) -> list[MailAccount]:
    """Connected mail accounts across the realm's discovered members, with
    the agent binding applied. Reading account records needs no provider
    claim."""
    client = await _hub_client()
    if client is None:
        return []
    members = list(specs) if specs is not None else await discover_mail_providers()
    out: list[MailAccount] = []
    for spec in members:
        try:
            rows = await client.list_accounts(provider_id=spec.provider_id)
        except Exception as exc:  # noqa: BLE001 - one provider's failure must not hide the others
            LOGGER.warning("[mail.realm] list_accounts failed provider=%s: %s", spec.provider_id, exc)
            continue
        for account in rows:
            if not getattr(account, "connected", False):
                continue
            bound = _binding_for(spec.provider_id, _clean(account.account_id))
            if bound is not None and not bound:
                continue
            out.append(
                MailAccount(
                    account_id=_clean(account.account_id),
                    provider=spec,
                    email=_clean(getattr(account, "email", "")),
                    display_name=_clean(getattr(account, "display_name", "")),
                    status=_clean(getattr(account, "status", "")),
                    claims=tuple(_clean(claim) for claim in (getattr(account, "claims", ()) or ())),
                    bound_claims=bound,
                )
            )
    return out


@dataclass
class MailChoice:
    account: MailAccount | None = None
    denial: dict[str, Any] | None = None
    candidates: list[MailAccount] = field(default_factory=list)


def _account_required_envelope(
    *, where: str, need: str, candidates: list[MailAccount]
) -> dict[str, Any]:
    """The same reason the broker mints for several eligible accounts of ONE
    provider, raised here for several eligible accounts ACROSS providers, so
    a client renders the same choice UI: labeled candidates, resend with
    account_id."""
    rows = [
        {
            "account_id": item.account_id,
            "label": f"{item.label} ({item.provider.label})",
            "email": item.email,
            "provider": item.provider.key,
            "provider_id": item.provider.provider_id,
            "status": item.status,
            "claims": list(item.claims),
        }
        for item in candidates
    ]
    message = (
        "Several connected mail accounts can answer this call; choose one and "
        "resend with its account_id."
    )
    return {
        "ok": False,
        "error": {
            "code": "account_required",
            "message": message,
            "where": where,
            "retryable": True,
        },
        "ret": {
            "reason": "account_required",
            "need": need,
            "candidates": rows,
            "retry_same_request": False,
            "instructions": "Pick one account_id from candidates and call again with it set.",
        },
        "consent": {
            "kind": "account_choice",
            "reason": "account_required",
            "tool_name": where,
            "candidates": rows,
        },
    }


def connect_required_envelope(
    *, where: str, need: str, specs: Iterable[MailProviderSpec], tenant: str = "", project: str = "",
    connection_hub_bundle_id: str = "",
) -> dict[str, Any]:
    """No connected mailbox can serve this call: the connect-first consent,
    offering EVERY mail provider rather than a hard-coded one. The consent
    block itself names one provider action (the hub deep link is per
    provider); the message and ``ret.providers`` carry the whole choice."""
    from connection_hub.delegated_to_kdcube.preflight import (
        connected_account_consent_payload,
    )
    from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connection_edges import (
        DEFAULT_CONNECTION_HUB_BUNDLE_ID,
    )

    options = [
        {
            "provider": spec.key,
            "provider_id": spec.provider_id,
            "provider_label": spec.label,
            "connector_app_id": spec.connector_app_id,
            "claim": spec.claim_for(need),
        }
        for spec in specs
    ]
    failures = [
        {
            "ok": False,
            "provider_id": option["provider_id"],
            "connector_app_id": option["connector_app_id"],
            "claim": option["claim"],
            "account_id": "",
            "error": "connect_required",
            "reason": "connect_required",
            "message": f"Connect {option['provider_label']} and approve {option['claim']}.",
        }
        for option in options
    ]
    payload = connected_account_consent_payload(
        tenant=_clean(tenant),
        project=_clean(project),
        connection_hub_bundle_id=_clean(connection_hub_bundle_id) or DEFAULT_CONNECTION_HUB_BUNDLE_ID,
        missing=[{"ok": False, "tool_name": where, "failures": failures}],
    )
    labels = " or ".join(option["provider_label"] for option in options) or "a mail provider"
    message = f"No connected mail account can do this yet. Connect a mailbox ({labels}) in Connection Hub."
    payload["ok"] = False
    payload["error"] = {
        **dict(payload.get("error") or {}),
        "code": "needs_connected_account_consent",
        "message": message,
        "where": where,
    }
    ret = dict(payload.get("ret") or {})
    ret.update({"reason": "connect_required", "need": need, "providers": options, "message": message})
    payload["ret"] = ret
    payload["consent_required"] = True
    payload["instructions"] = message
    return payload


def choose_mail_account(
    accounts: list[MailAccount],
    *,
    account_id: str = "",
    need: str,
    where: str,
) -> MailChoice:
    """Apply the realm's selection rule. ``need`` is read | send | draft.

    - explicit ``account_id``: that account (an unknown id answers with the
      candidates, never a silent fallback to another account);
    - no eligible account: no choice and no denial; the caller falls back to
      the provider enforcement path, whose connect-first denial names what to
      connect;
    - exactly one eligible account: chosen;
    - several: ``account_required`` with every eligible account as a labeled
      candidate."""
    wanted = _clean(account_id)
    eligible = [item for item in accounts if item.allows(need)]
    if wanted:
        for item in accounts:
            if item.account_id == wanted:
                return MailChoice(account=item, candidates=eligible)
        return MailChoice(
            denial={
                "ok": False,
                "error": {
                    "code": "account_not_found",
                    "message": f"No connected mail account has id {wanted!r}.",
                    "where": where,
                    "retryable": True,
                },
                "ret": {
                    "reason": "account_not_found",
                    "account_id": wanted,
                    "candidates": [item.public_dict() for item in eligible],
                },
            },
            candidates=eligible,
        )
    if not eligible:
        return MailChoice(candidates=[])
    if len(eligible) == 1:
        return MailChoice(account=eligible[0], candidates=eligible)
    return MailChoice(
        denial=_account_required_envelope(where=where, need=need, candidates=eligible),
        candidates=eligible,
    )


__all__ = [
    "EMAIL_CLAIMS",
    "GMAIL_CLAIMS",
    "IMAP_SMTP_ADAPTER",
    "MailAccount",
    "MailChoice",
    "MailProviderSpec",
    "bind_integrations",
    "bind_service",
    "choose_mail_account",
    "connect_required_envelope",
    "discover_mail_providers",
    "list_mail_accounts",
    "mail_requirement",
]
