# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# chat/sdk/examples/mcp/two_level_client/two_level.py
#
# The reusable, dependency-light classifier for KDCube's two-level consent
# lifecycle. This is THE piece other MCP clients copy.

"""Read a KDCube governed-door tool result and name which of the two
authorization levels denied the call, and exactly how the user fixes it.

A KDCube governed MCP door never answers a policy denial with a bare HTTP 403.
It returns a structured *consent envelope* AS THE TOOL RESULT, and every denial
belongs to exactly one of two sequential levels:

- **Level 1 - the caller's own grant.** The bearer's delegated credential
  lacks a grant the operation needs. The door names the missing grants and a
  Connection Hub deep link where the user approves them for this caller.
  Marker: ``error == "delegated_consent_required"`` (or ``consent.kind ==
  "delegated_agent_grant"``).

- **Level 2 - the connected account plus the per-account binding.** The caller
  holds the MCP grant, but the user-to-provider side cannot satisfy the call.
  A ``reason`` says precisely why, and ``retry_hint`` says whether replaying the
  same call works after the user acts. Marker: ``error.code ==
  "needs_connected_account_consent"`` (or ``consent.kind ==
  "delegated_to_kdcube.connected_account"``). The reason vocabulary:

  =====================  ==============================================================
  reason                 what the user does
  =====================  ==============================================================
  connect_required       connect an account on the backing provider at the hub URL
  claim_upgrade_required  approve the listed claims for an existing account
  reconnect_required     reconnect an account whose stored credential stopped working
  account_required       resend the SAME call with ``account_id`` from ``candidates``
  agent_grant_required   tick the claim for an account on this caller's grant card
  =====================  ==============================================================

The contract the door ships is *relay, do not retry blindly*: only
``account_required`` resolves by resending (with an ``account_id``); every other
reason needs a human action at the Connection Hub URL first.

``classify_tool_result`` collapses all of this into one small dataclass. Give it
a KDCube tool result (already parsed to a mapping - see
``result_payload_from_call_tool`` for pulling that out of an ``mcp`` SDK
``CallToolResult``) and it tells you the level, the reason, whether replay
works, and a single human sentence naming the fix.

The classifier is deliberately dependency-light (standard library only) and
never raises on a malformed or partial payload: an unrecognised shape degrades
to ``level=0`` with a clear message, so a consumer can always trust the
outcome object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping

# --------------------------------------------------------------------------
# The vocabulary, verbatim from the door (kept here so this module is a
# self-contained reference a consumer can copy without importing KDCube).
# --------------------------------------------------------------------------

LEVEL_1_ERROR = "delegated_consent_required"
LEVEL_1_CODE = "connections.consent_needed"
LEVEL_1_CONSENT_KIND = "delegated_agent_grant"

LEVEL_2_CODE = "needs_connected_account_consent"
LEVEL_2_CONSENT_KIND = "delegated_to_kdcube.connected_account"

# Level-2 reasons, in the order the door's own instructions list them.
REASON_CONNECT_REQUIRED = "connect_required"
REASON_CLAIM_UPGRADE_REQUIRED = "claim_upgrade_required"
REASON_RECONNECT_REQUIRED = "reconnect_required"
REASON_ACCOUNT_REQUIRED = "account_required"
REASON_AGENT_GRANT_REQUIRED = "agent_grant_required"

LEVEL_2_REASONS = frozenset({
    REASON_CONNECT_REQUIRED,
    REASON_CLAIM_UPGRADE_REQUIRED,
    REASON_RECONNECT_REQUIRED,
    REASON_ACCOUNT_REQUIRED,
    REASON_AGENT_GRANT_REQUIRED,
})

# Only account_required is fixed by replaying the call; every other reason
# needs a human action at the hub URL first. This set encodes the door's
# "relay, don't retry blindly" contract for consumers that want a boolean.
REASONS_FIXED_BY_RESEND = frozenset({REASON_ACCOUNT_REQUIRED})


@dataclass
class ConsentOutcome:
    """The verdict on one KDCube governed-door tool result.

    ``level`` is the load-bearing field:

    - ``0`` - not a two-level denial. Either the call SUCCEEDED (``ok`` is
      True) or the payload was unrecognised/malformed (``ok`` is False,
      ``code == "unrecognized"``); ``next_action`` explains which.
    - ``1`` - the caller's own grant is missing (level 1).
    - ``2`` - the connected account plus the per-account binding cannot satisfy
      the call (level 2); ``reason`` names why and ``retry_hint`` says whether
      resending works.

    ``next_action`` is a single human sentence stating what the USER must do to
    clear the denial (empty on success). It is safe to relay verbatim.
    """

    ok: bool
    level: int
    code: str = ""
    reason: str = ""          # level-2 only
    retry_hint: bool = False
    next_action: str = ""
    connection_hub_url: str = ""
    provider_id: str = ""
    claims: List[str] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resend_with_account_id(self) -> bool:
        """True when the fix is to replay the SAME call with an ``account_id``
        from ``candidates`` (the only reason resolved by resending)."""
        return self.level == 2 and self.reason in REASONS_FIXED_BY_RESEND

    def candidate_account_ids(self) -> List[str]:
        """The account ids offered for an ``account_required`` denial."""
        out: List[str] = []
        for item in self.candidates:
            if isinstance(item, Mapping):
                aid = _clean(item.get("account_id"))
                if aid:
                    out.append(aid)
        return out


# --------------------------------------------------------------------------
# small, total helpers (never raise)
# --------------------------------------------------------------------------

def _clean(value: Any) -> str:
    try:
        return str(value).strip() if value is not None else ""
    except Exception:
        return ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _str_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value if _clean(v)]
    single = _clean(value)
    return [single] if single else []


def _error_code(payload: Mapping[str, Any]) -> str:
    """The error code, robust to ``error`` being a bare string or a dict.

    Level 1 ships ``error`` as the string ``"delegated_consent_required"``;
    level 2 ships ``error`` as ``{"code": "needs_connected_account_consent",
    ...}``. Either way this returns the discriminating token.
    """
    error = payload.get("error")
    if isinstance(error, str):
        return _clean(error)
    if isinstance(error, Mapping):
        return _clean(error.get("code"))
    return ""


def _consent_block(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The consent block, wherever the envelope variant places it.

    Seen in the wild: top-level ``consent`` (both gates), ``error.consent``
    (the account-broker wrapper nests it under the error too), and
    ``error.details.consent`` (a defensive variant). First non-empty wins.
    """
    for candidate in (
        payload.get("consent"),
        _as_mapping(payload.get("error")).get("consent"),
        _as_mapping(_as_mapping(payload.get("error")).get("details")).get("consent"),
    ):
        block = _as_mapping(candidate)
        if block:
            return block
    return {}


def _pick_hub_url(payload: Mapping[str, Any], consent: Mapping[str, Any]) -> str:
    """The Connection Hub URL, across every field name the door uses for it."""
    for value in (
        consent.get("url"),
        consent.get("connection_hub_url"),
        payload.get("connection_hub_url"),
        _as_mapping(payload.get("error")).get("action_url"),
        payload.get("action_url"),
    ):
        url = _clean(value)
        if url:
            return url
    return ""


def _first_ok(*values: Any) -> str:
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return ""


# --------------------------------------------------------------------------
# next_action sentences - the human fix, one line, per level/reason
# --------------------------------------------------------------------------

def _level1_action(claims: List[str], hub_url: str) -> str:
    claims_txt = ", ".join(claims) if claims else "the missing grant"
    if hub_url:
        return f"Open {hub_url} and grant this caller: {claims_txt}. Do not retry until approved."
    return (
        f"Approve this caller's missing grant ({claims_txt}) in Connection Hub "
        "(Delegated by KDCube), then retry. Do not retry blindly."
    )


def _level2_action(
    reason: str,
    provider_id: str,
    claims: List[str],
    hub_url: str,
    candidates: List[dict],
) -> str:
    provider = provider_id or "the backing provider"
    claims_txt = ", ".join(claims) if claims else "the required access"
    at_hub = f" at {hub_url}" if hub_url else " in Connection Hub"

    if reason == REASON_CONNECT_REQUIRED:
        return f"Connect a {provider} account{at_hub}, then retry the same call."
    if reason == REASON_CLAIM_UPGRADE_REQUIRED:
        return f"Approve {claims_txt} for your {provider} account{at_hub}, then retry the same call."
    if reason == REASON_RECONNECT_REQUIRED:
        return f"Reconnect your {provider} account{at_hub} (its stored credential stopped working), then retry."
    if reason == REASON_AGENT_GRANT_REQUIRED:
        return (
            f"Tick {claims_txt} for a {provider} account on this caller's grant card"
            f"{at_hub} (Delegated by KDCube), then retry the same call."
        )
    if reason == REASON_ACCOUNT_REQUIRED:
        ids = [_clean(c.get("account_id")) for c in candidates if isinstance(c, Mapping) and _clean(c.get("account_id"))]
        pick = ids[0] if ids else "<one of candidates>"
        listing = f" (candidates: {', '.join(ids)})" if ids else ""
        return (
            f"Several {provider} accounts match - resend the SAME call with "
            f"account_id={pick}{listing}."
        )
    # A level-2 code with an unlisted/absent reason: still actionable via the hub.
    return f"Resolve the connected-account requirement for {provider}{at_hub}, then retry."


# --------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------

def classify_tool_result(payload: Mapping[str, Any]) -> ConsentOutcome:
    """Classify a KDCube governed-door tool result into a :class:`ConsentOutcome`.

    ``payload`` is the tool result already parsed to a mapping (KDCube returns
    the consent envelope as the tool result, not as an HTTP error). Use
    :func:`result_payload_from_call_tool` to obtain it from an ``mcp`` SDK
    ``CallToolResult``.

    The function is total: it never raises. An unrecognised or malformed
    payload degrades to ``level=0`` with ``code="unrecognized"`` and a
    ``next_action`` that says so.
    """
    if not isinstance(payload, Mapping):
        return ConsentOutcome(
            ok=False,
            level=0,
            code="unrecognized",
            next_action="The tool result was not a JSON object; nothing to classify.",
            raw={},
        )

    ok_flag = payload.get("ok")

    # Success: the door passed BOTH levels. `ok is True` is the explicit signal;
    # a payload with a `ret`/result and no error/consent is also a success.
    error_present = bool(_error_code(payload)) or bool(_consent_block(payload))
    if ok_flag is True or (ok_flag is None and not error_present):
        return ConsentOutcome(
            ok=True,
            level=0,
            code="ok",
            next_action="",
            raw=payload,
        )

    consent = _consent_block(payload)
    code = _error_code(payload)
    kind = _clean(consent.get("kind"))
    reason = _first_ok(consent.get("reason"), payload.get("reason"))
    provider_id = _first_ok(consent.get("provider_id"), payload.get("provider_id"))
    hub_url = _pick_hub_url(payload, consent)

    # ----- LEVEL 2: the connected account + the per-account binding -----
    # Checked first: its markers (the code, the consent kind, and the reason
    # vocabulary) are the most specific.
    is_level2 = (
        code == LEVEL_2_CODE
        or kind == LEVEL_2_CONSENT_KIND
        or reason in LEVEL_2_REASONS
    )
    if is_level2:
        claims = _str_list(consent.get("claims")) or _str_list(payload.get("missing_grants"))
        candidates = [c for c in (consent.get("candidates") or []) if isinstance(c, Mapping)]
        retry_hint = bool(consent.get("retry_hint")) or bool(payload.get("retry_hint"))
        reason = reason or REASON_CONNECT_REQUIRED
        return ConsentOutcome(
            ok=False,
            level=2,
            code=code or LEVEL_2_CODE,
            reason=reason,
            retry_hint=retry_hint,
            next_action=_level2_action(reason, provider_id, claims, hub_url, candidates),
            connection_hub_url=hub_url,
            provider_id=provider_id,
            claims=claims,
            candidates=[dict(c) for c in candidates],
            raw=payload,
        )

    # ----- LEVEL 1: the caller's own delegated grant -----
    is_level1 = (
        code in (LEVEL_1_ERROR, LEVEL_1_CODE)
        or kind == LEVEL_1_CONSENT_KIND
        or reason == LEVEL_1_ERROR
    )
    if is_level1:
        claims = _str_list(payload.get("missing_grants")) or _str_list(consent.get("claims"))
        return ConsentOutcome(
            ok=False,
            level=1,
            code=code or LEVEL_1_ERROR,
            reason="",  # level-2 concept only
            retry_hint=False,  # a level-1 grant miss is never fixed by a blind resend
            next_action=_level1_action(claims, hub_url),
            connection_hub_url=hub_url,
            provider_id=provider_id,
            claims=claims,
            raw=payload,
        )

    # ----- Unrecognised denial: degrade cleanly, stay useful -----
    message = _first_ok(
        _as_mapping(payload.get("error")).get("message"),
        payload.get("message"),
        code,
    )
    return ConsentOutcome(
        ok=False,
        level=0,
        code=code or "unrecognized",
        next_action=(
            "This tool result was not a recognised KDCube two-level consent "
            "envelope" + (f": {message}" if message else "; treat it as an opaque failure.")
        ),
        connection_hub_url=hub_url,
        raw=payload,
    )


# --------------------------------------------------------------------------
# pulling the payload out of an `mcp` SDK CallToolResult
# --------------------------------------------------------------------------

def result_payload_from_call_tool(result: Any) -> Mapping[str, Any]:
    """Extract the JSON payload from an ``mcp`` SDK ``CallToolResult``.

    KDCube returns the consent envelope (or the success result) as the tool
    RESULT. Prefer ``structuredContent`` when present; otherwise parse the first
    text content block as JSON. This mirrors the in-house adapter
    (``runtime/mcp/mcp_adapter.py``). Returns ``{}`` when nothing parses, so the
    classifier still receives a mapping.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, Mapping):
        return structured

    import json

    for block in (getattr(result, "content", None) or []):
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    return {}


__all__ = [
    "ConsentOutcome",
    "classify_tool_result",
    "result_payload_from_call_tool",
    "LEVEL_2_REASONS",
    "REASONS_FIXED_BY_RESEND",
]
