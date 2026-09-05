# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Human-session assurance boundary for non-delegable management actions."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.auth.AuthManager import RequireRoles, RequireUser

SESSION_CONFIRMATION = "session_confirmation"
FRESH_AUTHENTICATION = "fresh_authentication"
USER_VERIFICATION = "user_verification"

_ASSURANCE_RANK = {
    SESSION_CONFIRMATION: 1,
    FRESH_AUTHENTICATION: 2,
    USER_VERIFICATION: 3,
}
_METHOD_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_ACTION_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,512}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_MAX_CHALLENGE_URL_LENGTH = 4096
_CLOCK_SKEW_SECONDS = 30


class HumanApprovalError(RuntimeError):
    def __init__(self, code: str, *, status_code: int) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True)
class HumanApprovalEvidence:
    subject: str
    assurance: str
    method: str
    request_digest: str
    verified_at: int

    def __post_init__(self) -> None:
        subject = str(self.subject or "").strip()
        if (
            not subject
            or len(subject) > 512
            or any(
                ord(character) < 32 or ord(character) == 127 for character in subject
            )
        ):
            raise ValueError("human approval subject is required")
        if self.assurance not in _ASSURANCE_RANK:
            raise ValueError("human approval assurance is invalid")
        method = str(self.method or "").strip()
        if not _METHOD_RE.fullmatch(method):
            raise ValueError("human approval method is invalid")
        request_digest = str(self.request_digest or "").strip()
        if not _SHA256_RE.fullmatch(request_digest):
            raise ValueError("human approval request digest is invalid")
        if (
            isinstance(self.verified_at, bool)
            or not isinstance(self.verified_at, int)
            or self.verified_at <= 0
        ):
            raise ValueError("human approval verification time is invalid")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "request_digest", request_digest)


@dataclass(frozen=True)
class BrowserAdminSession:
    """Verified human browser session plus a non-secret browser binding."""

    subject: str
    session_id: str
    cookie_binding: str
    username: str = ""
    email: str = ""
    identity_hint: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        subject = str(self.subject or "").strip()
        session_id = str(self.session_id or "").strip()
        cookie_binding = str(self.cookie_binding or "").strip()
        if not subject or len(subject) > 512:
            raise ValueError("human browser subject is invalid")
        if not session_id or len(session_id) > 512:
            raise ValueError("human browser session is invalid")
        if not _SHA256_RE.fullmatch(cookie_binding):
            raise ValueError("human browser binding is invalid")
        hint = dict(self.identity_hint or {})
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "cookie_binding", cookie_binding)
        object.__setattr__(self, "username", str(self.username or "").strip()[:512])
        object.__setattr__(self, "email", str(self.email or "").strip()[:512])
        object.__setattr__(self, "identity_hint", hint)


@dataclass(frozen=True)
class HumanApprovalContext:
    """Exact action a human assurance adapter must approve or challenge."""

    action: str
    tenant: str
    project: str
    transaction_id: str
    request_digest: str
    required_assurance: str
    max_evidence_age_seconds: int
    return_url: str

    def __post_init__(self) -> None:
        action = str(self.action or "").strip()
        tenant = str(self.tenant or "").strip()
        project = str(self.project or "").strip()
        transaction_id = str(self.transaction_id or "").strip()
        request_digest = str(self.request_digest or "").strip()
        return_url = str(self.return_url or "").strip()
        if not _ACTION_RE.fullmatch(action):
            raise ValueError("human approval action is invalid")
        for name, value in (("tenant", tenant), ("project", project)):
            if (
                not value
                or len(value) > 256
                or any(
                    ord(character) < 32 or ord(character) == 127 for character in value
                )
            ):
                raise ValueError(f"human approval {name} is invalid")
        if not _TOKEN_RE.fullmatch(transaction_id):
            raise ValueError("human approval transaction is invalid")
        if not _SHA256_RE.fullmatch(request_digest):
            raise ValueError("human approval request digest is invalid")
        if self.required_assurance not in _ASSURANCE_RANK:
            raise ValueError("human approval assurance is invalid")
        if (
            isinstance(self.max_evidence_age_seconds, bool)
            or not isinstance(self.max_evidence_age_seconds, int)
            or not 1 <= self.max_evidence_age_seconds <= 900
        ):
            raise ValueError("human approval evidence age is invalid")
        parsed_return = urlsplit(return_url)
        if (
            not return_url.startswith("/")
            or return_url.startswith("//")
            or parsed_return.scheme
            or parsed_return.netloc
            or parsed_return.fragment
            or len(return_url) > 2048
        ):
            raise ValueError("human approval return URL is invalid")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "tenant", tenant)
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(self, "return_url", return_url)


@dataclass(frozen=True)
class HumanApprovalChallenge:
    """Provider-owned browser step-up bound to a HumanApprovalContext."""

    authorization_url: str
    method: str

    def __post_init__(self) -> None:
        authorization_url = str(self.authorization_url or "").strip()
        method = str(self.method or "").strip()
        if not _METHOD_RE.fullmatch(method):
            raise ValueError("human approval challenge method is invalid")
        if (
            not authorization_url
            or len(authorization_url) > _MAX_CHALLENGE_URL_LENGTH
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in authorization_url
            )
        ):
            raise ValueError("human approval challenge URL is invalid")
        try:
            parsed = urlsplit(authorization_url)
            _port = parsed.port
        except ValueError:
            raise ValueError("human approval challenge URL is invalid") from None
        is_relative = (
            not parsed.scheme
            and not parsed.netloc
            and authorization_url.startswith("/")
            and not authorization_url.startswith("//")
        )
        is_https = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
        )
        if parsed.fragment or not (is_relative or is_https):
            raise ValueError("human approval challenge URL is invalid")
        object.__setattr__(self, "authorization_url", authorization_url)
        object.__setattr__(self, "method", method)


HumanApprovalOutcome: TypeAlias = HumanApprovalEvidence | HumanApprovalChallenge
HumanApprovalPhase: TypeAlias = Literal["present", "commit"]


class HumanApprovalVerifier(Protocol):
    async def evaluate(
        self,
        request: Request,
        *,
        context: HumanApprovalContext,
        phase: HumanApprovalPhase,
    ) -> HumanApprovalOutcome: ...


def assurance_satisfies(actual: str, required: str) -> bool:
    if actual not in _ASSURANCE_RANK or required not in _ASSURANCE_RANK:
        return False
    return _ASSURANCE_RANK[actual] >= _ASSURANCE_RANK[required]


def _cookie_auth_only(request: Request) -> bool:
    auth = get_settings().AUTH
    if str(request.headers.get("authorization") or "").strip():
        return False
    if str(
        request.headers.get(auth.ID_TOKEN_HEADER_NAME)
        or request.headers.get(auth.ID_TOKEN_HEADER_NAME.lower())
        or ""
    ).strip():
        return False
    return any(
        bool(request.cookies.get(name))
        for name in (
            auth.AUTH_TOKEN_COOKIE_NAME,
            auth.ID_TOKEN_COOKIE_NAME,
        )
        if name
    )


def _subject(session: Any) -> str:
    authority = getattr(session, "identity_authority", None)
    if isinstance(authority, dict):
        candidate = str(authority.get("platform_user_id") or "").strip()
        if candidate:
            return candidate
    for value in (
        getattr(session, "user_id", None),
        getattr(session, "username", None),
    ):
        candidate = str(value or "").strip()
        if candidate:
            return candidate
    return ""


def _browser_cookie_binding(request: Request) -> str:
    auth = get_settings().AUTH
    rows: list[tuple[str, str]] = []
    for name in (
        auth.AUTH_TOKEN_COOKIE_NAME,
        auth.ID_TOKEN_COOKIE_NAME,
    ):
        if not name:
            continue
        value = str(request.cookies.get(name) or "")
        if value:
            rows.append((name, hashlib.sha256(value.encode("utf-8")).hexdigest()))
    if not rows:
        return ""
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _identity_hint(request: Request) -> dict[str, Any]:
    """Read routing hints from browser JWTs without trusting their contents."""

    auth = get_settings().AUTH
    retained = {
        "iss",
        "aud",
        "client_id",
        "sub",
        "provider",
        "provider_subject",
        "token_use",
    }
    for name in (auth.ID_TOKEN_COOKIE_NAME, auth.AUTH_TOKEN_COOKIE_NAME):
        token = str(request.cookies.get(name) or "").strip()
        if not token:
            continue
        try:
            import jwt

            claims = jwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "verify_exp": False,
                    "verify_aud": False,
                },
            )
        except Exception:
            continue
        if isinstance(claims, dict):
            return {key: claims[key] for key in retained if key in claims}
    return {}


async def resolve_browser_admin_session(request: Request) -> BrowserAdminSession:
    """Resolve a cookie-authenticated super-admin and bind the exact browser."""

    if not _cookie_auth_only(request):
        raise HumanApprovalError(
            "human_browser_session_required",
            status_code=401,
        )

    # Imported lazily because resolver initialization belongs to the running
    # processor, not module import or offline contract tests.
    from kdcube_ai_app.apps.chat.ingress.resolvers import get_fastapi_adapter

    try:
        session = await get_fastapi_adapter().process_request(
            request,
            requirements=[
                RequireUser(),
                RequireRoles("kdcube:role:super-admin"),
            ],
            bypass_throttling=True,
            bypass_gate=True,
            bypass_backpressure=True,
            header_only_auth=False,
            connection_hub=False,
        )
    except HTTPException as exc:
        status = 401 if int(exc.status_code) == 401 else 403
        raise HumanApprovalError(
            "human_browser_session_required"
            if status == 401
            else "human_admin_authority_required",
            status_code=status,
        ) from None

    subject = _subject(session)
    binding = _browser_cookie_binding(request)
    session_id = str(getattr(session, "session_id", None) or "").strip()
    if subject and binding and not session_id:
        session_id = hashlib.sha256(f"{subject}\n{binding}".encode("utf-8")).hexdigest()
    try:
        return BrowserAdminSession(
            subject=subject,
            session_id=session_id,
            cookie_binding=binding,
            username=str(getattr(session, "username", None) or ""),
            email=str(getattr(session, "email", None) or ""),
            identity_hint=_identity_hint(request),
        )
    except ValueError:
        raise HumanApprovalError(
            "human_browser_session_invalid",
            status_code=503,
        ) from None


class BrowserSessionHumanApprovalVerifier:
    """Verify an admin through the configured platform's browser cookie.

    This verifier deliberately disables the Connection Hub authentication
    surface. A delegated bearer, including one carrying an admin role, is not
    human approval. Stronger verifiers may implement the same protocol and
    return fresh-authentication or user-verification evidence.
    """

    async def evaluate(
        self,
        request: Request,
        *,
        context: HumanApprovalContext,
        phase: HumanApprovalPhase,
    ) -> HumanApprovalEvidence:
        if phase not in {"present", "commit"}:
            raise HumanApprovalError(
                "human_approval_phase_invalid",
                status_code=503,
            )
        admin = await resolve_browser_admin_session(request)

        try:
            evidence = HumanApprovalEvidence(
                subject=admin.subject,
                assurance=SESSION_CONFIRMATION,
                method="kdcube_platform_browser_session",
                request_digest=context.request_digest,
                verified_at=int(time.time()),
            )
        except ValueError:
            raise HumanApprovalError(
                "human_browser_session_invalid",
                status_code=503,
            ) from None
        if not assurance_satisfies(
            evidence.assurance,
            context.required_assurance,
        ):
            raise HumanApprovalError(
                "human_approval_step_up_unavailable",
                status_code=409,
            )
        return evidence


async def evaluate_human_approval(
    request: Request,
    *,
    context: HumanApprovalContext,
    phase: HumanApprovalPhase,
    clock: Any = time.time,
) -> HumanApprovalOutcome:
    """Resolve an adapter and enforce its typed, request-bound outcome."""

    if phase not in {"present", "commit"}:
        raise HumanApprovalError(
            "human_approval_phase_invalid",
            status_code=503,
        )
    verifier = getattr(request.app.state, "human_approval_verifier", None)
    if verifier is None:
        if context.required_assurance == SESSION_CONFIRMATION:
            verifier = BrowserSessionHumanApprovalVerifier()
        else:
            from kdcube_ai_app.apps.chat.proc.rest.management.human_approval_adapters import (
                descriptor_human_approval_verifier,
            )

            verifier = descriptor_human_approval_verifier(
                request,
                required_assurance=context.required_assurance,
            )
    outcome = await verifier.evaluate(
        request,
        context=context,
        phase=phase,
    )
    if isinstance(outcome, HumanApprovalChallenge):
        return outcome
    if not isinstance(outcome, HumanApprovalEvidence):
        raise HumanApprovalError(
            "human_approval_evidence_invalid",
            status_code=503,
        )
    now = int(clock())
    if (
        not secrets.compare_digest(outcome.request_digest, context.request_digest)
        or outcome.verified_at > now + _CLOCK_SKEW_SECONDS
        or now - outcome.verified_at > context.max_evidence_age_seconds
    ):
        raise HumanApprovalError(
            "human_approval_evidence_invalid",
            status_code=503,
        )
    if not assurance_satisfies(
        outcome.assurance,
        context.required_assurance,
    ):
        raise HumanApprovalError(
            "human_approval_step_up_unavailable",
            status_code=409,
        )
    return outcome


__all__ = [
    "FRESH_AUTHENTICATION",
    "SESSION_CONFIRMATION",
    "USER_VERIFICATION",
    "BrowserSessionHumanApprovalVerifier",
    "BrowserAdminSession",
    "HumanApprovalChallenge",
    "HumanApprovalContext",
    "HumanApprovalError",
    "HumanApprovalEvidence",
    "HumanApprovalOutcome",
    "HumanApprovalPhase",
    "HumanApprovalVerifier",
    "assurance_satisfies",
    "evaluate_human_approval",
    "resolve_browser_admin_session",
]
