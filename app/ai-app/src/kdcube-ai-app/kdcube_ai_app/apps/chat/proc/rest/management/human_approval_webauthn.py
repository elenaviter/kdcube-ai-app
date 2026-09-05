# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""WebAuthn enrollment and one-use user-verification approval."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from connection_hub.connection_edges import request_origin
from fastapi import Request
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval import (
    FRESH_AUTHENTICATION,
    USER_VERIFICATION,
    BrowserAdminSession,
    HumanApprovalChallenge,
    HumanApprovalContext,
    HumanApprovalError,
    HumanApprovalEvidence,
    HumanApprovalOutcome,
    HumanApprovalPhase,
    evaluate_human_approval,
    resolve_browser_admin_session,
)
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval_oidc import (
    human_approval_config,
    human_approval_store,
)
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval_store import (
    HumanProofRecord,
    PasskeyCredentialRecord,
)
from kdcube_ai_app.apps.chat.sdk.config import get_settings

PASSKEY_AUTHENTICATION_PATH = "/api/integrations/management/v1/human-approval/webauthn"
PASSKEY_REGISTRATION_PATH = (
    "/api/integrations/management/v1/human-approval/passkeys/register"
)
PASSKEY_REGISTRATION_ACTION = "kdcube.management.human-approval.passkey.register"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    raw = str(value or "").strip()
    if not raw or len(raw) > 65536:
        raise ValueError("base64url value is invalid")
    return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))


def _safe_return_url(value: str) -> str:
    result = str(value or "").strip()
    parsed = urlsplit(result)
    if (
        not result.startswith("/")
        or result.startswith("//")
        or parsed.scheme
        or parsed.netloc
        or parsed.fragment
        or len(result) > 2048
    ):
        raise HumanApprovalError(
            "human_approval_return_url_invalid",
            status_code=400,
        )
    return result


def _rp_context(request: Request) -> tuple[str, str]:
    config = human_approval_config(request).webauthn
    if not config.enabled:
        raise HumanApprovalError(
            "human_approval_webauthn_disabled",
            status_code=409,
        )
    origin = request_origin(request).rstrip("/")
    parsed = urlsplit(origin)
    if not parsed.hostname:
        raise HumanApprovalError(
            "human_approval_webauthn_origin_invalid",
            status_code=503,
        )
    if config.allowed_origins and origin not in config.allowed_origins:
        raise HumanApprovalError(
            "human_approval_webauthn_origin_not_allowed",
            status_code=403,
        )
    rp_id = config.rp_id or parsed.hostname
    host = parsed.hostname.lower()
    if host != rp_id and not host.endswith(f".{rp_id}"):
        raise HumanApprovalError(
            "human_approval_webauthn_rp_mismatch",
            status_code=503,
        )
    return rp_id, origin


def _same_admin(expected: BrowserAdminSession, actual: BrowserAdminSession) -> bool:
    return all(
        (
            secrets.compare_digest(expected.subject, actual.subject),
            secrets.compare_digest(expected.session_id, actual.session_id),
            secrets.compare_digest(expected.cookie_binding, actual.cookie_binding),
        )
    )


def _registration_digest(
    *,
    enrollment_id: str,
    admin: BrowserAdminSession,
    rp_id: str,
    policy: str,
    return_url: str,
) -> str:
    payload = json.dumps(
        {
            "action": PASSKEY_REGISTRATION_ACTION,
            "enrollment_id": enrollment_id,
            "subject": admin.subject,
            "rp_id": rp_id,
            "policy": policy,
            "return_url": return_url,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _require_webauthn() -> dict[str, Any]:
    try:
        from webauthn import (
            generate_authentication_options,
            generate_registration_options,
            verify_authentication_response,
            verify_registration_response,
        )
        from webauthn.helpers import options_to_json_dict, parse_attestation_object
        from webauthn.helpers.structs import (
            AttestationConveyancePreference,
            AttestationFormat,
            AuthenticatorSelectionCriteria,
            PublicKeyCredentialDescriptor,
            ResidentKeyRequirement,
            UserVerificationRequirement,
        )
    except ImportError:
        raise HumanApprovalError(
            "human_approval_webauthn_dependency_unavailable",
            status_code=503,
        ) from None
    return {
        "generate_authentication_options": generate_authentication_options,
        "generate_registration_options": generate_registration_options,
        "verify_authentication_response": verify_authentication_response,
        "verify_registration_response": verify_registration_response,
        "options_to_json_dict": options_to_json_dict,
        "parse_attestation_object": parse_attestation_object,
        "AttestationConveyancePreference": AttestationConveyancePreference,
        "AttestationFormat": AttestationFormat,
        "AuthenticatorSelectionCriteria": AuthenticatorSelectionCriteria,
        "PublicKeyCredentialDescriptor": PublicKeyCredentialDescriptor,
        "ResidentKeyRequirement": ResidentKeyRequirement,
        "UserVerificationRequirement": UserVerificationRequirement,
    }


def _trusted_roots(request: Request) -> dict[Any, list[bytes]]:
    library = _require_webauthn()
    configured = human_approval_config(request).webauthn.trusted_attestation_root_files
    roots: dict[Any, list[bytes]] = {}
    for name, paths in configured.items():
        try:
            fmt = library["AttestationFormat"](name)
            values = [Path(path).read_bytes() for path in paths]
        except Exception:  # noqa: BLE001 - normalize trust-root parsing failures
            raise HumanApprovalError(
                "human_approval_webauthn_trust_roots_invalid",
                status_code=503,
            ) from None
        if any(b"BEGIN CERTIFICATE" not in value for value in values):
            raise HumanApprovalError(
                "human_approval_webauthn_trust_roots_invalid",
                status_code=503,
            )
        roots[fmt] = values
    return roots


class WebAuthnHumanApprovalVerifier:
    """Require a fresh, operation-bound WebAuthn assertion."""

    async def evaluate(
        self,
        request: Request,
        *,
        context: HumanApprovalContext,
        phase: HumanApprovalPhase,
    ) -> HumanApprovalOutcome:
        config = human_approval_config(request)
        admin = await resolve_browser_admin_session(request)
        rp_id, origin = _rp_context(request)
        store = human_approval_store(request, config=config)
        try:
            proof = await store.proof(
                context=context,
                admin=admin,
                consume=phase == "commit",
            )
            credentials = _eligible_credentials(
                await store.passkeys(
                    subject=admin.subject,
                    rp_id=rp_id,
                ),
                policy=config.webauthn.credential_policy,
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
        if not credentials:
            return HumanApprovalChallenge(
                authorization_url=(
                    f"{PASSKEY_REGISTRATION_PATH}?"
                    + urlencode({"return_to": context.return_url})
                ),
                method="webauthn_enrollment_required",
            )
        challenge_bytes = secrets.token_bytes(32)
        try:
            challenge = await store.create_passkey_challenge(
                purpose="authentication",
                challenge=_b64(challenge_bytes),
                rp_id=rp_id,
                origin=origin,
                context=context,
                admin=admin,
                credential_ids=tuple(item.credential_id for item in credentials),
            )
        except Exception:  # noqa: BLE001 - normalize storage boundary failures
            raise HumanApprovalError(
                "human_approval_store_unavailable",
                status_code=503,
            ) from None
        return HumanApprovalChallenge(
            authorization_url=(
                f"{PASSKEY_AUTHENTICATION_PATH}?"
                + urlencode({"state": challenge.state})
            ),
            method="webauthn_user_verification",
        )


async def authentication_options(
    request: Request,
    *,
    state: str,
) -> dict[str, Any]:
    config = human_approval_config(request)
    store = human_approval_store(request, config=config)
    admin = await resolve_browser_admin_session(request)
    challenge = await store.passkey_challenge(state)
    if (
        challenge is None
        or challenge.purpose != "authentication"
        or not _same_admin(challenge.admin, admin)
    ):
        raise HumanApprovalError(
            "human_approval_passkey_challenge_invalid",
            status_code=403,
        )
    library = _require_webauthn()
    credentials = await store.passkeys(subject=admin.subject, rp_id=challenge.rp_id)
    by_id = {item.credential_id: item for item in credentials}
    if any(item not in by_id for item in challenge.credential_ids):
        raise HumanApprovalError(
            "human_approval_passkey_credential_changed",
            status_code=409,
        )
    options = library["generate_authentication_options"](
        rp_id=challenge.rp_id,
        challenge=_unb64(challenge.challenge),
        timeout=config.webauthn.timeout_milliseconds,
        allow_credentials=[
            library["PublicKeyCredentialDescriptor"](id=_unb64(credential_id))
            for credential_id in challenge.credential_ids
        ],
        user_verification=library["UserVerificationRequirement"].REQUIRED,
    )
    return {
        "state": state,
        "options": library["options_to_json_dict"](options),
    }


def _credential_id(payload: Mapping[str, Any]) -> str:
    credential_id = str(payload.get("id") or "").strip()
    raw_id = str(payload.get("rawId") or "").strip()
    if (
        not credential_id
        or not raw_id
        or len(credential_id) > 4096
        or not secrets.compare_digest(credential_id, raw_id)
    ):
        raise HumanApprovalError(
            "human_approval_passkey_response_invalid",
            status_code=400,
        )
    return credential_id


def _enforce_credential_policy(
    *,
    configured_policy: str,
    credential: PasskeyCredentialRecord,
    device_type: str,
    user_verified: bool,
) -> str:
    if not user_verified:
        raise HumanApprovalError(
            "human_approval_user_verification_required",
            status_code=403,
        )
    if configured_policy == "single_device" and device_type != "single_device":
        raise HumanApprovalError(
            "human_approval_single_device_required",
            status_code=403,
        )
    if (
        configured_policy == "attested_hardware"
        and credential.policy != "attested_hardware"
    ):
        raise HumanApprovalError(
            "human_approval_attested_hardware_required",
            status_code=403,
        )
    return {
        "verified_passkey": "webauthn_uv",
        "single_device": "webauthn_uv_single_device",
        "attested_hardware": "webauthn_uv_attested_hardware",
    }[configured_policy]


def _eligible_credentials(
    credentials: tuple[PasskeyCredentialRecord, ...],
    *,
    policy: str,
) -> tuple[PasskeyCredentialRecord, ...]:
    if policy == "single_device":
        return tuple(
            credential
            for credential in credentials
            if credential.device_type == "single_device"
        )
    if policy == "attested_hardware":
        return tuple(
            credential
            for credential in credentials
            if credential.policy == "attested_hardware"
        )
    return credentials


async def complete_authentication(
    request: Request,
    *,
    state: str,
    credential_payload: Mapping[str, Any],
) -> str:
    config = human_approval_config(request)
    store = human_approval_store(request, config=config)
    admin = await resolve_browser_admin_session(request)
    challenge = await store.passkey_challenge(state)
    if (
        challenge is None
        or challenge.purpose != "authentication"
        or not _same_admin(challenge.admin, admin)
    ):
        raise HumanApprovalError(
            "human_approval_passkey_challenge_invalid",
            status_code=403,
        )
    credential_id = _credential_id(credential_payload)
    credentials = await store.passkeys(subject=admin.subject, rp_id=challenge.rp_id)
    credential = next(
        (
            item
            for item in credentials
            if secrets.compare_digest(item.credential_id, credential_id)
            and item.credential_id in challenge.credential_ids
        ),
        None,
    )
    if credential is None:
        raise HumanApprovalError(
            "human_approval_passkey_credential_invalid",
            status_code=403,
        )
    library = _require_webauthn()
    verifier = getattr(
        request.app.state,
        "human_approval_webauthn_authentication_verifier",
        library["verify_authentication_response"],
    )
    try:
        verified = verifier(
            credential=dict(credential_payload),
            expected_challenge=_unb64(challenge.challenge),
            expected_rp_id=challenge.rp_id,
            expected_origin=challenge.origin,
            credential_public_key=_unb64(credential.public_key),
            credential_current_sign_count=credential.sign_count,
            require_user_verification=True,
        )
    except HumanApprovalError:
        raise
    except Exception:  # noqa: BLE001 - conceal authenticator verifier details
        raise HumanApprovalError(
            "human_approval_passkey_response_invalid",
            status_code=403,
        ) from None
    device_type = str(
        getattr(
            verified.credential_device_type, "value", verified.credential_device_type
        )
    )
    method = _enforce_credential_policy(
        configured_policy=config.webauthn.credential_policy,
        credential=credential,
        device_type=device_type,
        user_verified=bool(verified.user_verified),
    )
    updated = await store.update_passkey_counter(
        subject=admin.subject,
        rp_id=challenge.rp_id,
        credential_id=credential.credential_id,
        expected_count=credential.sign_count,
        new_count=int(verified.new_sign_count),
        device_type=device_type,
        backed_up=bool(verified.credential_backed_up),
    )
    if not updated or not await store.consume_passkey_challenge(challenge):
        raise HumanApprovalError(
            "human_approval_passkey_response_replayed",
            status_code=409,
        )
    evidence = HumanApprovalEvidence(
        subject=admin.subject,
        assurance=USER_VERIFICATION,
        method=method,
        request_digest=challenge.context.request_digest,
        verified_at=int(time.time()),
    )
    if not await store.put_proof(
        HumanProofRecord(
            evidence=evidence,
            context=challenge.context,
            admin=admin,
        )
    ):
        raise HumanApprovalError(
            "human_approval_passkey_response_replayed",
            status_code=409,
        )
    return challenge.context.return_url


async def start_enrollment(
    request: Request,
    *,
    final_return_url: str,
) -> HumanApprovalChallenge:
    config = human_approval_config(request)
    admin = await resolve_browser_admin_session(request)
    rp_id, _origin = _rp_context(request)
    final_return = _safe_return_url(final_return_url)
    enrollment_id = secrets.token_urlsafe(32)
    settings = get_settings()
    context = HumanApprovalContext(
        action=PASSKEY_REGISTRATION_ACTION,
        tenant=str(getattr(settings, "TENANT", "") or ""),
        project=str(getattr(settings, "PROJECT", "") or ""),
        transaction_id=enrollment_id,
        request_digest=_registration_digest(
            enrollment_id=enrollment_id,
            admin=admin,
            rp_id=rp_id,
            policy=config.webauthn.credential_policy,
            return_url=final_return,
        ),
        required_assurance=FRESH_AUTHENTICATION,
        max_evidence_age_seconds=min(config.challenge_ttl_seconds, 300),
        return_url=(
            f"{PASSKEY_REGISTRATION_PATH}?" + urlencode({"enrollment": enrollment_id})
        ),
    )
    store = human_approval_store(request, config=config)
    await store.create_passkey_enrollment(
        enrollment_id=enrollment_id,
        context=context,
        admin=admin,
        final_return_url=final_return,
    )
    outcome = await evaluate_human_approval(
        request,
        context=context,
        phase="present",
    )
    if not isinstance(outcome, HumanApprovalChallenge):
        raise HumanApprovalError(
            "human_approval_enrollment_proof_invalid",
            status_code=503,
        )
    return outcome


async def registration_options(
    request: Request,
    *,
    enrollment_id: str,
) -> dict[str, Any] | HumanApprovalChallenge:
    config = human_approval_config(request)
    store = human_approval_store(request, config=config)
    admin = await resolve_browser_admin_session(request)
    enrollment = await store.passkey_enrollment(enrollment_id)
    if enrollment is None or not _same_admin(enrollment.admin, admin):
        raise HumanApprovalError(
            "human_approval_passkey_enrollment_invalid",
            status_code=403,
        )
    fresh = await evaluate_human_approval(
        request,
        context=enrollment.context,
        phase="present",
    )
    if isinstance(fresh, HumanApprovalChallenge):
        return fresh
    rp_id, origin = _rp_context(request)
    existing = await store.passkeys(subject=admin.subject, rp_id=rp_id)
    challenge_bytes = secrets.token_bytes(32)
    challenge = await store.create_passkey_challenge(
        purpose="registration",
        challenge=_b64(challenge_bytes),
        rp_id=rp_id,
        origin=origin,
        context=enrollment.context,
        admin=admin,
        credential_ids=tuple(item.credential_id for item in existing),
        enrollment_id=enrollment.enrollment_id,
    )
    library = _require_webauthn()
    policy = config.webauthn.credential_policy
    attestation = (
        library["AttestationConveyancePreference"].DIRECT
        if policy == "attested_hardware"
        else library["AttestationConveyancePreference"].NONE
    )
    options = library["generate_registration_options"](
        rp_id=rp_id,
        rp_name=config.webauthn.rp_name,
        user_name=admin.email or admin.username or admin.subject,
        user_display_name=admin.username or admin.email or admin.subject,
        user_id=hashlib.sha256(admin.subject.encode("utf-8")).digest(),
        challenge=challenge_bytes,
        timeout=config.webauthn.timeout_milliseconds,
        attestation=attestation,
        authenticator_selection=library["AuthenticatorSelectionCriteria"](
            resident_key=library["ResidentKeyRequirement"].PREFERRED,
            user_verification=library["UserVerificationRequirement"].REQUIRED,
        ),
        exclude_credentials=[
            library["PublicKeyCredentialDescriptor"](id=_unb64(item.credential_id))
            for item in existing
        ],
    )
    return {
        "state": challenge.state,
        "options": library["options_to_json_dict"](options),
    }


async def complete_registration(
    request: Request,
    *,
    state: str,
    credential_payload: Mapping[str, Any],
) -> str:
    config = human_approval_config(request)
    store = human_approval_store(request, config=config)
    admin = await resolve_browser_admin_session(request)
    challenge = await store.passkey_challenge(state)
    if (
        challenge is None
        or challenge.purpose != "registration"
        or not _same_admin(challenge.admin, admin)
    ):
        raise HumanApprovalError(
            "human_approval_passkey_challenge_invalid",
            status_code=403,
        )
    enrollment = await store.passkey_enrollment(challenge.enrollment_id)
    if enrollment is None or not _same_admin(enrollment.admin, admin):
        raise HumanApprovalError(
            "human_approval_passkey_enrollment_invalid",
            status_code=403,
        )
    library = _require_webauthn()
    verifier = getattr(
        request.app.state,
        "human_approval_webauthn_registration_verifier",
        library["verify_registration_response"],
    )
    roots = _trusted_roots(request)
    try:
        verified = verifier(
            credential=dict(credential_payload),
            expected_challenge=_unb64(challenge.challenge),
            expected_rp_id=challenge.rp_id,
            expected_origin=challenge.origin,
            require_user_presence=True,
            require_user_verification=True,
            pem_root_certs_bytes_by_fmt=roots or None,
        )
    except Exception:  # noqa: BLE001 - conceal authenticator verifier details
        raise HumanApprovalError(
            "human_approval_passkey_registration_invalid",
            status_code=403,
        ) from None
    fmt = str(getattr(verified.fmt, "value", verified.fmt))
    device_type = str(
        getattr(
            verified.credential_device_type, "value", verified.credential_device_type
        )
    )
    policy = config.webauthn.credential_policy
    if not bool(verified.user_verified):
        raise HumanApprovalError(
            "human_approval_user_verification_required",
            status_code=403,
        )
    if policy == "single_device" and device_type != "single_device":
        raise HumanApprovalError(
            "human_approval_single_device_required",
            status_code=403,
        )
    if policy == "attested_hardware":
        attestation = library["parse_attestation_object"](verified.attestation_object)
        trusted_formats = {str(getattr(item, "value", item)) for item in roots}
        if (
            fmt == "none"
            or fmt not in trusted_formats
            or not getattr(attestation.att_stmt, "x5c", None)
        ):
            raise HumanApprovalError(
                "human_approval_attested_hardware_required",
                status_code=403,
            )
    credential = PasskeyCredentialRecord(
        credential_id=_b64(verified.credential_id),
        public_key=_b64(verified.credential_public_key),
        sign_count=int(verified.sign_count),
        aaguid=str(verified.aaguid or ""),
        attestation_format=fmt,
        device_type=device_type,
        backed_up=bool(verified.credential_backed_up),
        policy=policy,
        label="Passkey",
        created_at=int(time.time()),
    )
    fresh = await evaluate_human_approval(
        request,
        context=enrollment.context,
        phase="commit",
    )
    if isinstance(fresh, HumanApprovalChallenge):
        raise HumanApprovalError(
            "human_approval_restart_required",
            status_code=409,
        )
    try:
        created = await store.add_passkey(
            subject=admin.subject,
            rp_id=challenge.rp_id,
            credential=credential,
            maximum=config.webauthn.max_credentials_per_user,
        )
    except ValueError:
        raise HumanApprovalError(
            "human_approval_passkey_credential_limit",
            status_code=409,
        ) from None
    if (
        not created
        or not await store.consume_passkey_challenge(challenge)
        or not await store.consume_passkey_enrollment(enrollment)
    ):
        raise HumanApprovalError(
            "human_approval_passkey_registration_replayed",
            status_code=409,
        )
    return enrollment.final_return_url


__all__ = [
    "PASSKEY_AUTHENTICATION_PATH",
    "PASSKEY_REGISTRATION_ACTION",
    "PASSKEY_REGISTRATION_PATH",
    "WebAuthnHumanApprovalVerifier",
    "authentication_options",
    "complete_authentication",
    "complete_registration",
    "registration_options",
    "start_enrollment",
]
