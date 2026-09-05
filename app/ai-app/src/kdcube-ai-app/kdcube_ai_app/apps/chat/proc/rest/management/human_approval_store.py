# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Distributed one-use state for stronger human-approval adapters."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from kdcube_ai_app.apps.chat.proc.rest.management.human_approval import (
    BrowserAdminSession,
    HumanApprovalContext,
    HumanApprovalEvidence,
)

OIDC_CHALLENGE_SCHEMA = "kdcube.human_approval.oidc_challenge.v1"
HUMAN_PROOF_SCHEMA = "kdcube.human_approval.proof.v1"
PASSKEY_CREDENTIALS_SCHEMA = "kdcube.human_approval.passkeys.v1"
PASSKEY_CHALLENGE_SCHEMA = "kdcube.human_approval.passkey_challenge.v1"
PASSKEY_ENROLLMENT_SCHEMA = "kdcube.human_approval.passkey_enrollment.v1"


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _decode_json(raw: Any, *, schema: str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(raw)
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        raise RuntimeError("human approval state is invalid")
    return dict(value)


def _context_dict(context: HumanApprovalContext) -> dict[str, Any]:
    return asdict(context)


def _context_from(value: Any) -> HumanApprovalContext:
    if not isinstance(value, Mapping):
        raise TypeError("human approval context is invalid")
    return HumanApprovalContext(**dict(value))


def _binding_matches(record: Mapping[str, Any], admin: BrowserAdminSession) -> bool:
    return all(
        (
            secrets.compare_digest(str(record.get("subject") or ""), admin.subject),
            secrets.compare_digest(
                str(record.get("session_id") or ""), admin.session_id
            ),
            secrets.compare_digest(
                str(record.get("cookie_binding") or ""), admin.cookie_binding
            ),
        )
    )


def _context_matches(
    record: Mapping[str, Any],
    context: HumanApprovalContext,
) -> bool:
    try:
        stored = _context_from(record.get("context"))
    except (TypeError, ValueError):
        return False
    return secrets.compare_digest(
        _json(_context_dict(stored)),
        _json(_context_dict(context)),
    )


@dataclass(frozen=True)
class OidcChallengeRecord:
    state: str
    provider: str
    provider_alias: str
    nonce: str
    code_verifier: str
    redirect_uri: str
    context: HumanApprovalContext
    admin: BrowserAdminSession
    created_at: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OIDC_CHALLENGE_SCHEMA,
            "provider": self.provider,
            "provider_alias": self.provider_alias,
            "nonce": self.nonce,
            "code_verifier": self.code_verifier,
            "redirect_uri": self.redirect_uri,
            "context": _context_dict(self.context),
            "subject": self.admin.subject,
            "session_id": self.admin.session_id,
            "cookie_binding": self.admin.cookie_binding,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class HumanProofRecord:
    evidence: HumanApprovalEvidence
    context: HumanApprovalContext
    admin: BrowserAdminSession

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HUMAN_PROOF_SCHEMA,
            "evidence": asdict(self.evidence),
            "context": _context_dict(self.context),
            "subject": self.admin.subject,
            "session_id": self.admin.session_id,
            "cookie_binding": self.admin.cookie_binding,
        }


@dataclass(frozen=True)
class PasskeyCredentialRecord:
    credential_id: str
    public_key: str
    sign_count: int
    aaguid: str
    attestation_format: str
    device_type: str
    backed_up: bool
    policy: str
    label: str
    created_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PasskeyCredentialRecord:
        record = cls(
            credential_id=str(value.get("credential_id") or "").strip(),
            public_key=str(value.get("public_key") or "").strip(),
            sign_count=int(value.get("sign_count") or 0),
            aaguid=str(value.get("aaguid") or "").strip(),
            attestation_format=str(value.get("attestation_format") or "").strip(),
            device_type=str(value.get("device_type") or "").strip(),
            backed_up=bool(value.get("backed_up")),
            policy=str(value.get("policy") or "").strip(),
            label=str(value.get("label") or "").strip(),
            created_at=int(value.get("created_at") or 0),
        )
        if (
            not record.credential_id
            or not record.public_key
            or record.sign_count < 0
            or record.policy
            not in {"verified_passkey", "single_device", "attested_hardware"}
            or record.created_at <= 0
        ):
            raise ValueError("passkey credential record is invalid")
        return record


@dataclass(frozen=True)
class PasskeyChallengeRecord:
    state: str
    purpose: str
    challenge: str
    rp_id: str
    origin: str
    context: HumanApprovalContext
    admin: BrowserAdminSession
    credential_ids: tuple[str, ...]
    enrollment_id: str
    created_at: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PASSKEY_CHALLENGE_SCHEMA,
            "purpose": self.purpose,
            "challenge": self.challenge,
            "rp_id": self.rp_id,
            "origin": self.origin,
            "context": _context_dict(self.context),
            "subject": self.admin.subject,
            "session_id": self.admin.session_id,
            "cookie_binding": self.admin.cookie_binding,
            "credential_ids": list(self.credential_ids),
            "enrollment_id": self.enrollment_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class PasskeyEnrollmentRecord:
    enrollment_id: str
    context: HumanApprovalContext
    admin: BrowserAdminSession
    final_return_url: str
    created_at: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PASSKEY_ENROLLMENT_SCHEMA,
            "context": _context_dict(self.context),
            "subject": self.admin.subject,
            "session_id": self.admin.session_id,
            "cookie_binding": self.admin.cookie_binding,
            "final_return_url": self.final_return_url,
            "created_at": self.created_at,
        }


class RedisHumanApprovalStore:
    """Redis-backed challenge and proof state shared by processor workers."""

    _CONSUME_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
if raw ~= ARGV[1] then return -1 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 1
"""
    _CAS_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if ARGV[1] == '__missing__' then
  if raw then return -1 end
else
  if not raw or raw ~= ARGV[1] then return -1 end
end
redis.call('SET', KEYS[1], ARGV[2])
return 1
"""

    def __init__(
        self,
        redis: Any,
        *,
        tenant: str,
        project: str,
        ttl_seconds: int,
        clock: Any = time.time,
    ) -> None:
        self._redis = redis
        self._tenant = str(tenant or "").strip()
        self._project = str(project or "").strip()
        self._ttl_seconds = int(ttl_seconds)
        self._clock = clock
        if not self._tenant or not self._project:
            raise ValueError("human approval store coordinates are required")
        if not 30 <= self._ttl_seconds <= 900:
            raise ValueError("human approval store TTL is invalid")

    def _key(self, kind: str, token: str) -> str:
        digest = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
        return (
            "kdcube:management:human-approval:"
            f"{self._tenant}:{self._project}:{kind}:{digest}"
        )

    def _proof_token(
        self,
        *,
        context: HumanApprovalContext,
        admin: BrowserAdminSession,
    ) -> str:
        return _json(
            {
                "action": context.action,
                "transaction_id": context.transaction_id,
                "request_digest": context.request_digest,
                "subject": admin.subject,
                "session_id": admin.session_id,
                "cookie_binding": admin.cookie_binding,
            }
        )

    async def create_oidc_challenge(
        self,
        *,
        provider: str,
        provider_alias: str,
        nonce: str,
        code_verifier: str,
        redirect_uri: str,
        context: HumanApprovalContext,
        admin: BrowserAdminSession,
    ) -> OidcChallengeRecord:
        state = secrets.token_urlsafe(32)
        record = OidcChallengeRecord(
            state=state,
            provider=str(provider or "").strip(),
            provider_alias=str(provider_alias or "").strip(),
            nonce=str(nonce or "").strip(),
            code_verifier=str(code_verifier or "").strip(),
            redirect_uri=str(redirect_uri or "").strip(),
            context=context,
            admin=admin,
            created_at=int(self._clock()),
        )
        if record.provider not in {"cognito", "google"}:
            raise ValueError("human approval OIDC provider is invalid")
        if len(record.nonce) < 32 or len(record.code_verifier) > 256:
            raise ValueError("human approval OIDC challenge is invalid")
        created = await self._redis.set(
            self._key("oidc", state),
            _json(record.to_dict()),
            nx=True,
            ex=self._ttl_seconds,
        )
        if not created:
            raise RuntimeError("human approval challenge collision")
        return record

    async def oidc_challenge(self, state: str) -> OidcChallengeRecord | None:
        raw = await self._redis.get(self._key("oidc", state))
        if raw is None:
            return None
        value = _decode_json(raw, schema=OIDC_CHALLENGE_SCHEMA)
        if value.get("consumed") is True:
            return None
        context = _context_from(value.get("context"))
        admin = BrowserAdminSession(
            subject=value.get("subject"),
            session_id=value.get("session_id"),
            cookie_binding=value.get("cookie_binding"),
        )
        return OidcChallengeRecord(
            state=state,
            provider=str(value.get("provider") or ""),
            provider_alias=str(value.get("provider_alias") or ""),
            nonce=str(value.get("nonce") or ""),
            code_verifier=str(value.get("code_verifier") or ""),
            redirect_uri=str(value.get("redirect_uri") or ""),
            context=context,
            admin=admin,
            created_at=int(value.get("created_at") or 0),
        )

    async def consume_oidc_challenge(self, record: OidcChallengeRecord) -> bool:
        raw = _json(record.to_dict())
        tombstone = _json(
            {
                "schema": OIDC_CHALLENGE_SCHEMA,
                "consumed": True,
                "consumed_at": int(self._clock()),
            }
        )
        result = await self._redis.eval(
            self._CONSUME_SCRIPT,
            1,
            self._key("oidc", record.state),
            raw,
            tombstone,
            str(self._ttl_seconds),
        )
        return int(result or 0) == 1

    async def put_proof(self, record: HumanProofRecord) -> bool:
        key = self._key(
            "proof",
            self._proof_token(context=record.context, admin=record.admin),
        )
        return bool(
            await self._redis.set(
                key,
                _json(record.to_dict()),
                nx=True,
                ex=self._ttl_seconds,
            )
        )

    async def proof(
        self,
        *,
        context: HumanApprovalContext,
        admin: BrowserAdminSession,
        consume: bool,
    ) -> HumanApprovalEvidence | None:
        key = self._key(
            "proof",
            self._proof_token(context=context, admin=admin),
        )
        raw = await self._redis.get(key)
        if raw is None:
            return None
        value = _decode_json(raw, schema=HUMAN_PROOF_SCHEMA)
        if value.get("consumed") is True:
            return None
        if not _binding_matches(value, admin) or not _context_matches(value, context):
            raise RuntimeError("human approval proof binding is invalid")
        evidence = HumanApprovalEvidence(**dict(value.get("evidence") or {}))
        if not consume:
            return evidence
        tombstone = _json(
            {
                "schema": HUMAN_PROOF_SCHEMA,
                "consumed": True,
                "consumed_at": int(self._clock()),
            }
        )
        result = await self._redis.eval(
            self._CONSUME_SCRIPT,
            1,
            key,
            raw.decode("utf-8") if isinstance(raw, bytes) else raw,
            tombstone,
            str(self._ttl_seconds),
        )
        if int(result or 0) != 1:
            return None
        return evidence

    def _credential_key(self, *, subject: str, rp_id: str) -> str:
        return self._key("passkeys", f"{subject}\n{rp_id}")

    async def passkeys(
        self,
        *,
        subject: str,
        rp_id: str,
    ) -> tuple[PasskeyCredentialRecord, ...]:
        raw = await self._redis.get(self._credential_key(subject=subject, rp_id=rp_id))
        if raw is None:
            return ()
        value = _decode_json(raw, schema=PASSKEY_CREDENTIALS_SCHEMA)
        if value.get("subject") != subject or value.get("rp_id") != rp_id:
            raise RuntimeError("passkey credential binding is invalid")
        records = value.get("credentials")
        if not isinstance(records, list):
            raise TypeError("passkey credential set is invalid")
        return tuple(
            PasskeyCredentialRecord.from_mapping(item)
            for item in records
            if isinstance(item, Mapping)
        )

    async def add_passkey(
        self,
        *,
        subject: str,
        rp_id: str,
        credential: PasskeyCredentialRecord,
        maximum: int,
    ) -> bool:
        key = self._credential_key(subject=subject, rp_id=rp_id)
        for _attempt in range(4):
            raw = await self._redis.get(key)
            if raw is None:
                current: list[PasskeyCredentialRecord] = []
                expected = "__missing__"
            else:
                value = _decode_json(raw, schema=PASSKEY_CREDENTIALS_SCHEMA)
                if value.get("subject") != subject or value.get("rp_id") != rp_id:
                    raise RuntimeError("passkey credential binding is invalid")
                current = [
                    PasskeyCredentialRecord.from_mapping(item)
                    for item in value.get("credentials") or []
                    if isinstance(item, Mapping)
                ]
                expected = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if any(
                secrets.compare_digest(item.credential_id, credential.credential_id)
                for item in current
            ):
                return False
            if len(current) >= maximum:
                raise ValueError("passkey credential limit reached")
            replacement = _json(
                {
                    "schema": PASSKEY_CREDENTIALS_SCHEMA,
                    "subject": subject,
                    "rp_id": rp_id,
                    "credentials": [item.to_dict() for item in (*current, credential)],
                }
            )
            changed = await self._redis.eval(
                self._CAS_SCRIPT,
                1,
                key,
                expected,
                replacement,
            )
            if int(changed or 0) == 1:
                return True
        raise RuntimeError("passkey credential set changed concurrently")

    async def update_passkey_counter(
        self,
        *,
        subject: str,
        rp_id: str,
        credential_id: str,
        expected_count: int,
        new_count: int,
        device_type: str,
        backed_up: bool,
    ) -> bool:
        key = self._credential_key(subject=subject, rp_id=rp_id)
        raw = await self._redis.get(key)
        if raw is None:
            return False
        value = _decode_json(raw, schema=PASSKEY_CREDENTIALS_SCHEMA)
        records = [
            PasskeyCredentialRecord.from_mapping(item)
            for item in value.get("credentials") or []
            if isinstance(item, Mapping)
        ]
        changed = False
        replacement_records: list[PasskeyCredentialRecord] = []
        for record in records:
            if secrets.compare_digest(record.credential_id, credential_id):
                if record.sign_count != expected_count or new_count < expected_count:
                    return False
                record = PasskeyCredentialRecord(
                    **{
                        **record.to_dict(),
                        "sign_count": new_count,
                        "device_type": device_type,
                        "backed_up": backed_up,
                    }
                )
                changed = True
            replacement_records.append(record)
        if not changed:
            return False
        replacement = _json(
            {
                "schema": PASSKEY_CREDENTIALS_SCHEMA,
                "subject": subject,
                "rp_id": rp_id,
                "credentials": [item.to_dict() for item in replacement_records],
            }
        )
        expected = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        result = await self._redis.eval(
            self._CAS_SCRIPT,
            1,
            key,
            expected,
            replacement,
        )
        return int(result or 0) == 1

    async def create_passkey_challenge(
        self,
        *,
        purpose: str,
        challenge: str,
        rp_id: str,
        origin: str,
        context: HumanApprovalContext,
        admin: BrowserAdminSession,
        credential_ids: tuple[str, ...] = (),
        enrollment_id: str = "",
    ) -> PasskeyChallengeRecord:
        if purpose not in {"authentication", "registration"}:
            raise ValueError("passkey challenge purpose is invalid")
        state = secrets.token_urlsafe(32)
        record = PasskeyChallengeRecord(
            state=state,
            purpose=purpose,
            challenge=challenge,
            rp_id=rp_id,
            origin=origin,
            context=context,
            admin=admin,
            credential_ids=tuple(credential_ids),
            enrollment_id=enrollment_id,
            created_at=int(self._clock()),
        )
        created = await self._redis.set(
            self._key("passkey-challenge", state),
            _json(record.to_dict()),
            nx=True,
            ex=self._ttl_seconds,
        )
        if not created:
            raise RuntimeError("passkey challenge collision")
        return record

    async def passkey_challenge(
        self,
        state: str,
    ) -> PasskeyChallengeRecord | None:
        raw = await self._redis.get(self._key("passkey-challenge", state))
        if raw is None:
            return None
        value = _decode_json(raw, schema=PASSKEY_CHALLENGE_SCHEMA)
        if value.get("consumed") is True:
            return None
        return PasskeyChallengeRecord(
            state=state,
            purpose=str(value.get("purpose") or ""),
            challenge=str(value.get("challenge") or ""),
            rp_id=str(value.get("rp_id") or ""),
            origin=str(value.get("origin") or ""),
            context=_context_from(value.get("context")),
            admin=BrowserAdminSession(
                subject=value.get("subject"),
                session_id=value.get("session_id"),
                cookie_binding=value.get("cookie_binding"),
            ),
            credential_ids=tuple(value.get("credential_ids") or []),
            enrollment_id=str(value.get("enrollment_id") or ""),
            created_at=int(value.get("created_at") or 0),
        )

    async def consume_passkey_challenge(
        self,
        record: PasskeyChallengeRecord,
    ) -> bool:
        return await self._consume_record(
            key=self._key("passkey-challenge", record.state),
            raw=_json(record.to_dict()),
            schema=PASSKEY_CHALLENGE_SCHEMA,
        )

    async def create_passkey_enrollment(
        self,
        *,
        enrollment_id: str,
        context: HumanApprovalContext,
        admin: BrowserAdminSession,
        final_return_url: str,
    ) -> PasskeyEnrollmentRecord:
        record = PasskeyEnrollmentRecord(
            enrollment_id=enrollment_id,
            context=context,
            admin=admin,
            final_return_url=final_return_url,
            created_at=int(self._clock()),
        )
        created = await self._redis.set(
            self._key("passkey-enrollment", record.enrollment_id),
            _json(record.to_dict()),
            nx=True,
            ex=self._ttl_seconds,
        )
        if not created:
            raise RuntimeError("passkey enrollment collision")
        return record

    async def passkey_enrollment(
        self,
        enrollment_id: str,
    ) -> PasskeyEnrollmentRecord | None:
        raw = await self._redis.get(self._key("passkey-enrollment", enrollment_id))
        if raw is None:
            return None
        value = _decode_json(raw, schema=PASSKEY_ENROLLMENT_SCHEMA)
        if value.get("consumed") is True:
            return None
        return PasskeyEnrollmentRecord(
            enrollment_id=enrollment_id,
            context=_context_from(value.get("context")),
            admin=BrowserAdminSession(
                subject=value.get("subject"),
                session_id=value.get("session_id"),
                cookie_binding=value.get("cookie_binding"),
            ),
            final_return_url=str(value.get("final_return_url") or ""),
            created_at=int(value.get("created_at") or 0),
        )

    async def consume_passkey_enrollment(
        self,
        record: PasskeyEnrollmentRecord,
    ) -> bool:
        return await self._consume_record(
            key=self._key("passkey-enrollment", record.enrollment_id),
            raw=_json(record.to_dict()),
            schema=PASSKEY_ENROLLMENT_SCHEMA,
        )

    async def _consume_record(self, *, key: str, raw: str, schema: str) -> bool:
        result = await self._redis.eval(
            self._CONSUME_SCRIPT,
            1,
            key,
            raw,
            _json(
                {
                    "schema": schema,
                    "consumed": True,
                    "consumed_at": int(self._clock()),
                }
            ),
            str(self._ttl_seconds),
        )
        return int(result or 0) == 1


__all__ = [
    "HUMAN_PROOF_SCHEMA",
    "OIDC_CHALLENGE_SCHEMA",
    "PASSKEY_CHALLENGE_SCHEMA",
    "PASSKEY_CREDENTIALS_SCHEMA",
    "PASSKEY_ENROLLMENT_SCHEMA",
    "HumanProofRecord",
    "OidcChallengeRecord",
    "PasskeyChallengeRecord",
    "PasskeyCredentialRecord",
    "PasskeyEnrollmentRecord",
    "RedisHumanApprovalStore",
]
