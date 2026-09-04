from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from types import SimpleNamespace

import pytest
from kdcube_ai_app.apps.chat.proc.rest.management import human_approval
from kdcube_ai_app.apps.chat.proc.rest.management.config import (
    HumanSecretExportConfig,
)
from kdcube_ai_app.apps.chat.proc.rest.management.human_approval import (
    FRESH_AUTHENTICATION,
    SESSION_CONFIRMATION,
    USER_VERIFICATION,
    BrowserSessionHumanApprovalVerifier,
    HumanApprovalChallenge,
    HumanApprovalContext,
    HumanApprovalError,
    HumanApprovalEvidence,
    assurance_satisfies,
)
from kdcube_ai_app.apps.chat.proc.rest.management.secret_export import (
    SECRET_EXPORT_REQUEST_SCHEMA,
    RedisSecretExportStore,
    SecretExportError,
    SecretExportRequest,
)
from starlette.requests import Request


def _challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(self, key, value, *, nx=False, ex=None):
        del ex
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def eval(self, _script, _keys, key, expected, _ttl, replacement):
        current = self.values.get(key)
        if current is None:
            return 0
        if current != expected:
            return -1
        self.values[key] = replacement
        return 1


def _request(*, targets=None) -> SecretExportRequest:
    verifier = "v" * 64
    return SecretExportRequest.from_mapping(
        {
            "schema": SECRET_EXPORT_REQUEST_SCHEMA,
            "callback_uri": "http://127.0.0.1:51234/callback",
            "state": "s" * 43,
            "code_challenge": _challenge(verifier),
            "code_challenge_method": "S256",
            "targets": targets
            or [
                {
                    "scope": "platform",
                    "key": "services.brave.api_key",
                },
                {
                    "scope": "bundle",
                    "bundle_id": "connection-hub@1-0",
                    "key": "connections.oauth_state_secret",
                },
            ],
        },
        tenant="tenant-a",
        project="project-a",
        max_targets=8,
    )


def _evidence(
    *,
    request: SecretExportRequest | None = None,
    verified_at: int | None = None,
) -> HumanApprovalEvidence:
    request = request or _request()
    return HumanApprovalEvidence(
        subject="user-a",
        assurance=SESSION_CONFIRMATION,
        method="browser_session",
        request_digest=request.request_digest,
        verified_at=verified_at or int(time.time()),
    )


async def _create(
    store: RedisSecretExportStore,
    request: SecretExportRequest | None = None,
    *,
    required_assurance: str = SESSION_CONFIRMATION,
    max_evidence_age_seconds: int = 300,
    max_total_value_bytes: int = 1024 * 1024,
):
    return await store.create(
        request or _request(),
        required_assurance=required_assurance,
        max_evidence_age_seconds=max_evidence_age_seconds,
        max_total_value_bytes=max_total_value_bytes,
    )


def _approval_context(
    request: SecretExportRequest | None = None,
    *,
    required_assurance: str = SESSION_CONFIRMATION,
) -> HumanApprovalContext:
    request = request or _request()
    return HumanApprovalContext(
        action="kdcube.management.secret.export",
        tenant="tenant-a",
        project="project-a",
        transaction_id="t" * 43,
        request_digest=request.request_digest,
        required_assurance=required_assurance,
        max_evidence_age_seconds=300,
        return_url="/api/integrations/management/v1/secrets/export/authorize?"
        f"transaction={'t' * 43}",
    )


def test_human_export_config_is_bounded_and_rejects_invalid_direct_values() -> None:
    values = {
        "management.secret_export.enabled": True,
        "management.secret_export.required_assurance": "fresh_authentication",
        "management.secret_export.max_evidence_age_seconds": 900,
        "management.secret_export.transaction_ttl_seconds": 900,
        "management.secret_export.consumed_tombstone_seconds": 86400,
        "management.secret_export.max_targets": 256,
        "management.secret_export.max_total_value_bytes": 8 * 1024 * 1024,
    }
    settings = SimpleNamespace(
        plain=lambda path, default=None: values.get(path, default)
    )

    config = HumanSecretExportConfig.from_settings(settings)

    assert config.max_evidence_age_seconds == 900
    assert config.transaction_ttl_seconds == 900
    assert config.consumed_tombstone_seconds == 86400
    assert config.max_targets == 256
    assert config.max_total_value_bytes == 8 * 1024 * 1024
    config.validate()

    for path, value in (
        ("management.secret_export.enabled", "true"),
        ("management.secret_export.max_evidence_age_seconds", True),
        ("management.secret_export.transaction_ttl_seconds", "180"),
    ):
        malformed = {**values, path: value}
        with pytest.raises(TypeError, match="secret export"):
            HumanSecretExportConfig.from_settings(
                SimpleNamespace(
                    plain=lambda key, default=None, source=malformed: source.get(
                        key,
                        default,
                    )
                )
            )

    oversized = {
        **values,
        "management.secret_export.max_total_value_bytes": 8 * 1024 * 1024 + 1,
    }
    oversized_config = HumanSecretExportConfig.from_settings(
        SimpleNamespace(
            plain=lambda path, default=None: oversized.get(path, default)
        )
    )
    with pytest.raises(ValueError, match="result bytes"):
        oversized_config.validate()

    invalid = HumanSecretExportConfig(
        enabled=True,
        required_assurance=SESSION_CONFIRMATION,
        max_evidence_age_seconds=0,
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=64,
        max_total_value_bytes=1024,
    )
    with pytest.raises(ValueError, match="evidence age"):
        invalid.validate()


def test_request_is_exact_sorted_and_bound_to_loopback_pkce() -> None:
    request = _request()

    assert [target.scope for target in request.targets] == ["bundle", "platform"]
    assert len(request.request_digest) == 64
    assert request.canonical_payload["callback_uri"] == (
        "http://127.0.0.1:51234/callback"
    )

    with pytest.raises(SecretExportError) as remote:
        SecretExportRequest.from_mapping(
            {
                **{
                    key: value
                    for key, value in request.canonical_payload.items()
                    if key not in {"tenant", "project"}
                },
                "callback_uri": "https://attacker.example/callback",
            },
            tenant="tenant-a",
            project="project-a",
            max_targets=8,
        )
    assert remote.value.code == "secret_export_callback_invalid"

    with pytest.raises(SecretExportError) as duplicate:
        _request(
            targets=[
                {"scope": "platform", "key": "services.brave.api_key"},
                {"scope": "platform", "key": "services.brave.api_key"},
            ]
        )
    assert duplicate.value.code == "secret_export_targets_invalid"

    non_text_state = {
        key: value
        for key, value in request.canonical_payload.items()
        if key not in {"tenant", "project"}
    }
    non_text_state["state"] = int("1" * 43)
    with pytest.raises(SecretExportError) as invalid_state:
        SecretExportRequest.from_mapping(
            non_text_state,
            tenant="tenant-a",
            project="project-a",
            max_targets=8,
        )
    assert invalid_state.value.code == "secret_export_state_invalid"


def test_assurance_order_is_explicit_and_fail_closed() -> None:
    assert assurance_satisfies(SESSION_CONFIRMATION, SESSION_CONFIRMATION)
    assert not assurance_satisfies(SESSION_CONFIRMATION, FRESH_AUTHENTICATION)
    assert assurance_satisfies(USER_VERIFICATION, FRESH_AUTHENTICATION)
    assert not assurance_satisfies("unknown", SESSION_CONFIRMATION)


def test_human_approval_accepts_platform_cookies_and_rejects_bearers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        human_approval,
        "get_settings",
        lambda: SimpleNamespace(
            AUTH=SimpleNamespace(
                AUTH_TOKEN_COOKIE_NAME="KDCubeAccess",
                ID_TOKEN_COOKIE_NAME="KDCubeIdentity",
                ID_TOKEN_HEADER_NAME="X-ID-Token",
            )
        ),
    )

    def request(*headers: tuple[bytes, bytes]) -> Request:
        return Request({"type": "http", "headers": list(headers)})

    assert human_approval._cookie_auth_only(
        request((b"cookie", b"KDCubeAccess=platform-session"))
    )
    assert human_approval._cookie_auth_only(
        request((b"cookie", b"KDCubeIdentity=id-token"))
    )
    assert not human_approval._cookie_auth_only(
        request(
            (b"cookie", b"KDCubeAccess=platform-session"),
            (b"authorization", b"Bearer delegated-card"),
        )
    )
    assert not human_approval._cookie_auth_only(
        request(
            (b"cookie", b"KDCubeIdentity=id-token"),
            (b"x-id-token", b"explicit-token"),
        )
    )


@pytest.mark.asyncio
async def test_browser_verifier_enforces_admin_session_and_fails_closed_on_step_up(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        human_approval,
        "get_settings",
        lambda: SimpleNamespace(
            AUTH=SimpleNamespace(
                AUTH_TOKEN_COOKIE_NAME="KDCubeAccess",
                ID_TOKEN_COOKIE_NAME="KDCubeIdentity",
                ID_TOKEN_HEADER_NAME="X-ID-Token",
            )
        ),
    )
    from kdcube_ai_app.apps.chat.ingress import resolvers

    calls = []

    class _Adapter:
        async def process_request(self, _request, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(user_id="user-a")

    monkeypatch.setattr(resolvers, "get_fastapi_adapter", lambda: _Adapter())
    request = Request(
        {
            "type": "http",
            "headers": [(b"cookie", b"KDCubeIdentity=id-token")],
        }
    )
    verifier = BrowserSessionHumanApprovalVerifier()

    context = _approval_context()
    evidence = await verifier.evaluate(
        request,
        context=context,
        phase="commit",
    )

    assert evidence == HumanApprovalEvidence(
        subject="user-a",
        assurance=SESSION_CONFIRMATION,
        method="kdcube_platform_browser_session",
        request_digest=context.request_digest,
        verified_at=evidence.verified_at,
    )
    assert calls[0]["connection_hub"] is False
    assert calls[0]["header_only_auth"] is False

    with pytest.raises(HumanApprovalError) as unavailable:
        await verifier.evaluate(
            request,
            context=_approval_context(
                required_assurance=FRESH_AUTHENTICATION,
            ),
            phase="commit",
        )
    assert unavailable.value.code == "human_approval_step_up_unavailable"


def test_human_approval_evidence_rejects_unbounded_or_ambiguous_fields() -> None:
    with pytest.raises(ValueError, match="subject"):
        HumanApprovalEvidence(
            subject="user\nother",
            assurance=SESSION_CONFIRMATION,
            method="browser_session",
            request_digest="d" * 64,
            verified_at=1000,
        )
    with pytest.raises(ValueError, match="method"):
        HumanApprovalEvidence(
            subject="user-a",
            assurance=SESSION_CONFIRMATION,
            method="browser session",
            request_digest="d" * 64,
            verified_at=1000,
        )

    with pytest.raises(ValueError, match="digest"):
        HumanApprovalEvidence(
            subject="user-a",
            assurance=SESSION_CONFIRMATION,
            method="browser_session",
            request_digest="not-a-digest",
            verified_at=1000,
        )

    with pytest.raises(ValueError, match="challenge URL"):
        HumanApprovalChallenge(
            authorization_url="http://attacker.example/step-up",
            method="test_step_up",
        )


@pytest.mark.asyncio
async def test_verifier_boundary_rejects_lower_or_untyped_evidence() -> None:
    class _Verifier:
        def __init__(self, evidence) -> None:
            self.evidence = evidence

        async def evaluate(self, _request, *, context, phase):
            assert context.required_assurance == FRESH_AUTHENTICATION
            assert phase == "commit"
            return self.evidence

    async def verify(evidence):
        app = SimpleNamespace(
            state=SimpleNamespace(human_approval_verifier=_Verifier(evidence))
        )
        request = Request({"type": "http", "headers": [], "app": app})
        return await human_approval.evaluate_human_approval(
            request,
            context=_approval_context(
                required_assurance=FRESH_AUTHENTICATION,
            ),
            phase="commit",
            clock=lambda: 1000,
        )

    lower = HumanApprovalEvidence(
        subject="user-a",
        assurance=SESSION_CONFIRMATION,
        method="browser_session",
        request_digest=_request().request_digest,
        verified_at=1000,
    )
    with pytest.raises(HumanApprovalError) as insufficient:
        await verify(lower)
    assert insufficient.value.code == "human_approval_step_up_unavailable"

    with pytest.raises(HumanApprovalError) as invalid:
        await verify(SimpleNamespace(assurance=USER_VERIFICATION))
    assert invalid.value.code == "human_approval_evidence_invalid"

    challenge = HumanApprovalChallenge(
        authorization_url="https://identity.example/step-up?state=opaque",
        method="test_step_up",
    )
    assert await verify(challenge) == challenge

    mismatched = HumanApprovalEvidence(
        subject="user-a",
        assurance=USER_VERIFICATION,
        method="test_user_verification",
        request_digest="e" * 64,
        verified_at=1000,
    )
    with pytest.raises(HumanApprovalError) as wrong_request:
        await verify(mismatched)
    assert wrong_request.value.code == "human_approval_evidence_invalid"

    stale = HumanApprovalEvidence(
        subject="user-a",
        assurance=USER_VERIFICATION,
        method="test_user_verification",
        request_digest=_request().request_digest,
        verified_at=699,
    )
    with pytest.raises(HumanApprovalError) as expired:
        await verify(stale)
    assert expired.value.code == "human_approval_evidence_invalid"


@pytest.mark.asyncio
async def test_verifier_boundary_rejects_unknown_phase_before_adapter() -> None:
    calls = 0

    class _Verifier:
        async def evaluate(self, _request, *, context, phase):
            nonlocal calls
            del context, phase
            calls += 1
            raise AssertionError("adapter must not receive an invalid phase")

    app = SimpleNamespace(
        state=SimpleNamespace(human_approval_verifier=_Verifier())
    )
    request = Request({"type": "http", "headers": [], "app": app})

    with pytest.raises(HumanApprovalError) as invalid:
        await human_approval.evaluate_human_approval(
            request,
            context=_approval_context(),
            phase="unknown",
        )

    assert invalid.value.code == "human_approval_phase_invalid"
    assert calls == 0


@pytest.mark.asyncio
async def test_export_code_is_pkce_bound_and_consumed_once() -> None:
    redis = _Redis()
    now = [1000]
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
        clock=lambda: now[0],
    )
    transaction = await _create(store)
    approved = await store.approve(
        transaction.transaction_id,
        csrf_token=transaction.csrf_token,
        evidence=_evidence(verified_at=1000),
    )

    with pytest.raises(SecretExportError) as wrong_verifier:
        await store.consume(
            transaction.transaction_id,
            code=approved.code,
            code_verifier="x" * 64,
        )
    assert wrong_verifier.value.code == "secret_export_code_invalid"

    consumed = await store.consume(
        transaction.transaction_id,
        code=approved.code,
        code_verifier="v" * 64,
    )
    assert consumed.subject == "user-a"
    assert consumed.request_digest == transaction.request_digest
    assert [target.provider_key for target in consumed.request.targets] == [
        "bundles.connection-hub@1-0.secrets.connections.oauth_state_secret",
        "services.brave.api_key",
    ]

    with pytest.raises(SecretExportError) as replay:
        await store.consume(
            transaction.transaction_id,
            code=approved.code,
            code_verifier="v" * 64,
        )
    assert replay.value.code == "secret_export_not_approved"
    raw_records = "\n".join(redis.values.values())
    assert approved.code not in raw_records
    assert "secret-value-canary" not in raw_records
    assert json.loads(next(iter(redis.values.values())))["status"] == "consumed"


@pytest.mark.asyncio
async def test_concurrent_export_exchange_has_one_winner() -> None:
    redis = _Redis()
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
    )
    transaction = await _create(store)
    approved = await store.approve(
        transaction.transaction_id,
        csrf_token=transaction.csrf_token,
        evidence=_evidence(),
    )

    outcomes = await asyncio.gather(
        store.consume(
            transaction.transaction_id,
            code=approved.code,
            code_verifier="v" * 64,
        ),
        store.consume(
            transaction.transaction_id,
            code=approved.code,
            code_verifier="v" * 64,
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    failures = [item for item in outcomes if isinstance(item, SecretExportError)]
    assert len(failures) == 1
    assert failures[0].code in {
        "secret_export_not_approved",
        "secret_export_transaction_moved",
    }


@pytest.mark.asyncio
async def test_invalid_csrf_does_not_move_pending_transaction() -> None:
    redis = _Redis()
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
    )
    transaction = await _create(store)

    with pytest.raises(SecretExportError) as invalid:
        await store.approve(
            transaction.transaction_id,
            csrf_token="x" * 43,
            evidence=_evidence(),
        )
    assert invalid.value.code == "secret_export_csrf_invalid"
    assert (await store.load(transaction.transaction_id)).status == "pending"


@pytest.mark.asyncio
async def test_approval_evidence_is_bound_to_transaction_and_creation_time() -> None:
    redis = _Redis()
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
        clock=lambda: 1000,
    )
    request = _request()
    transaction = await _create(store, request)

    wrong_request = HumanApprovalEvidence(
        subject="user-a",
        assurance=SESSION_CONFIRMATION,
        method="browser_session",
        request_digest="e" * 64,
        verified_at=1000,
    )
    with pytest.raises(SecretExportError) as mismatched:
        await store.approve(
            transaction.transaction_id,
            csrf_token=transaction.csrf_token,
            evidence=wrong_request,
        )
    assert mismatched.value.code == "secret_export_approval_invalid"

    before_transaction = _evidence(request=request, verified_at=999)
    with pytest.raises(SecretExportError) as stale:
        await store.approve(
            transaction.transaction_id,
            csrf_token=transaction.csrf_token,
            evidence=before_transaction,
        )
    assert stale.value.code == "secret_export_approval_invalid"
    assert (await store.load(transaction.transaction_id)).status == "pending"


@pytest.mark.asyncio
async def test_transaction_pins_assurance_evidence_age_and_result_bound() -> None:
    redis = _Redis()
    now = [900]
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
        clock=lambda: now[0],
    )
    request = _request()
    transaction = await _create(
        store,
        request,
        required_assurance=FRESH_AUTHENTICATION,
        max_evidence_age_seconds=10,
        max_total_value_bytes=321,
    )

    assert transaction.required_assurance == FRESH_AUTHENTICATION
    assert transaction.max_evidence_age_seconds == 10
    assert transaction.max_total_value_bytes == 321

    now[0] = 1000
    for evidence in (
        HumanApprovalEvidence(
            subject="user-a",
            assurance=SESSION_CONFIRMATION,
            method="browser_session",
            request_digest=request.request_digest,
            verified_at=1000,
        ),
        HumanApprovalEvidence(
            subject="user-a",
            assurance=FRESH_AUTHENTICATION,
            method="fresh_authentication",
            request_digest=request.request_digest,
            verified_at=989,
        ),
    ):
        with pytest.raises(SecretExportError) as invalid:
            await store.approve(
                transaction.transaction_id,
                csrf_token=transaction.csrf_token,
                evidence=evidence,
            )
        assert invalid.value.code == "secret_export_approval_invalid"

    approved = await store.approve(
        transaction.transaction_id,
        csrf_token=transaction.csrf_token,
        evidence=HumanApprovalEvidence(
            subject="user-a",
            assurance=USER_VERIFICATION,
            method="webauthn_user_verification",
            request_digest=request.request_digest,
            verified_at=995,
        ),
    )
    consumed = await store.consume(
        transaction.transaction_id,
        code=approved.code,
        code_verifier="v" * 64,
    )
    assert consumed.max_total_value_bytes == 321


@pytest.mark.asyncio
async def test_stored_transaction_rejects_duplicate_json_fields() -> None:
    redis = _Redis()
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
    )
    transaction = await _create(store)
    key = next(iter(redis.values))
    redis.values[key] = redis.values[key].replace(
        '{"approval_method":',
        '{"approval_method":"duplicate","approval_method":',
        1,
    )

    with pytest.raises(SecretExportError) as invalid:
        await store.load(transaction.transaction_id)

    assert invalid.value.code == "secret_export_transaction_invalid"


@pytest.mark.asyncio
async def test_stored_transaction_cannot_move_to_another_deployment() -> None:
    redis = _Redis()
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
    )
    transaction = await _create(store)
    key = next(iter(redis.values))
    record = json.loads(redis.values[key])
    record["request"]["tenant"] = "tenant-b"
    record["request_digest"] = hashlib.sha256(
        json.dumps(
            record["request"],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    redis.values[key] = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(SecretExportError) as moved:
        await store.load(transaction.transaction_id)

    assert moved.value.code == "secret_export_transaction_invalid"


@pytest.mark.asyncio
async def test_stored_approval_evidence_rejects_log_injection() -> None:
    redis = _Redis()
    store = RedisSecretExportStore(
        redis,
        tenant="tenant-a",
        project="project-a",
        transaction_ttl_seconds=180,
        consumed_tombstone_seconds=600,
        max_targets=8,
    )
    transaction = await _create(store)
    approved = await store.approve(
        transaction.transaction_id,
        csrf_token=transaction.csrf_token,
        evidence=_evidence(),
    )
    del approved
    key = next(iter(redis.values))
    record = json.loads(redis.values[key])
    record["subject"] = "user-a\ninjected-log-line"
    redis.values[key] = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(SecretExportError) as invalid:
        await store.load(transaction.transaction_id)

    assert invalid.value.code == "secret_export_transaction_invalid"
