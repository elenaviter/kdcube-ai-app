from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from kdcube_ai_app.apps.chat.proc.rest.management.admission import AdmissionDecision
from kdcube_ai_app.apps.chat.proc.rest.management.admission import AdmissionUnavailable
from kdcube_ai_app.apps.chat.proc.rest.management.contracts import (
    INSPECT_OPERATION,
    RELOAD_OPERATION,
    SURFACES_OPERATION,
)
from kdcube_ai_app.apps.chat.proc.rest.management.effect_ledger import (
    ACTION_CONFLICT,
    ACTION_EXECUTE,
    ACTION_PENDING,
    ACTION_REPLAY,
    EffectReservation,
)
from kdcube_ai_app.apps.chat.proc.rest.management.service import (
    DelegatedManagementService,
    ManagementApplicationNotFound,
)


def _allow() -> AdmissionDecision:
    return AdmissionDecision(
        200,
        {
            "allowed": True,
            "decision_id": "decision-1",
            "principal": {"client_id": "caller-profile-1"},
            "provenance": {
                "access_id": "access-1",
                "card_revision": 4,
                "card_catalog_version": "catalog-3",
                "active_catalog_version": "catalog-3",
                "invocation_policy_revision": 2,
                "request_permit_revision": 1,
            },
        },
    )


class _Admission:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.revoked = False
        self.inspect_only = False
        self.request_permit_required = False
        self.unavailable = False
        self.unexpected_failure = False

    async def evaluate(self, **kwargs: Any) -> AdmissionDecision:
        self.calls.append(kwargs)
        if self.unavailable:
            raise AdmissionUnavailable("admission unavailable")
        if self.unexpected_failure:
            raise RuntimeError("unexpected admission failure")
        if self.revoked or (
            self.inspect_only and kwargs["operation"] != INSPECT_OPERATION
        ):
            return AdmissionDecision(
                403,
                {
                    "allowed": False,
                    "error": {
                        "code": "delegated_operation_not_granted",
                        "message": "The live card does not grant this operation.",
                        "retryable": False,
                    },
                },
            )
        if self.request_permit_required:
            return AdmissionDecision(
                403,
                {
                    "allowed": False,
                    "error": {
                        "code": "delegated_invocation_policy_required",
                        "message": "Approve this exact request.",
                        "retryable": False,
                    },
                    "consent": {
                        "kind": "delegated_request_permit",
                        "reason": "delegated_invocation_policy_required",
                        "connection_hub_url": "https://runtime.example/approve?request=1",
                        "access_id": "access-1",
                        "resource": kwargs["resource"],
                        "outer_operation": kwargs["operation"],
                        "invocation_id": kwargs["invocation_id"],
                        "request_digest": kwargs["request_digest"],
                        "card_revision": 4,
                        "catalog_version": "catalog-3",
                        "expires_at": 1_788_380_000,
                        "approval_context": dict(kwargs["approval_context"]),
                        "available_choices": ["allow_once", "allow_always"],
                    },
                },
            )
        return _allow()


class _Ledger:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def reserve(self, **kwargs: Any) -> EffectReservation:
        key = (
            kwargs["access_id"],
            kwargs["operation"],
            kwargs["invocation_id"],
        )
        async with self.lock:
            current = self.records.get(key)
            if current is None:
                current = {
                    "request_digest": kwargs["request_digest"],
                    "state": "effect_started",
                    "audit": dict(kwargs["audit"]),
                }
                self.records[key] = current
                return EffectReservation(ACTION_EXECUTE, owner="owner-1", record=current)
            if current["request_digest"] != kwargs["request_digest"]:
                return EffectReservation(ACTION_CONFLICT, record=current)
            if current["state"] in {"effect_completed", "effect_failed"}:
                return EffectReservation(ACTION_REPLAY, record=current)
            return EffectReservation(ACTION_PENDING, record=current)

    async def finish(self, **kwargs: Any) -> None:
        key = (
            kwargs["access_id"],
            kwargs["operation"],
            kwargs["invocation_id"],
        )
        async with self.lock:
            self.records[key] = {
                **self.records[key],
                "state": "effect_failed" if kwargs["failed"] else "effect_completed",
                "status_code": kwargs["status_code"],
                "response": dict(kwargs["response"]),
            }


@dataclass
class _Runtime:
    reload_calls: int = 0
    inspect_calls: int = 0
    surfaces_calls: int = 0
    block_reload: asyncio.Event | None = None
    fail_reload: bool = False

    async def inspect_deployment(self) -> Mapping[str, Any]:
        self.inspect_calls += 1
        return {"readiness": "ready", "platform_release": "release-1", "applications": []}

    async def application_surfaces(self, application_id: str) -> Mapping[str, Any]:
        self.surfaces_calls += 1
        if application_id != "app-a@1-0":
            raise ManagementApplicationNotFound(application_id)
        return {"application_id": application_id, "surfaces": {}}

    async def reload_application(
        self, application_id: str, *, caller_profile: str
    ) -> Mapping[str, Any]:
        assert caller_profile == "caller-profile-1"
        if application_id != "app-a@1-0":
            raise ManagementApplicationNotFound(application_id)
        self.reload_calls += 1
        if self.fail_reload:
            raise RuntimeError("reload fixture failed")
        if self.block_reload is not None:
            await self.block_reload.wait()
        return {
            "application_id": application_id,
            "state": "completed",
            "changed_application_ids": [application_id],
            "generation": "generation-1",
        }


def _service(
    admission: _Admission, ledger: _Ledger, runtime: _Runtime
) -> DelegatedManagementService:
    return DelegatedManagementService(
        tenant="tenant-a",
        project="project-a",
        resource="urn:kdcube:management:deployment:tenant-a:project-a",
        runtime_instance="proc-1",
        admission=admission,
        ledger=ledger,
        runtime=runtime,
    )


@pytest.mark.asyncio
async def test_inspect_only_card_cannot_reload() -> None:
    admission = _Admission()
    admission.inspect_only = True
    runtime = _Runtime()
    service = _service(admission, _Ledger(), runtime)

    inspect = await service.execute(
        operation=INSPECT_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="inspect-1",
    )
    reload_result = await service.execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )

    assert inspect.status_code == 200
    assert reload_result.status_code == 403
    assert runtime.inspect_calls == 1
    assert runtime.reload_calls == 0


@pytest.mark.asyncio
async def test_request_bound_denial_preserves_exact_browser_recovery() -> None:
    admission = _Admission()
    admission.request_permit_required = True
    service = _service(admission, _Ledger(), _Runtime())

    response = await service.execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )

    assert response.status_code == 403
    recovery = response.payload["recovery"]
    assert recovery["application_id"] == "app-a@1-0"
    assert recovery["invocation_id"] == "reload-1"
    assert recovery["request_digest"] == admission.calls[0]["request_digest"]
    assert recovery["reason"] == "delegated_request_permit_required"
    assert recovery["expires_at"] == 1_788_380_000
    assert "permit_ttl_seconds" not in recovery
    assert recovery["choices"] == ["allow_once", "allow_always"]


@pytest.mark.asyncio
async def test_non_request_bound_denial_has_no_browser_recovery() -> None:
    class _OrdinaryConsentAdmission(_Admission):
        async def evaluate(self, **kwargs: Any) -> AdmissionDecision:
            self.calls.append(kwargs)
            return AdmissionDecision(
                403,
                {
                    "allowed": False,
                    "error": {
                        "code": "delegated_operation_not_granted",
                        "message": "The live card does not grant this operation.",
                        "retryable": False,
                    },
                    "consent": {
                        "kind": "delegated_agent_grant",
                        "connection_hub_url": "https://runtime.example/approve?request=1",
                    },
                },
            )

    response = await _service(
        _OrdinaryConsentAdmission(), _Ledger(), _Runtime()
    ).execute(
        operation=INSPECT_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="inspect-1",
    )

    assert response.status_code == 403
    assert "recovery" not in response.payload


@pytest.mark.asyncio
async def test_same_invocation_replays_without_second_effect() -> None:
    admission = _Admission()
    runtime = _Runtime()
    service = _service(admission, _Ledger(), runtime)

    first = await service.execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )
    replay = await service.execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.payload["invocation"]["replay"] is True
    assert runtime.reload_calls == 1
    assert len(admission.calls) == 2


@pytest.mark.asyncio
async def test_allowed_admission_requires_card_and_caller_identity() -> None:
    class _IncompleteAdmission(_Admission):
        async def evaluate(self, **kwargs: Any) -> AdmissionDecision:
            self.calls.append(kwargs)
            return AdmissionDecision(200, {"allowed": True, "decision_id": "decision-1"})

    ledger = _Ledger()
    runtime = _Runtime()
    response = await _service(_IncompleteAdmission(), ledger, runtime).execute(
        operation=INSPECT_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="inspect-1",
    )

    assert response.status_code == 503
    assert response.payload["error"]["code"] == "delegated_admission_invalid"
    assert ledger.records == {}
    assert runtime.inspect_calls == 0


@pytest.mark.asyncio
async def test_concurrent_duplicate_performs_one_effect() -> None:
    admission = _Admission()
    release = asyncio.Event()
    runtime = _Runtime(block_reload=release)
    service = _service(admission, _Ledger(), runtime)

    first_task = asyncio.create_task(
        service.execute(
            operation=RELOAD_OPERATION,
            delegated_bearer="opaque-bearer",
            invocation_id="reload-1",
            application_id="app-a@1-0",
        )
    )
    while runtime.reload_calls == 0:
        await asyncio.sleep(0)
    duplicate = await service.execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )
    release.set()
    first = await first_task

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.payload["error"]["code"] == "effect_outcome_pending"
    assert runtime.reload_calls == 1


@pytest.mark.asyncio
async def test_changed_application_conflicts_and_revocation_applies_next_call() -> None:
    admission = _Admission()
    ledger = _Ledger()
    runtime = _Runtime()
    service = _service(admission, ledger, runtime)

    first = await service.execute(
        operation=SURFACES_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="surfaces-1",
        application_id="app-a@1-0",
    )
    conflict = await service.execute(
        operation=SURFACES_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="surfaces-1",
        application_id="app-b@1-0",
    )
    admission.revoked = True
    revoked = await service.execute(
        operation=SURFACES_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="surfaces-2",
        application_id="app-a@1-0",
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.payload["error"]["code"] == "invocation_id_conflict"
    assert revoked.status_code == 403
    assert runtime.surfaces_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unexpected", [False, True])
async def test_admission_failure_stops_before_ledger_and_runtime(
    unexpected: bool,
) -> None:
    admission = _Admission()
    admission.unavailable = not unexpected
    admission.unexpected_failure = unexpected
    ledger = _Ledger()
    runtime = _Runtime()

    response = await _service(admission, ledger, runtime).execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )

    assert response.status_code == 503
    assert response.payload["error"]["code"] == "delegated_admission_unavailable"
    assert response.payload["error"]["retryable"] is True
    assert ledger.records == {}
    assert runtime.reload_calls == 0


@pytest.mark.asyncio
async def test_ledger_failure_stops_before_runtime() -> None:
    class _UnavailableLedger(_Ledger):
        async def reserve(self, **kwargs: Any) -> EffectReservation:
            del kwargs
            raise RuntimeError("ledger unavailable")

    runtime = _Runtime()
    response = await _service(_Admission(), _UnavailableLedger(), runtime).execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )

    assert response.status_code == 503
    assert response.payload["error"]["code"] == "effect_ledger_unavailable"
    assert response.payload["error"]["retryable"] is True
    assert runtime.reload_calls == 0


@pytest.mark.asyncio
async def test_runtime_failure_is_terminal_for_the_same_invocation() -> None:
    runtime = _Runtime(fail_reload=True)
    service = _service(_Admission(), _Ledger(), runtime)

    first = await service.execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )
    replay = await service.execute(
        operation=RELOAD_OPERATION,
        delegated_bearer="opaque-bearer",
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )

    assert first.status_code == 500
    assert first.payload["error"]["code"] == "management_operation_failed"
    assert replay.status_code == 500
    assert replay.payload["invocation_id"] == "reload-1"
    assert runtime.reload_calls == 1


@pytest.mark.asyncio
async def test_effect_audit_is_complete_and_contains_no_bearer(caplog) -> None:
    bearer = "tested-bearer-marker-that-must-not-leak"
    ledger = _Ledger()
    response = await _service(_Admission(), ledger, _Runtime()).execute(
        operation=RELOAD_OPERATION,
        delegated_bearer=bearer,
        invocation_id="reload-1",
        application_id="app-a@1-0",
    )

    key = ("access-1", RELOAD_OPERATION, "reload-1")
    audit = ledger.records[key]["audit"]
    assert audit == {
        "decision_id": "decision-1",
        "caller_profile": "caller-profile-1",
        "access_id": "access-1",
        "card_revision": 4,
        "card_catalog_version": "catalog-3",
        "active_catalog_version": "catalog-3",
        "invocation_policy_revision": 2,
        "request_permit_revision": 1,
        "tenant": "tenant-a",
        "project": "project-a",
        "resource": "urn:kdcube:management:deployment:tenant-a:project-a",
        "operation": RELOAD_OPERATION,
        "application_id": "app-a@1-0",
        "invocation_id": "reload-1",
        "request_digest": ledger.records[key]["request_digest"],
        "runtime_instance": "proc-1",
    }
    assert bearer not in str(response.payload)
    assert bearer not in str(ledger.records)
    assert bearer not in caplog.text
