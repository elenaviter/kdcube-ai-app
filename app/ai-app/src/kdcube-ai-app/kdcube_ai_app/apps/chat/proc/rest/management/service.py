# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Transport-neutral delegated management orchestration."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from kdcube_ai_app.apps.chat.proc.rest.management.admission import (
    AdmissionDecision,
    AdmissionUnavailable,
)
from kdcube_ai_app.apps.chat.proc.rest.management.contracts import (
    INSPECT_OPERATION,
    RELOAD_OPERATION,
    SURFACES_OPERATION,
    management_error,
    management_request_digest,
    management_success,
)
from kdcube_ai_app.apps.chat.proc.rest.management.effect_ledger import (
    ACTION_CONFLICT,
    ACTION_PENDING,
    ACTION_REPLAY,
    ACTION_UNKNOWN,
    EffectLedger,
)

LOGGER = logging.getLogger("ChatProc.DelegatedManagement")


class ManagementApplicationNotFound(LookupError):
    pass


class ManagementRuntimeUnavailable(RuntimeError):
    pass


class AdmissionEvaluator(Protocol):
    async def evaluate(
        self,
        *,
        delegated_bearer: str,
        resource: str,
        operation: str,
        invocation_id: str,
        request_digest: str,
        approval_context: Mapping[str, str] | None = None,
    ) -> AdmissionDecision: ...


class ManagementRuntime(Protocol):
    async def inspect_deployment(self) -> Mapping[str, Any]: ...

    async def application_surfaces(
        self, application_id: str
    ) -> Mapping[str, Any]: ...

    async def reload_application(
        self, application_id: str, *, caller_profile: str
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ManagementResponse:
    status_code: int
    payload: Mapping[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _authority_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    principal = _mapping(payload.get("principal"))
    provenance = _mapping(payload.get("provenance"))
    evidence: dict[str, Any] = {}
    values = {
        "decision_id": payload.get("decision_id"),
        "caller_profile": principal.get("client_id"),
        "access_id": provenance.get("access_id"),
        "card_revision": provenance.get("card_revision"),
        "card_catalog_version": provenance.get("card_catalog_version"),
        "active_catalog_version": provenance.get("active_catalog_version"),
        "invocation_policy_revision": provenance.get("invocation_policy_revision"),
        "request_permit_revision": provenance.get("request_permit_revision"),
    }
    for key, value in values.items():
        if value not in (None, "", 0):
            evidence[key] = value
    return evidence


def _admission_recovery(payload: Mapping[str, Any]) -> dict[str, Any]:
    consent = _mapping(payload.get("consent"))
    if not consent:
        details = _mapping(_mapping(payload.get("ret")).get("details"))
        consent = _mapping(details.get("recovery"))
    if consent.get("kind") != "delegated_request_permit":
        return {}
    authorization_url = str(
        consent.get("connection_hub_url") or consent.get("authorization_url") or ""
    ).strip()
    if not authorization_url:
        return {}
    context = _mapping(consent.get("approval_context"))
    recovery: dict[str, Any] = {
        "type": "consent_required",
        "reason": "delegated_request_permit_required",
        "authorization_url": authorization_url,
    }
    values = {
        "access_id": consent.get("access_id"),
        "resource": consent.get("resource"),
        "operation": consent.get("outer_operation") or consent.get("operation"),
        "application_id": context.get("application_id")
        or consent.get("application_id"),
        "invocation_id": consent.get("invocation_id"),
        "request_digest": consent.get("request_digest"),
        "card_revision": consent.get("card_revision"),
        "catalog_version": consent.get("catalog_version"),
        "expires_at": consent.get("expires_at"),
        "choices": consent.get("available_choices") or consent.get("choices"),
    }
    for key, value in values.items():
        if value not in (None, "", 0, [], ()):
            recovery[key] = value
    return recovery


class DelegatedManagementService:
    def __init__(
        self,
        *,
        tenant: str,
        project: str,
        resource: str,
        runtime_instance: str,
        admission: AdmissionEvaluator,
        ledger: EffectLedger,
        runtime: ManagementRuntime,
    ) -> None:
        self._tenant = tenant
        self._project = project
        self._resource = resource
        self._runtime_instance = str(runtime_instance or "").strip()
        self._admission = admission
        self._ledger = ledger
        self._runtime = runtime

    def _error(
        self,
        *,
        operation: str,
        invocation_id: str,
        code: str,
        message: str,
        status_code: int,
        retryable: bool = False,
        recovery: Mapping[str, Any] | None = None,
    ) -> ManagementResponse:
        return ManagementResponse(
            status_code,
            management_error(
                operation=operation,
                resource=self._resource,
                tenant=self._tenant,
                project=self._project,
                invocation_id=invocation_id,
                code=code,
                message=message,
                retryable=retryable,
                recovery=recovery,
            ),
        )

    async def execute(
        self,
        *,
        operation: str,
        delegated_bearer: str,
        invocation_id: str,
        application_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> ManagementResponse:
        request_digest = management_request_digest(
            resource=self._resource,
            operation=operation,
            application_id=application_id,
            body=body,
        )
        approval_context = (
            {"application_id": application_id} if application_id else {}
        )
        try:
            decision = await self._admission.evaluate(
                delegated_bearer=delegated_bearer,
                resource=self._resource,
                operation=operation,
                invocation_id=invocation_id,
                request_digest=request_digest,
                approval_context=approval_context,
            )
        except AdmissionUnavailable:
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="delegated_admission_unavailable",
                message="Connection Hub admission is unavailable.",
                status_code=503,
                retryable=True,
            )
        except Exception:
            LOGGER.exception(
                "delegated management admission failed operation=%s "
                "application=%s invocation=%s",
                operation,
                application_id or "<deployment>",
                invocation_id,
            )
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="delegated_admission_unavailable",
                message="Connection Hub admission is unavailable.",
                status_code=503,
                retryable=True,
            )

        if not decision.allowed:
            error = _mapping(decision.payload.get("error"))
            code = str(error.get("code") or "delegated_admission_denied")
            message = str(
                error.get("message") or "Connection Hub denied this operation."
            )
            retryable = bool(error.get("retryable", False))
            status_code = (
                decision.status_code
                if decision.status_code in {400, 401, 403, 409, 503}
                else 503
            )
            LOGGER.info(
                "delegated management denied operation=%s application=%s "
                "invocation=%s code=%s",
                operation,
                application_id or "<deployment>",
                invocation_id,
                code,
            )
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code=code,
                message=message,
                status_code=status_code,
                retryable=retryable,
                recovery=_admission_recovery(decision.payload),
            )

        authority = _authority_evidence(decision.payload)
        access_id = str(authority.get("access_id") or "").strip()
        caller_profile = str(authority.get("caller_profile") or "").strip()
        if not access_id or not caller_profile:
            LOGGER.error(
                "delegated management admission omitted caller identity "
                "operation=%s application=%s invocation=%s",
                operation,
                application_id or "<deployment>",
                invocation_id,
            )
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="delegated_admission_invalid",
                message="Connection Hub returned incomplete caller authority.",
                status_code=503,
                retryable=True,
            )
        audit = {
            **authority,
            "tenant": self._tenant,
            "project": self._project,
            "resource": self._resource,
            "operation": operation,
            "application_id": application_id,
            "invocation_id": invocation_id,
            "request_digest": request_digest,
            "runtime_instance": self._runtime_instance,
        }
        try:
            reservation = await self._ledger.reserve(
                access_id=access_id,
                resource=self._resource,
                operation=operation,
                invocation_id=invocation_id,
                request_digest=request_digest,
                audit=audit,
            )
        except Exception:
            LOGGER.exception(
                "delegated management effect ledger unavailable operation=%s "
                "application=%s invocation=%s",
                operation,
                application_id or "<deployment>",
                invocation_id,
            )
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="effect_ledger_unavailable",
                message="Management effect tracking is unavailable.",
                status_code=503,
                retryable=True,
            )

        if reservation.action == ACTION_REPLAY:
            record = _mapping(reservation.record)
            stored = _mapping(record.get("response"))
            replayed = copy.deepcopy(dict(stored))
            invocation = replayed.get("invocation")
            if isinstance(invocation, Mapping):
                replayed["invocation"] = {**invocation, "replay": True}
            return ManagementResponse(int(record.get("status_code") or 200), replayed)
        if reservation.action == ACTION_CONFLICT:
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="invocation_id_conflict",
                message="This Idempotency-Key belongs to a different request.",
                status_code=409,
            )
        if reservation.action == ACTION_PENDING:
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="effect_outcome_pending",
                message="The accepted management operation is still in progress.",
                status_code=409,
                retryable=True,
            )
        if reservation.action == ACTION_UNKNOWN:
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="effect_outcome_unknown",
                message="The accepted management operation has no recorded outcome.",
                status_code=409,
            )

        try:
            if operation == INSPECT_OPERATION:
                result = await self._runtime.inspect_deployment()
            elif operation == SURFACES_OPERATION:
                result = await self._runtime.application_surfaces(application_id)
            elif operation == RELOAD_OPERATION:
                result = await self._runtime.reload_application(
                    application_id,
                    caller_profile=caller_profile,
                )
            else:
                raise ValueError("unsupported management operation")
        except ManagementApplicationNotFound:
            response = self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="application_not_found",
                message="The exact application is not declared by this deployment.",
                status_code=404,
            )
            await self._settle(
                reservation=reservation,
                access_id=access_id,
                operation=operation,
                invocation_id=invocation_id,
                request_digest=request_digest,
                response=response,
                failed=True,
            )
            return response
        except ManagementRuntimeUnavailable:
            response = self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="management_runtime_unavailable",
                message="The KDCube management runtime is unavailable.",
                status_code=503,
                retryable=True,
            )
            await self._settle(
                reservation=reservation,
                access_id=access_id,
                operation=operation,
                invocation_id=invocation_id,
                request_digest=request_digest,
                response=response,
                failed=True,
            )
            return response
        except Exception:
            LOGGER.exception(
                "delegated management operation failed operation=%s application=%s "
                "invocation=%s",
                operation,
                application_id or "<deployment>",
                invocation_id,
            )
            response = self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="management_operation_failed",
                message="The KDCube management operation failed.",
                status_code=500,
            )
            await self._settle(
                reservation=reservation,
                access_id=access_id,
                operation=operation,
                invocation_id=invocation_id,
                request_digest=request_digest,
                response=response,
                failed=True,
            )
            return response

        response = ManagementResponse(
            200,
            management_success(
                operation=operation,
                resource=self._resource,
                tenant=self._tenant,
                project=self._project,
                invocation_id=invocation_id,
                authority=authority,
                result=result,
            ),
        )
        settled = await self._settle(
            reservation=reservation,
            access_id=access_id,
            operation=operation,
            invocation_id=invocation_id,
            request_digest=request_digest,
            response=response,
        )
        if not settled:
            return self._error(
                operation=operation,
                invocation_id=invocation_id,
                code="effect_outcome_unrecorded",
                message="The management effect completed but its outcome was not recorded.",
                status_code=503,
            )
        LOGGER.info(
            "delegated management completed operation=%s application=%s "
            "invocation=%s decision=%s caller=%s access_id=%s",
            operation,
            application_id or "<deployment>",
            invocation_id,
            authority.get("decision_id", ""),
            authority.get("caller_profile", ""),
            authority.get("access_id", ""),
        )
        return response

    async def _settle(
        self,
        *,
        reservation: Any,
        access_id: str,
        operation: str,
        invocation_id: str,
        request_digest: str,
        response: ManagementResponse,
        failed: bool = False,
    ) -> bool:
        try:
            await self._ledger.finish(
                access_id=access_id,
                resource=self._resource,
                operation=operation,
                invocation_id=invocation_id,
                request_digest=request_digest,
                owner=reservation.owner,
                status_code=response.status_code,
                response=response.payload,
                failed=failed,
            )
            return True
        except Exception:
            LOGGER.exception(
                "delegated management effect settlement failed operation=%s "
                "invocation=%s",
                operation,
                invocation_id,
            )
            return False


__all__ = [
    "AdmissionEvaluator",
    "DelegatedManagementService",
    "ManagementApplicationNotFound",
    "ManagementResponse",
    "ManagementRuntime",
    "ManagementRuntimeUnavailable",
]
