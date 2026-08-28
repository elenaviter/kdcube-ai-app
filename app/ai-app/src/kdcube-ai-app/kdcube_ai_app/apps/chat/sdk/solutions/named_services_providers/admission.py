# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Explicit authority at the named-service dispatch boundary.

Identity and provider payload are existing, separate contracts. Admission says
which authority regime permits this one decoded named-service invocation and
which request-scoped execution bindings must surround provider dispatch.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    NamedServiceRequest,
    NamedServiceResponse,
)

ADMISSION_MODE_APPLICATION = "application"
ADMISSION_MODE_DELEGATED = "delegated"

DELEGATED_SELECTOR_AGENT = "agent_card"
DELEGATED_SELECTOR_BEARER = "bearer_card"

_ADMISSION_MODES = {ADMISSION_MODE_APPLICATION, ADMISSION_MODE_DELEGATED}
_DELEGATED_SELECTOR_KINDS = {DELEGATED_SELECTOR_AGENT, DELEGATED_SELECTOR_BEARER}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def effective_named_service_operation(request: NamedServiceRequest) -> str:
    """Return the capability operation represented by a decoded request."""

    operation = _clean(request.operation)
    action = _clean(request.action)
    if operation == "object.action" and action:
        return f"{operation}.{action}"
    return operation


@dataclass(frozen=True)
class NamedServiceAdmissionSelector:
    """JSON-safe authority selector carried by the platform relay.

    It contains non-secret identity needed to resolve current authority in the
    target runtime. It never contains a card snapshot, catalog, account binding,
    bearer token, or provider credential.
    """

    mode: str
    source: str
    delegated_kind: str = ""
    access_id: str = ""
    client_id: str = ""
    grantor_user_id: str = ""
    delegate_identity: str = ""
    expires_at: int = 0
    source_bundle_id: str = ""
    source_agent_id: str = ""

    def __post_init__(self) -> None:
        mode = _clean(self.mode)
        source = _clean(self.source)
        delegated_kind = _clean(self.delegated_kind)
        if mode not in _ADMISSION_MODES:
            raise ValueError(f"Unsupported named-service admission mode: {mode!r}")
        if not source:
            raise ValueError("Named-service admission source is required")
        if mode == ADMISSION_MODE_DELEGATED:
            if delegated_kind not in _DELEGATED_SELECTOR_KINDS:
                raise ValueError("Delegated named-service admission requires a selector kind")
            if not _clean(self.client_id):
                raise ValueError("Delegated named-service admission requires client_id")
            if delegated_kind == DELEGATED_SELECTOR_AGENT and not (
                _clean(self.source_bundle_id) and _clean(self.source_agent_id)
            ):
                raise ValueError(
                    "Agent-card named-service admission requires source bundle and agent identities"
                )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "delegated_kind", delegated_kind)
        object.__setattr__(self, "access_id", _clean(self.access_id))
        object.__setattr__(self, "client_id", _clean(self.client_id))
        object.__setattr__(self, "grantor_user_id", _clean(self.grantor_user_id))
        object.__setattr__(self, "delegate_identity", _clean(self.delegate_identity))
        object.__setattr__(self, "source_bundle_id", _clean(self.source_bundle_id))
        object.__setattr__(self, "source_agent_id", _clean(self.source_agent_id))
        object.__setattr__(self, "expires_at", int(self.expires_at or 0))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NamedServiceAdmissionSelector":
        data = dict(value or {})
        return cls(
            mode=data.get("mode") or "",
            source=data.get("source") or "",
            delegated_kind=data.get("delegated_kind") or "",
            access_id=data.get("access_id") or "",
            client_id=data.get("client_id") or "",
            grantor_user_id=data.get("grantor_user_id") or "",
            delegate_identity=data.get("delegate_identity") or "",
            expires_at=int(data.get("expires_at") or 0),
            source_bundle_id=data.get("source_bundle_id") or "",
            source_agent_id=data.get("source_agent_id") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        values = {
            "mode": self.mode,
            "source": self.source,
            "delegated_kind": self.delegated_kind,
            "access_id": self.access_id,
            "client_id": self.client_id,
            "grantor_user_id": self.grantor_user_id,
            "delegate_identity": self.delegate_identity,
            "expires_at": self.expires_at,
            "source_bundle_id": self.source_bundle_id,
            "source_agent_id": self.source_agent_id,
        }
        return {key: value for key, value in values.items() if value not in ("", 0, None)}


class NamedServiceExecutionScope(Protocol):
    """Request-scoped bindings required while the provider is invoked."""

    def bind(self) -> AbstractContextManager[Any]:
        ...


class NamedServiceAdmissionAuthorizer(Protocol):
    async def authorize(
        self,
        request: NamedServiceRequest,
    ) -> "NamedServiceAdmissionDecision":
        ...


@dataclass(frozen=True)
class NamedServiceAdmissionDecision:
    allowed: bool
    denial: NamedServiceResponse | None = None
    execution_scope: NamedServiceExecutionScope | None = None
    audit: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        *,
        execution_scope: NamedServiceExecutionScope | None = None,
        audit: Mapping[str, Any] | None = None,
    ) -> "NamedServiceAdmissionDecision":
        return cls(
            allowed=True,
            execution_scope=execution_scope,
            audit=dict(audit or {}),
        )

    @classmethod
    def deny(
        cls,
        denial: NamedServiceResponse,
        *,
        audit: Mapping[str, Any] | None = None,
    ) -> "NamedServiceAdmissionDecision":
        return cls(allowed=False, denial=denial, audit=dict(audit or {}))

    def bind(self) -> AbstractContextManager[Any]:
        if self.execution_scope is None:
            return nullcontext()
        return self.execution_scope.bind()


@dataclass(frozen=True)
class NamedServiceAdmission:
    """Authority declaration required for one named-service invocation."""

    selector: NamedServiceAdmissionSelector
    authorizer: NamedServiceAdmissionAuthorizer | None = None

    @classmethod
    def application(cls, *, source: str) -> "NamedServiceAdmission":
        return cls(
            selector=NamedServiceAdmissionSelector(
                mode=ADMISSION_MODE_APPLICATION,
                source=source,
            )
        )

    @classmethod
    def delegated(
        cls,
        *,
        selector: NamedServiceAdmissionSelector,
        authorizer: NamedServiceAdmissionAuthorizer | None = None,
    ) -> "NamedServiceAdmission":
        if selector.mode != ADMISSION_MODE_DELEGATED:
            raise ValueError("Delegated admission requires a delegated selector")
        return cls(selector=selector, authorizer=authorizer)

    @property
    def mode(self) -> str:
        return self.selector.mode

    @property
    def can_authorize_locally(self) -> bool:
        return self.mode == ADMISSION_MODE_APPLICATION or self.authorizer is not None

    def validate(self, request: NamedServiceRequest) -> None:
        if not isinstance(request, NamedServiceRequest):
            # A caller loaded by path holds its own generation of this package,
            # so its decoded request is a different class object with the same
            # shape. `coerce` accepts that and still refuses anything it cannot
            # decode, which is what the type check is for.
            try:
                request = NamedServiceRequest.coerce(request)
            except TypeError as exc:
                raise TypeError(
                    "Named-service admission validates a decoded NamedServiceRequest"
                ) from exc
        if not _clean(request.namespace):
            raise ValueError("Named-service admission requires a request namespace")
        if not effective_named_service_operation(request):
            raise ValueError("Named-service admission requires a request operation")

    async def authorize(self, request: NamedServiceRequest) -> NamedServiceAdmissionDecision:
        self.validate(request)
        if self.mode == ADMISSION_MODE_APPLICATION:
            return NamedServiceAdmissionDecision.allow(
                audit={"mode": self.mode, "source": self.selector.source}
            )
        if self.authorizer is None:
            return NamedServiceAdmissionDecision.deny(
                NamedServiceResponse.error_response(
                    code="named_service_admission_resolution_required",
                    message="Delegated named-service authority must be resolved in the provider runtime.",
                    status=503,
                    namespace=request.namespace,
                ),
                audit={"mode": self.mode, "source": self.selector.source},
            )
        return await self.authorizer.authorize(request)

    def relay_selector(self) -> dict[str, Any]:
        if (
            self.selector.delegated_kind == DELEGATED_SELECTOR_BEARER
            and not self.selector.access_id
        ):
            raise ValueError(
                "A legacy bearer snapshot without access_id cannot cross the named-service relay"
            )
        return self.selector.to_dict()


__all__ = [
    "ADMISSION_MODE_APPLICATION",
    "ADMISSION_MODE_DELEGATED",
    "DELEGATED_SELECTOR_AGENT",
    "DELEGATED_SELECTOR_BEARER",
    "NamedServiceAdmission",
    "NamedServiceAdmissionAuthorizer",
    "NamedServiceAdmissionDecision",
    "NamedServiceAdmissionSelector",
    "NamedServiceExecutionScope",
    "effective_named_service_operation",
]
