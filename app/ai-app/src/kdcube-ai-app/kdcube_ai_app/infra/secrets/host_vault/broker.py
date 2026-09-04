# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Stateless ``kdcube-secrets`` broker adapter.

Translates the existing internal secrets-service operations (``get``,
``set``, ``delete`` by a KDCube secrets-manager key) into host-vault protocol
requests over the deployment's mTLS identity. Nothing is cached: no value,
no negative result, no last response. A mutation is acknowledged to the
caller only when the vault answered ``ok`` (the store committed).

Namespace binding: the broker runs inside the deployment and knows the
canonical tenant/project. The HTTP adapter binds one logical application for
the whole deployment-wide secrets manager (``kdcube-runtime``); library users
may bind a narrower trusted application. The secret reference is DERIVED here
from the internal key, so no external party ever names a vault path.

The bearer model of the existing service (``X-KDCUBE-SECRET-TOKEN`` /
``X-KDCUBE-ADMIN-TOKEN``) is deliberately absent: possession of a copied
string is not workload identity, and this adapter offers no way to present
one to the vault.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from kdcube_ai_app.infra.secrets.host_vault.protocol import (
    ErrorCode,
    Operation,
    SecretNamespace,
    SecretReference,
    VaultError,
    VaultRequest,
    VaultResponse,
)

LOGGER = logging.getLogger("kdcube.host_vault.broker")


class VaultTransport:
    """What the broker needs from the wire: one call, one response."""

    def call(self, request: VaultRequest) -> VaultResponse:  # pragma: no cover - protocol
        raise NotImplementedError


@dataclass(frozen=True)
class BrokerResult:
    ok: bool
    code: ErrorCode
    generation: int | None = None


@dataclass(frozen=True)
class BrokerReadResult:
    ok: bool
    code: ErrorCode
    value: str | None = None
    generation: int | None = None


class SecretsBroker:
    def __init__(self, *, transport: VaultTransport, tenant: str, project: str) -> None:
        self._transport = transport
        self._tenant = tenant
        self._project = project

    def _reference(self, application: str, key: str) -> SecretReference:
        namespace = SecretNamespace(tenant=self._tenant, project=self._project, application=application)
        return SecretReference.derive(namespace=namespace, internal_key=key)

    def _call(self, request: VaultRequest) -> VaultResponse:
        response = self._transport.call(request)
        if not response.ok and response.code is ErrorCode.INTERNAL:
            LOGGER.warning("[kdcube-secrets] vault internal failure request=%s", request.request_id)
        return response

    # ── the internal secrets-service operations ───────────────────────────

    def read(self, *, application: str, key: str) -> BrokerReadResult:
        """Read one value while retaining the fixed protocol result code."""
        try:
            response = self._call(VaultRequest.new(Operation.GET, self._reference(application, key)))
        except VaultError as exc:
            return BrokerReadResult(ok=False, code=exc.code)
        return BrokerReadResult(
            ok=response.ok,
            code=response.code,
            value=response.value if response.ok else None,
            generation=response.generation,
        )

    def get(self, *, application: str, key: str) -> str | None:
        """Read one value. ``None`` for not found, forbidden, or unreachable:
        the same shape ``SecretsServiceSecretsManager.get_secret`` returns,
        so the later provider wiring changes nothing for readers."""
        result = self.read(application=application, key=key)
        return result.value if result.ok else None

    def set(self, *, application: str, key: str, value: str, expected_generation: int | None = None) -> BrokerResult:
        response = self._call(VaultRequest.new(
            Operation.SET, self._reference(application, key), value=value, expected_generation=expected_generation,
        ))
        return BrokerResult(ok=response.ok, code=response.code, generation=response.generation)

    def rotate(self, *, application: str, key: str, value: str, expected_generation: int | None = None) -> BrokerResult:
        response = self._call(VaultRequest.new(
            Operation.ROTATE, self._reference(application, key), value=value, expected_generation=expected_generation,
        ))
        return BrokerResult(ok=response.ok, code=response.code, generation=response.generation)

    def delete(self, *, application: str, key: str, expected_generation: int | None = None) -> BrokerResult:
        response = self._call(VaultRequest.new(
            Operation.DELETE, self._reference(application, key), expected_generation=expected_generation,
        ))
        if not response.ok and response.code is ErrorCode.NOT_FOUND:
            # Deleting an absent value is a settled state, as the existing
            # service treats a 404 delete.
            return BrokerResult(ok=True, code=ErrorCode.NOT_FOUND)
        return BrokerResult(ok=response.ok, code=response.code, generation=response.generation)

    def health(self) -> dict[str, Any]:
        try:
            response = self._call(VaultRequest.new(Operation.HEALTH))
        except VaultError as exc:
            return {"ok": False, "code": exc.code.value}
        return {"ok": response.ok, "code": response.code.value, **{k: v for k, v in response.extra.items() if k == "deployment_id"}}


__all__ = ["BrokerReadResult", "BrokerResult", "SecretsBroker", "VaultTransport"]
