# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Signed client for Connection Hub direct protected-service admission."""

from __future__ import annotations

import copy
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
from connection_hub.delegated_credentials.admission import (
    SERVICE_ID_HEADER,
    SERVICE_NONCE_HEADER,
    SERVICE_SIGNATURE_HEADER,
    SERVICE_TIMESTAMP_HEADER,
    AdmissionRequest,
    sign_admission_request,
)
from connection_hub.delegated_credentials.request_approval import (
    RequestApprovalTicketError,
    verify_request_approval_ticket,
)


class AdmissionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AdmissionDecision:
    status_code: int
    payload: Mapping[str, Any]

    @property
    def allowed(self) -> bool:
        return self.status_code == 200 and self.payload.get("allowed") is True


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _request_bound_consent(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    consent = _mapping(payload.get("consent"))
    if consent:
        return consent
    details = _mapping(_mapping(payload.get("ret")).get("details"))
    return _mapping(details.get("recovery"))


def _validated_request_bound_recovery(
    payload: Mapping[str, Any],
    *,
    admission: AdmissionRequest,
    service_id: str,
    service_secret: str,
) -> dict[str, Any]:
    consent = _request_bound_consent(payload)
    if consent.get("kind") != "delegated_request_permit":
        return dict(payload)

    authorization_url = str(
        consent.get("connection_hub_url") or consent.get("authorization_url") or ""
    ).strip()
    try:
        query = parse_qs(
            urlsplit(authorization_url).query,
            keep_blank_values=True,
            max_num_fields=128,
        )
    except ValueError as exc:
        raise AdmissionUnavailable(
            "Connection Hub request approval is invalid"
        ) from exc
    approval_tokens = query.get("request_approval_ticket", [])
    if len(approval_tokens) != 1:
        raise AdmissionUnavailable("Connection Hub request approval is invalid")
    try:
        ticket = verify_request_approval_ticket(
            approval_tokens[0],
            secret=service_secret,
        )
        card_revision = int(consent.get("card_revision") or 0)
    except (RequestApprovalTicketError, TypeError, ValueError) as exc:
        raise AdmissionUnavailable(
            "Connection Hub request approval is invalid"
        ) from exc

    context = {
        str(key or "").strip(): str(value or "").strip()
        for key, value in dict(_mapping(consent.get("approval_context"))).items()
        if str(key or "").strip() and str(value or "").strip()
    }
    expected = (
        service_id,
        admission.resource,
        admission.operation,
        admission.invocation_id,
        admission.request_digest,
        dict(sorted(admission.approval_context.items())),
    )
    actual = (
        ticket.service_id,
        ticket.resource,
        ticket.operation,
        ticket.invocation_id,
        ticket.request_digest,
        dict(ticket.approval_context),
    )
    if (
        actual != expected
        or str(consent.get("access_id") or "").strip() != ticket.access_id
        or card_revision != ticket.card_revision
        or str(consent.get("catalog_version") or "").strip()
        != ticket.authority_revision
        or context != dict(ticket.approval_context)
    ):
        raise AdmissionUnavailable("Connection Hub request approval is invalid")

    normalized = copy.deepcopy(dict(payload))
    top_level = normalized.get("consent")
    if isinstance(top_level, Mapping):
        normalized["consent"] = {**top_level, "expires_at": ticket.expires_at}
    ret = normalized.get("ret")
    if isinstance(ret, Mapping):
        details = ret.get("details")
        if isinstance(details, Mapping):
            recovery = details.get("recovery")
            if isinstance(recovery, Mapping):
                normalized["ret"] = {
                    **ret,
                    "details": {
                        **details,
                        "recovery": {
                            **recovery,
                            "expires_at": ticket.expires_at,
                        },
                    },
                }
    return normalized


class ConnectionHubAdmissionClient:
    def __init__(
        self,
        *,
        admission_url: str,
        service_id: str,
        service_secret: str,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._admission_url = str(admission_url or "").strip()
        self._service_id = str(service_id or "").strip()
        self._service_secret = str(service_secret or "")
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._client = client

    async def evaluate(
        self,
        *,
        delegated_bearer: str,
        resource: str,
        operation: str,
        invocation_id: str,
        request_digest: str,
        approval_context: Mapping[str, str] | None = None,
    ) -> AdmissionDecision:
        admission = AdmissionRequest(
            resource=resource,
            operation=operation,
            invocation_id=invocation_id,
            request_digest=request_digest,
            approval_context=dict(approval_context or {}),
        )
        validation_error = admission.validation_error()
        if validation_error:
            raise ValueError(validation_error)
        timestamp = str(int(time.time()))
        nonce = secrets.token_urlsafe(24)
        signature = sign_admission_request(
            secret=self._service_secret,
            service_id=self._service_id,
            timestamp=timestamp,
            nonce=nonce,
            delegated_token=delegated_bearer,
            request=admission,
        )
        headers = {
            "Authorization": f"Bearer {delegated_bearer}",
            SERVICE_ID_HEADER: self._service_id,
            SERVICE_TIMESTAMP_HEADER: timestamp,
            SERVICE_NONCE_HEADER: nonce,
            SERVICE_SIGNATURE_HEADER: signature,
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.post(
                self._admission_url,
                headers=headers,
                json=admission.signing_dict(),
            )
        except httpx.HTTPError as exc:
            raise AdmissionUnavailable("Connection Hub admission is unavailable") from exc
        finally:
            if owns_client:
                await client.aclose()

        try:
            parsed = response.json()
        except ValueError as exc:
            raise AdmissionUnavailable(
                "Connection Hub admission returned an invalid response"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise AdmissionUnavailable(
                "Connection Hub admission returned an invalid response"
            )
        normalized = _validated_request_bound_recovery(
            parsed,
            admission=admission,
            service_id=self._service_id,
            service_secret=self._service_secret,
        )
        return AdmissionDecision(response.status_code, normalized)


__all__ = [
    "AdmissionDecision",
    "AdmissionUnavailable",
    "ConnectionHubAdmissionClient",
]
