from __future__ import annotations

import time
from urllib.parse import urlencode

import httpx
import pytest
from connection_hub.delegated_credentials.request_approval import (
    RequestApprovalTicket,
    issue_request_approval_ticket,
)
from kdcube_ai_app.apps.chat.proc.rest.management.admission import (
    AdmissionUnavailable,
    ConnectionHubAdmissionClient,
)

SERVICE_SECRET = "management-service-secret-with-at-least-32-bytes"
RESOURCE = "urn:kdcube:management:deployment:tenant-a:project-a"
OPERATION = "kdcube.management.application.reload"
DIGEST = "a" * 64


def _ticket(*, invocation_id: str = "reload-1") -> tuple[str, int]:
    moment = int(time.time())
    ticket = RequestApprovalTicket(
        service_id="kdcube-management",
        client_id="cli-profile",
        access_id="access-1",
        resource=RESOURCE,
        operation=OPERATION,
        invocation_id=invocation_id,
        request_digest=DIGEST,
        card_revision=4,
        authority_revision="catalog-3",
        issued_at=moment,
        expires_at=moment + 600,
        approval_context={"application_id": "app-a@1-0"},
    )
    return (
        issue_request_approval_ticket(ticket, secret=SERVICE_SECRET),
        ticket.expires_at,
    )


def _denial(token: str) -> dict:
    consent = {
        "kind": "delegated_request_permit",
        "reason": "delegated_invocation_policy_required",
        "connection_hub_url": (
            "https://runtime.example/approve?"
            + urlencode({"request_approval_ticket": token})
        ),
        "access_id": "access-1",
        "resource": RESOURCE,
        "outer_operation": OPERATION,
        "invocation_id": "reload-1",
        "request_digest": DIGEST,
        "card_revision": 4,
        "catalog_version": "catalog-3",
        "expires_at": 1,
        "approval_context": {"application_id": "app-a@1-0"},
        "available_choices": ["allow_once", "allow_always"],
    }
    return {
        "allowed": False,
        "error": {"code": "delegated_invocation_policy_required"},
        "consent": consent,
        "ret": {"details": {"recovery": dict(consent)}},
    }


def _client(payload: dict) -> ConnectionHubAdmissionClient:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json=payload)

    transport = httpx.MockTransport(handler)
    return ConnectionHubAdmissionClient(
        admission_url="https://runtime.example/delegated_admission",
        service_id="kdcube-management",
        service_secret=SERVICE_SECRET,
        client=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.asyncio
async def test_request_bound_expiry_comes_from_verified_ticket() -> None:
    token, expires_at = _ticket()
    client = _client(_denial(token))

    decision = await client.evaluate(
        delegated_bearer="opaque-bearer",
        resource=RESOURCE,
        operation=OPERATION,
        invocation_id="reload-1",
        request_digest=DIGEST,
        approval_context={"application_id": "app-a@1-0"},
    )

    assert decision.payload["consent"]["expires_at"] == expires_at
    assert decision.payload["ret"]["details"]["recovery"]["expires_at"] == expires_at
    await client._client.aclose()


@pytest.mark.asyncio
async def test_request_bound_ticket_mismatch_fails_closed() -> None:
    token, _expires_at = _ticket(invocation_id="another-reload")
    client = _client(_denial(token))

    with pytest.raises(AdmissionUnavailable):
        await client.evaluate(
            delegated_bearer="opaque-bearer",
            resource=RESOURCE,
            operation=OPERATION,
            invocation_id="reload-1",
            request_digest=DIGEST,
            approval_context={"application_id": "app-a@1-0"},
        )
    await client._client.aclose()
