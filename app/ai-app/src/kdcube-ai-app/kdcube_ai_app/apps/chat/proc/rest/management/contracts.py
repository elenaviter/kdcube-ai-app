# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Wire contracts for delegated management of one running KDCube deployment."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

REQUEST_SCHEMA = "kdcube.management.request.v1"
RESULT_SCHEMA = "kdcube.management.result.v1"
ERROR_SCHEMA = "kdcube.management.error.v1"
EFFECT_SCHEMA = "kdcube.management.effect.v1"

INSPECT_OPERATION = "kdcube.management.deployment.inspect"
SURFACES_OPERATION = "kdcube.management.application.surfaces.read"
RELOAD_OPERATION = "kdcube.management.application.reload"
OPERATIONS = (INSPECT_OPERATION, SURFACES_OPERATION, RELOAD_OPERATION)

RESOURCE_SELECTOR = "urn:kdcube:management:deployment:*:*"
_INVOCATION_ID_RE = re.compile(r"^[!-~]{1,256}$")


def _text(value: Any) -> str:
    return str(value or "").strip()


def management_resource(tenant: str, project: str) -> str:
    """Return the descriptor-bound protected-resource identifier."""

    encoded_tenant = quote(_text(tenant), safe="-._~")
    encoded_project = quote(_text(project), safe="-._~")
    if not encoded_tenant or not encoded_project:
        raise ValueError("management deployment identity is incomplete")
    return f"urn:kdcube:management:deployment:{encoded_tenant}:{encoded_project}"


def validate_invocation_id(value: Any) -> str:
    invocation_id = _text(value)
    if not _INVOCATION_ID_RE.fullmatch(invocation_id):
        raise ValueError("Idempotency-Key must contain 1..256 printable ASCII characters")
    return invocation_id


def validate_application_id(value: Any) -> str:
    application_id = _text(value)
    if (
        not application_id
        or len(application_id) > 256
        or application_id in {".", ".."}
        or any(char in application_id for char in ("*", "/", "\\", "?", "#"))
        or any(ord(char) < 33 or ord(char) == 127 for char in application_id)
    ):
        raise ValueError("application_id must be one exact declared application identifier")
    return application_id


def management_request_document(
    *,
    resource: str,
    operation: str,
    application_id: str = "",
    body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "application_id": _text(application_id),
        "body": dict(body or {}),
        "operation": _text(operation),
        "resource": _text(resource),
        "schema": REQUEST_SCHEMA,
    }


def management_request_digest(
    *,
    resource: str,
    operation: str,
    application_id: str = "",
    body: Mapping[str, Any] | None = None,
    secret: str = "",
) -> str:
    encoded = json.dumps(
        management_request_document(
            resource=resource,
            operation=operation,
            application_id=application_id,
            body=body,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    secret_bytes = str(secret or "").encode("utf-8")
    if secret_bytes:
        if len(secret_bytes) < 32:
            raise ValueError("management request digest secret is too short")
        return hmac.new(
            secret_bytes,
            b"kdcube.management.request-digest.v1\0" + encoded,
            hashlib.sha256,
        ).hexdigest()
    return hashlib.sha256(encoded).hexdigest()


def target_document(*, tenant: str, project: str) -> dict[str, str]:
    return {"tenant": _text(tenant), "project": _text(project)}


def management_error(
    *,
    operation: str,
    resource: str,
    tenant: str,
    project: str,
    invocation_id: str,
    code: str,
    message: str,
    retryable: bool = False,
    recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": ERROR_SCHEMA,
        "ok": False,
        "operation": _text(operation),
        "resource": _text(resource),
        "target": target_document(tenant=tenant, project=project),
        "invocation_id": _text(invocation_id),
        "error": {
            "code": _text(code) or "management_request_failed",
            "message": (_text(message) or "The management request failed.")[:500],
            "retryable": bool(retryable),
        },
    }
    if recovery:
        payload["recovery"] = dict(recovery)
    return payload


def management_success(
    *,
    operation: str,
    resource: str,
    tenant: str,
    project: str,
    invocation_id: str,
    authority: Mapping[str, Any],
    result: Mapping[str, Any],
    replay: bool = False,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "ok": True,
        "operation": _text(operation),
        "resource": _text(resource),
        "target": target_document(tenant=tenant, project=project),
        "invocation": {"id": _text(invocation_id), "replay": bool(replay)},
        "authority": dict(authority),
        "result": dict(result),
    }


__all__ = [
    "EFFECT_SCHEMA",
    "ERROR_SCHEMA",
    "INSPECT_OPERATION",
    "OPERATIONS",
    "RELOAD_OPERATION",
    "REQUEST_SCHEMA",
    "RESOURCE_SELECTOR",
    "RESULT_SCHEMA",
    "SURFACES_OPERATION",
    "management_error",
    "management_request_digest",
    "management_request_document",
    "management_resource",
    "management_success",
    "target_document",
    "validate_application_id",
    "validate_invocation_id",
]
