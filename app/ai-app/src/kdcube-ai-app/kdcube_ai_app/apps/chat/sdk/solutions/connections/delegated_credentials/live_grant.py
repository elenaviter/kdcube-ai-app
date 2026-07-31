# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Authoritative live resolution for pointer-backed delegated grants."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.automation_access import (
    ACCESS_SOURCE_AGENT,
    ACCESS_SOURCE_MANUAL,
    ACCESS_SOURCE_OAUTH,
    AUTOMATION_ACCESS_SCHEMA,
    AutomationAccessRecord,
    automation_record_key,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.credential_view import (
    resource_matches,
)


class LiveGrantCardError(RuntimeError):
    """The current registry card could not be trusted as authorization state."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _required_text(value: Any, reason: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise LiveGrantCardError(reason)
    return text


async def resolve_live_grant_card(
    redis: Any,
    *,
    tenant: str,
    project: str,
    access_id: str,
    expected_client_id: str = "",
    expected_grantor_subject: str = "",
    expected_delegate_subject: str = "",
) -> AutomationAccessRecord | None:
    """Return the current valid card, None when revoked/expired, or raise.

    A pointer-bearing token has no snapshot fallback. Store failures, malformed
    records, and binding mismatches are authorization failures.
    """

    pointer = _required_text(access_id, "access_id_missing")
    try:
        raw = await redis.get(automation_record_key(tenant, project, pointer))
    except Exception as exc:
        raise LiveGrantCardError("lookup_unavailable") from exc
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise LiveGrantCardError("malformed_json") from exc
    if not isinstance(payload, Mapping):
        raise LiveGrantCardError("record_not_object")
    if str(payload.get("schema") or "").strip() != AUTOMATION_ACCESS_SCHEMA:
        raise LiveGrantCardError("schema_mismatch")
    if not isinstance(payload.get("resource_grants"), Mapping):
        raise LiveGrantCardError("resource_grants_invalid")
    if not isinstance(payload.get("operations"), list):
        raise LiveGrantCardError("operations_invalid")
    for field_name in ("account_scope", "named_service_operations"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            raise LiveGrantCardError(f"{field_name}_invalid")
    try:
        record = AutomationAccessRecord.from_mapping(payload)
    except Exception as exc:
        raise LiveGrantCardError("record_invalid") from exc

    _required_text(record.client_id, "client_id_missing")
    _required_text(record.grantor_subject, "grantor_subject_missing")
    _required_text(record.delegate_subject, "delegate_subject_missing")
    if record.access_id != pointer:
        raise LiveGrantCardError("access_id_mismatch")
    if record.source not in {ACCESS_SOURCE_MANUAL, ACCESS_SOURCE_OAUTH, ACCESS_SOURCE_AGENT}:
        raise LiveGrantCardError("source_invalid")
    if record.expires_at <= 0:
        raise LiveGrantCardError("expiry_missing")
    if record.expires_at <= int(time.time()):
        return None

    expected = (
        (expected_client_id, record.client_id, "client_id_mismatch"),
        (expected_grantor_subject, record.grantor_subject, "grantor_subject_mismatch"),
        (expected_delegate_subject, record.delegate_subject, "delegate_subject_mismatch"),
    )
    for expected_value, actual_value, reason in expected:
        clean_expected = str(expected_value or "").strip()
        if clean_expected and clean_expected != actual_value:
            raise LiveGrantCardError(reason)
    return record


def live_grants_for_resource(
    record: AutomationAccessRecord,
    resource: str,
) -> tuple[str, ...] | None:
    """Return the live grant union for a resource, preserving an empty grant."""

    matched = False
    grants: list[str] = []
    for configured_resource, configured_grants in record.resource_grants.items():
        if not resource_matches(str(configured_resource), str(resource or "")):
            continue
        matched = True
        for grant in configured_grants:
            text = str(grant or "").strip()
            if text and text not in grants:
                grants.append(text)
    return tuple(grants) if matched else None


__all__ = [
    "LiveGrantCardError",
    "live_grants_for_resource",
    "resolve_live_grant_card",
]
