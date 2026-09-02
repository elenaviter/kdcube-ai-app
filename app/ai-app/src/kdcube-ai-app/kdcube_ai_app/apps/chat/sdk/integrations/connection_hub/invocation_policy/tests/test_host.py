from __future__ import annotations

import pytest

from connection_hub.invocation_policy import (
    POLICY_ONCE,
    SURFACE_OUTER,
    InvocationAuthority,
    canonical_request_digest,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.invocation_policy import (
    build_invocation_policy_service,
)


@pytest.mark.asyncio
async def test_host_binding_persists_and_consumes_a_once_policy(tmp_path):
    authority = InvocationAuthority(
        access_id="access-1",
        resource="urn:example:service",
        surface=SURFACE_OUTER,
        operation="records.delete",
    )
    first = build_invocation_policy_service(storage_root=tmp_path)
    await first.set_policy(
        owner_subject="user-1",
        authority=authority,
        mode=POLICY_ONCE,
        now=100,
    )

    recomposed = build_invocation_policy_service(storage_root=tmp_path)
    decision = await recomposed.begin(
        owner_subject="user-1",
        authority=authority,
        invocation_id="invoke-1",
        request_digest=canonical_request_digest({"record_id": "record-7"}),
        now=101,
    )

    assert decision.allowed is True
    assert decision.policy is not None
    assert decision.policy.remaining == 0
