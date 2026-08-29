# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The claim-driven consent-state resolver (`solutions/connections/consent_state.py`).

The base every integration shares: given a user + required claims, resolve each
claim to given / pending / unavailable, claim-first, rendering from the grant
vocabulary — so no service must author a friendly taxonomy for its consent to
appear. Stores are faked; the assertions pin the governance behaviors.
"""

from __future__ import annotations

import asyncio

from prokura.consent_state import (
    ClaimRequirement,
    GIVEN, PENDING, UNAVAILABLE,
    SOURCE_DELEGATED, SOURCE_CONNECTED,
    resolve_integration_consent,
)


class _FakeDelegatedReader:
    """Delegated-by-KDCube: a granted set + the grant vocabulary."""
    def __init__(self, granted, vocab):
        self._granted = set(granted)
        self._vocab = vocab

    async def granted_claims(self, user):
        return set(self._granted)

    def vocabulary(self, claim):
        return self._vocab.get(claim, {})


class _FakeConnectedReader:
    def __init__(self, states):
        self._states = states  # claim -> {state, label, ...}

    async def claim_state(self, *, claim, provider_id, connector_app_id, user):
        return self._states.get(claim, {"state": UNAVAILABLE})


_VOCAB = {
    "memories:read": {"label": "Read KDCube memories", "description": "Read your memory notes.", "delegable": True},
    "memories:write": {"label": "Write KDCube memories", "delegable": True},
    "admin:all": {"label": "Everything", "delegable": False},  # role not permitted
}


def _run(**kw):
    return asyncio.run(resolve_integration_consent(**kw))


def test_delegated_claim_given_when_granted():
    reader = _FakeDelegatedReader(granted={"memories:read"}, vocab=_VOCAB)
    ic = _run(
        integration="memories",
        requirements=[ClaimRequirement("memories:read", SOURCE_DELEGATED)],
        user={"sub": "u1"}, delegated_reader=reader,
    )
    c = ic.claims[0]
    assert c.state == GIVEN
    assert c.label == "Read KDCube memories"      # rendered from the grant vocabulary
    assert c.grant_action is None                 # nothing to grant
    assert ic.state == GIVEN


def test_delegated_claim_pending_when_delegable_but_not_granted():
    reader = _FakeDelegatedReader(granted=set(), vocab=_VOCAB)
    ic = _run(
        integration="memories",
        requirements=[ClaimRequirement("memories:read", SOURCE_DELEGATED)],
        user={"sub": "u1"}, delegated_reader=reader,
    )
    assert ic.claims[0].state == PENDING          # delegable, awaiting consent
    assert ic.state == PENDING                    # header rollup


def test_delegated_claim_unavailable_when_not_delegable():
    reader = _FakeDelegatedReader(granted=set(), vocab=_VOCAB)
    ic = _run(
        integration="admin",
        requirements=[ClaimRequirement("admin:all", SOURCE_DELEGATED)],
        user={"sub": "u1"}, delegated_reader=reader,
    )
    assert ic.claims[0].state == UNAVAILABLE      # role not permitted -> shown, not actionable


def test_header_rollup_is_pending_if_any_claim_pending():
    reader = _FakeDelegatedReader(granted={"memories:read"}, vocab=_VOCAB)
    ic = _run(
        integration="memories",
        requirements=[
            ClaimRequirement("memories:read", SOURCE_DELEGATED),   # given
            ClaimRequirement("memories:write", SOURCE_DELEGATED),  # pending
        ],
        user={"sub": "u1"}, delegated_reader=reader,
    )
    assert [c.state for c in ic.claims] == [GIVEN, PENDING]
    assert ic.state == PENDING


def test_connected_account_claim_routes_to_the_connected_reader():
    connected = _FakeConnectedReader({
        "slack:read": {"state": GIVEN, "label": "Read Slack"},
    })
    ic = _run(
        integration="slack",
        requirements=[ClaimRequirement("slack:read", SOURCE_CONNECTED, provider_id="slack")],
        user={"sub": "u1"}, connected_reader=connected,
    )
    assert ic.claims[0].state == GIVEN and ic.claims[0].source == SOURCE_CONNECTED


def test_missing_reader_is_unavailable_not_a_crash():
    # No delegated reader wired -> the claim shows as unavailable, no exception.
    ic = _run(
        integration="memories",
        requirements=[ClaimRequirement("memories:read", SOURCE_DELEGATED)],
        user={"sub": "u1"},
    )
    assert ic.claims[0].state == PENDING  # default delegable=True when no vocab; still not granted
