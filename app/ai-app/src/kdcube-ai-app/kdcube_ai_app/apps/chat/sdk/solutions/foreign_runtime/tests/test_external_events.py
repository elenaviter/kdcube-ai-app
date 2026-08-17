# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""The foreign-runtime turn-batch fold (``foreign_runtime/external_events.py``).

The lane-wakeup dispatch hands a run-to-completion turn ONE external event
(the prompt), while the user's attachments ride separate lane events of the
same ingress batch. The fold must deliver the whole batch — the exact
surfaced bug was a hosted agent answering "whats here" blind to the attached
image. Read-only on the lane: nothing here consumes or reserves anything.

Ported from the ported-langgraph-agents bundle's tests/test_turn_batch.py
(the seam is that module generalized). Offline: the lane source is faked;
no redis, no store, no network.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kdcube_ai_app.apps.chat.external_events import ConversationExternalEvent
from kdcube_ai_app.apps.chat.sdk.protocol import hosted_external_event_attachments
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import external_events as mod


def _prompt_accepted(text: str = "whats here") -> dict:
    return {
        "event_id": "evt-prompt",
        "type": "event.user.prompt",
        "reactive": True,
        "payload": {"mime": "text/plain", "event": {"text": text}},
    }


def _attachment_accepted() -> dict:
    return {
        "event_id": "evt-att",
        "type": "event.user.attachment.file",
        "reactive": True,
        "payload": {
            "mime": "image/png",
            "event": {
                "filename": "photo.png",
                "mime": "image/png",
                "file_index": 0,
                "hosted_uri": "conv/turn_1/files/photo.png",
            },
        },
    }


def _lane_event(*, message_id: str, sequence: int, batch_id: str = "batch-1",
                accepted: dict, consumed_at: float | None = None) -> ConversationExternalEvent:
    return ConversationExternalEvent(
        message_id=message_id,
        batch_id=batch_id,
        kind="external_event",
        created_at=1000.0 + sequence,
        sequence=sequence,
        payload={"text": "", "event": dict(accepted), "is_continuation": False},
        consumed_at=consumed_at,
    )


class _FakeSource:
    def __init__(self, events):
        self._events = list(events)

    async def get_event(self, message_id):
        for item in self._events:
            if item.message_id == message_id:
                return item
        return None

    async def read_since(self, cursor, *, limit=None):
        return list(self._events)


def _wakeup_raw(event_id: str) -> dict:
    return {
        "meta": {"task_id": "task-1", "created_at": 1000.0},
        "routing": {"conversation_id": "conv-1", "session_id": "sess-1", "bundle_id": "bundle-1"},
        "actor": {"tenant_id": "tenant-a", "project_id": "project-a"},
        "user": {"user_id": "user-1", "user_type": "registered"},
        "event_lane": {
            "tenant": "tenant-a",
            "project": "project-a",
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "agent_id": "lg-react",
            "event_id": event_id,
        },
    }


def _entrypoint(event_id: str = "m-prompt") -> SimpleNamespace:
    return SimpleNamespace(
        redis=object(),
        comm_context=SimpleNamespace(bundle_call_context={"event_lane_wakeup": _wakeup_raw(event_id)}),
    )


def test_fold_delivers_the_hosted_attachment_beside_the_prompt(monkeypatch):
    """The surfaced case: prompt + hosted PNG in one batch — the fold must
    surface BOTH so the app's attachment seam finds the image."""
    prompt = _lane_event(message_id="m-prompt", sequence=2, accepted=_prompt_accepted())
    attachment = _lane_event(message_id="m-att", sequence=3, accepted=_attachment_accepted())
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: _FakeSource([attachment, prompt]))

    state = {"external_events": [_prompt_accepted()]}
    folded = asyncio.run(mod.fold_turn_external_events(_entrypoint(), state))

    assert [item["event_id"] for item in folded] == ["evt-prompt", "evt-att"]
    assert mod.folded_external_events_message_ids(state) == ["m-prompt", "m-att"]
    hosted = hosted_external_event_attachments(folded)
    assert len(hosted) == 1
    assert hosted[0]["hosted_uri"] == "conv/turn_1/files/photo.png"
    assert hosted[0]["mime"] == "image/png"


def test_fold_skips_batch_siblings_a_previous_turn_consumed(monkeypatch):
    """A re-woken queued event promotes ALONE: consumed siblings stay out and
    the dispatched events stand untouched."""
    consumed_prompt = _lane_event(
        message_id="m-prompt", sequence=2, accepted=_prompt_accepted(), consumed_at=1234.5,
    )
    followup = _lane_event(message_id="m-follow", sequence=4, accepted=_prompt_accepted("and this?"))
    monkeypatch.setattr(
        mod, "_lane_source", lambda redis, wakeup: _FakeSource([consumed_prompt, followup])
    )

    state = {"external_events": [_prompt_accepted("and this?")]}
    folded = asyncio.run(mod.fold_turn_external_events(_entrypoint("m-follow"), state))

    assert folded == [_prompt_accepted("and this?")]


def test_fold_is_inert_without_a_lane_wakeup():
    """Direct invocations (tests, ops) carry no wakeup context — the state's
    events pass through untouched, and the lane is never opened."""
    ep = SimpleNamespace(redis=object(), comm_context=SimpleNamespace(bundle_call_context={}))
    state = {"external_events": [_prompt_accepted()]}

    assert asyncio.run(mod.fold_turn_external_events(ep, state)) == [_prompt_accepted()]


def test_fold_fails_open_when_the_lane_read_breaks(monkeypatch):
    class _BrokenSource:
        async def get_event(self, message_id):
            raise RuntimeError("lane offline")

    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: _BrokenSource())
    state = {"external_events": [_prompt_accepted()]}

    folded = asyncio.run(mod.fold_turn_external_events(_entrypoint(), state))

    assert folded == [_prompt_accepted()]


def test_folded_batch_carries_the_attachment_body(monkeypatch):
    """The folded batch carries the attachment's full accepted body (base64
    included), so a downstream multimodality seam can materialize the image
    without any further lane/store read. (The bundle's own test drove its
    attachments seam here; the seam-level assertion is the hosted payload.)"""
    accepted = _attachment_accepted()
    # A real 1x1 PNG body so a downstream image normalizer has valid bytes.
    accepted["payload"]["event"]["base64"] = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    prompt = _lane_event(message_id="m-prompt", sequence=1, accepted=_prompt_accepted())
    attachment = _lane_event(message_id="m-att", sequence=2, accepted=accepted)
    monkeypatch.setattr(mod, "_lane_source", lambda redis, wakeup: _FakeSource([prompt, attachment]))

    state = {"external_events": [_prompt_accepted()]}
    folded = asyncio.run(mod.fold_turn_external_events(_entrypoint(), state))

    assert len(folded) == 2
    hosted = hosted_external_event_attachments(folded)
    assert len(hosted) == 1
    assert hosted[0]["base64"] == accepted["payload"]["event"]["base64"]
    assert hosted[0]["mime"] == "image/png"
