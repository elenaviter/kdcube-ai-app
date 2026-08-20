# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Framework-neutral conversation recording (option A) — module + entrypoint wiring."""

import asyncio
import json

import pytest

from kdcube_ai_app.apps.chat.sdk.infra.economics.policy import EconomicsLimitException
from kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint import BaseEntrypoint
from kdcube_ai_app.apps.chat.sdk.runtime.turn_recording import (
    ASSISTANT_COMPLETION_BLOCK_TYPE,
    TURN_LOG_RECORDING_MINIMAL,
    TURN_LOG_RECORDING_NONE,
    TURN_LOG_RECORDING_RICH,
    build_error_turn_log_payload,
    build_minimal_turn_log_payload,
    mark_turn_error_surfaced,
    mark_turn_log_recorded,
    record_conversation_timeline,
    record_error_turn_log_if_absent,
    record_minimal_turn_log_if_absent,
    reset_turn_error_surfaced,
    reset_turn_log_recorded,
    rich_turn_log_was_recorded,
    turn_error_was_surfaced,
    turn_log_recording_kind,
    turn_log_was_recorded,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.payload import (
    TIMELINE_KIND,
)


class _FakeConversationClient:
    """Stand-in for ContextRAGClient: records save calls and, like the real
    ``save_turn_log_as_artifact``, marks the turn recorded after writing. Its
    ``recent`` reports back the timeline artifacts written so far (matched by
    kind + conversation), so the register-once timeline logic is exercised."""

    def __init__(self):
        self.calls = []
        self.artifact_calls = []
        self.recent_calls = []

    async def save_turn_log_as_artifact(self, **kwargs):
        self.calls.append(kwargs)
        mark_turn_log_recorded(kwargs.get("recording_kind", TURN_LOG_RECORDING_RICH))
        return {"hosted_uri": "u", "message_id": "m", "rn": "r"}

    async def save_artifact(self, **kwargs):
        self.artifact_calls.append(kwargs)
        return {"hosted_uri": "u", "message_id": "m", "rn": "r"}

    async def recent(self, **kwargs):
        self.recent_calls.append(kwargs)
        wanted = set(kwargs.get("kinds") or [])
        conv = kwargs.get("conversation_id")
        limit = kwargs.get("limit")
        # Mirror the real index row: `text` == the artifact's content_str, and rows
        # come back newest-first (most-recent timeline wins).
        items = [
            {"turn_id": a.get("turn_id"), "text": a.get("content_str") or ""}
            for a in reversed(self.artifact_calls)
            if f"artifact:{a.get('kind')}" in wanted
            and (conv is None or a.get("conversation_id") == conv)
        ]
        if "artifact:turn.log" in wanted:
            items.extend(
                {
                    "turn_id": c.get("turn_id"),
                    "text": json.dumps(c.get("payload") or {}, ensure_ascii=False),
                    "payload": c.get("payload") or {},
                    "bundle_id": c.get("bundle_id"),
                    "tags": ["artifact:turn.log", f"turn:{c.get('turn_id')}"],
                }
                for c in reversed(self.calls)
                if conv is None or c.get("conversation_id") == conv
            )
        if isinstance(limit, int):
            items = items[:limit]
        return {"items": items}


# ---------------------------------------------------------------- module unit


@pytest.mark.asyncio
async def test_records_minimal_log_for_non_react_turn():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        agent_id="raw", final_answer="Hello there.",
    )
    assert wrote is True
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["turn_id"] == "turn-1"
    assert call["agent_id"] == "raw"
    blocks = call["payload"]["blocks"]
    completions = [b for b in blocks if b["type"] == ASSISTANT_COMPLETION_BLOCK_TYPE]
    assert len(completions) == 1
    assert completions[0]["text"] == "Hello there."
    assert call["recording_kind"] == TURN_LOG_RECORDING_MINIMAL
    assert call["index_transcript"] is True
    assert turn_log_recording_kind() == TURN_LOG_RECORDING_MINIMAL
    assert rich_turn_log_was_recorded() is False


@pytest.mark.asyncio
async def test_no_double_write_when_turn_log_already_recorded():
    # React path: a rich turn log was written this turn -> flag set -> no-op.
    reset_turn_log_recorded()
    mark_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        final_answer="Would be written if the flag were unset.",
    )
    assert wrote is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_recorded_signal_crosses_child_task_boundary():
    # Regression (the overwrite bug): React persists its rich turn log inside a
    # SEPARATE asyncio task. asyncio copies the context at task creation, so a
    # `ContextVar.set(True)` there never reaches run()'s task and the fallback used
    # to overwrite React's rich log with an assistant-only one. The signal now
    # lives in a MUTABLE dict on the ContextVar: the child's copy shares the same
    # dict object, so a mutation inside the child IS visible to the parent.
    reset_turn_log_recorded()                       # bind the shared dict in THIS (parent) task

    async def _child_persists():
        # Runs in its own task (own copied context), exactly like React's persist.
        mark_turn_log_recorded()

    await asyncio.create_task(_child_persists())     # copies context -> shares the dict

    # Parent task now sees the child's mutation — this is what was broken before.
    assert turn_log_was_recorded() is True
    client = _FakeConversationClient()
    wrote = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-x", turn_id="turn-x", bundle_id="bundle.demo",
        final_answer="Assistant-only log that must NOT overwrite React's rich log.",
    )
    assert wrote is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_empty_final_answer_is_not_recorded():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    for answer in ("", "   ", "\n\t "):
        wrote = await record_minimal_turn_log_if_absent(
            conversation_client=client,
            tenant="t", project="p", user="u", user_type="registered",
            conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
            final_answer=answer,
        )
        assert wrote is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_steps_become_assistant_step_blocks():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        final_answer="Done.",
        steps=[
            {"text": "Thinking", "status": "started"},
            {"step": "Working", "status": "running"},
            {"text": "   "},  # dropped: blank
        ],
    )
    assert wrote is True
    blocks = client.calls[0]["payload"]["blocks"]
    steps = [b for b in blocks if b["type"] == "assistant.step"]
    assert [b["text"] for b in steps] == ["Thinking", "Working"]
    assert blocks[-1]["type"] == ASSISTANT_COMPLETION_BLOCK_TYPE


def test_build_minimal_turn_log_payload_shape():
    payload = build_minimal_turn_log_payload(final_answer="Answer", turn_id="turn-9")
    assert payload["blocks_count"] == len(payload["blocks"]) == 1
    assert payload["blocks"][0]["type"] == ASSISTANT_COMPLETION_BLOCK_TYPE
    assert payload["blocks"][0]["turn"] == "turn-9"
    assert "ts" in payload and "end_ts" in payload
    # No title key unless one is supplied (later turns).
    assert "conversation_title" not in payload


def test_build_minimal_turn_log_payload_carries_title():
    payload = build_minimal_turn_log_payload(
        final_answer="Answer", turn_id="turn-9", conversation_title="  Weekend plans  ",
    )
    assert payload["conversation_title"] == "Weekend plans"


def test_minimal_turn_log_records_user_prompt_attachments_and_files():
    """So a run-to-completion turn reloads like React: the payload carries a
    user.prompt block, a user.attachment.meta block (pullable conv:fi: ref), and a
    react.tool.result file block (pullable conv:fi: ref) — the exact block types the
    shared reload reader reconstructs the user bubble + attachments + files from."""
    payload = build_minimal_turn_log_payload(
        final_answer="Done.",
        turn_id="turn-9",
        user_prompt_text="  Use run_python to make primes.csv  ",
        user_attachments=[{"filename": "data.xlsx", "mime": "application/vnd.ms-excel",
                           "hosted_uri": "s3://b/u/c/t/data.xlsx", "rn": "rn:data"}],
        assistant_files=[{"filename": "primes.csv", "mime": "text/csv",
                          "logical_path": "conv:fi:conv-1.turn-9.files/primes.csv",
                          "hosted_uri": "s3://b/u/c/t/primes.csv", "rn": "rn:primes"}],
    )
    by_type = {b["type"]: b for b in payload["blocks"]}

    up = by_type["user.prompt"]
    assert up["text"] == "Use run_python to make primes.csv"   # stripped
    assert up["turn_id"] == "turn-9" and up["mime"] == "text/markdown"
    assert up["path"] == "conv:ar:turn-9.user.prompt"

    att = by_type["user.attachment.meta"]
    assert att["path"] == "conv:fi:turn-9.user.attachments/data.xlsx"   # pullable ref
    assert att["meta"]["hosted_uri"] == "s3://b/u/c/t/data.xlsx"
    digest = json.loads(att["text"])
    assert digest["kind"] == "file" and digest["visibility"] == "external"

    fb = by_type["react.tool.result"]
    meta_json = json.loads(fb["text"])
    assert meta_json["kind"] == "file" and meta_json["visibility"] == "external"
    assert meta_json["artifact_path"] == "conv:fi:conv-1.turn-9.files/primes.csv"   # pullable
    assert meta_json["hosted_uri"] == "s3://b/u/c/t/primes.csv"
    assert fb["path"] == "conv:tc:turn-9.code_exec_0.result"

    # The final answer is still last.
    assert payload["blocks"][-1]["type"] == ASSISTANT_COMPLETION_BLOCK_TYPE


def test_minimal_turn_log_preserves_followup_text_attachments_and_events():
    """A non-ReAct follow-up can carry typed text, files, and context objects in
    one accepted event batch. Reload must keep that whole batch together and must
    not turn the follow-up into a fresh prompt."""
    from kdcube_ai_app.apps.chat.sdk.runtime.user_inputs import iter_turn_user_input_entries

    ctx_event = {
        "id": "ctx1",
        "type": "event.user.context.object",
        "batch_id": "batch-followup",
        "payload": {
            "object_ref": "conv:ar:turn-1.artifact",
            "label": "runtime note",
        },
    }
    payload = build_minimal_turn_log_payload(
        final_answer="Done.",
        turn_id="turn-9",
        user_prompt_text="  please send this too  ",
        user_event_type="event.user.followup",
        user_attachments=[{
            "filename": "keep-the-agent-add-the-runtime.png",
            "mime": "image/png",
            "hosted_uri": "s3://bucket/user/conv/turn/file.png",
            "rn": "rn:file",
        }],
        user_events=[ctx_event],
        batch_id="batch-followup",
    )

    prompt = [b for b in payload["blocks"] if b["type"] == "user.prompt"][0]
    assert prompt["text"] == "please send this too"
    assert prompt["meta"]["event_type"] == "event.user.followup"
    assert prompt["meta"]["batch_id"] == "batch-followup"

    entries = iter_turn_user_input_entries(payload["blocks"], turn_id="turn-9")
    assert len(entries) == 1
    assert entries[0]["plain_text"] == "please send this too"
    assert entries[0]["user_event_type"] == "event.user.followup"
    assert entries[0]["attachments"][0]["label"] == "keep-the-agent-add-the-runtime.png"
    assert entries[0]["contexts"][0]["ref"] == "conv:ar:turn-1.artifact"


def test_minimal_turn_log_skips_empty_prompt_and_bad_refs():
    """Empty prompt → no user.prompt block; an attachment without a filename or a
    file without a ref is dropped (no malformed blocks that the reader can't use)."""
    payload = build_minimal_turn_log_payload(
        final_answer="Done.",
        turn_id="turn-9",
        user_prompt_text="   ",
        user_attachments=[{"mime": "text/plain"}],                    # no filename
        assistant_files=[{"filename": "x", "logical_path": ""}],      # no ref/path
    )
    types = {b["type"] for b in payload["blocks"]}
    assert "user.prompt" not in types
    assert "user.attachment.meta" not in types
    assert "react.tool.result" not in types
    assert types == {ASSISTANT_COMPLETION_BLOCK_TYPE}


def test_resolve_request_identity_is_svc_first_with_comm_fallback():
    from types import SimpleNamespace
    from kdcube_ai_app.apps.chat.sdk.event_identity import resolve_request_identity
    # svc carries the real user; comm.user_id is empty (the exact case that broke
    # file download) — owner must come from svc so write + read agree.
    comm = SimpleNamespace(
        service={"tenant": "t", "project": "p", "user": "alice",
                 "user_type": "registered", "conversation_id": "c1", "request_id": "r1"},
        tenant="", project="", user_id="", user_type="",
    )
    ident = resolve_request_identity(comm)
    assert ident == {"tenant": "t", "project": "p", "owner": "alice",
                     "user_type": "registered", "conversation_id": "c1", "request_id": "r1"}
    # Falls back to comm attributes when svc is empty.
    comm2 = SimpleNamespace(service={}, tenant="t2", project="p2", user_id="bob", user_type="reg")
    assert resolve_request_identity(comm2)["owner"] == "bob"
    assert resolve_request_identity(comm2)["tenant"] == "t2"


# ------------------------------------------------- conversation-title persist
#
# The conversation LIST reads the title from the per-conversation timeline
# artifact (kind conv.timeline.v1), NOT the turn log. The minimal recorder writes
# that artifact on a new conversation so a run-to-completion bundle's turns list
# titled, matching the React path's shape.


@pytest.mark.asyncio
async def test_records_conversation_title_timeline_artifact():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_conversation_timeline(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        conversation_title="  Weekend plans  ", agent_id="raw",
    )
    assert wrote is True
    assert len(client.artifact_calls) == 1
    call = client.artifact_calls[0]
    assert call["kind"] == TIMELINE_KIND
    assert call["conversation_id"] == "conv-1"
    # The conversation list json-parses content_str and reads conversation_title.
    parsed = json.loads(call["content_str"])
    assert parsed["conversation_title"] == "Weekend plans"
    # The stored payload also carries it (timeline shape).
    assert call["content"]["conversation_title"] == "Weekend plans"


@pytest.mark.asyncio
async def test_blank_title_still_registers_conversation():
    # No title -> the conversation is still registered (so it lists), title empty.
    client = _FakeConversationClient()
    wrote = await record_conversation_timeline(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        conversation_title="",
    )
    assert wrote is True
    assert len(client.artifact_calls) == 1
    assert client.artifact_calls[0]["kind"] == TIMELINE_KIND
    assert json.loads(client.artifact_calls[0]["content_str"])["conversation_title"] == ""


@pytest.mark.asyncio
async def test_later_blank_turn_carries_title_forward_and_refreshes():
    # A first titled turn registers the conversation; a later title-less turn
    # refreshes the timeline (advancing recency) while carrying the title forward
    # -- never shadowing the title with a blank one.
    client = _FakeConversationClient()
    await record_conversation_timeline(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        conversation_title="Weekend plans", conversation_started_at="2026-07-13T00:00:00Z",
    )
    wrote2 = await record_conversation_timeline(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-2", bundle_id="bundle.demo",
        conversation_title="",
    )
    assert wrote2 is True
    # Two timeline rows now (one per turn); the most recent still carries the title
    # and the original started_at (carried forward).
    assert len(client.artifact_calls) == 2
    latest = json.loads(client.artifact_calls[-1]["content_str"])
    assert latest["conversation_title"] == "Weekend plans"
    assert latest["conversation_started_at"] == "2026-07-13T00:00:00Z"
    assert client.artifact_calls[-1]["turn_id"] == "turn-2"


@pytest.mark.asyncio
async def test_blank_title_no_recent_still_registers():
    # A client without `recent` cannot read a prior timeline, but must still
    # register the conversation so it lists (title-less).
    class _NoRecent:
        def __init__(self):
            self.artifact_calls = []

        async def save_artifact(self, **kwargs):
            self.artifact_calls.append(kwargs)
            return {}

    client = _NoRecent()
    wrote = await record_conversation_timeline(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        conversation_title="",
    )
    assert wrote is True
    assert len(client.artifact_calls) == 1


@pytest.mark.asyncio
async def test_minimal_log_with_title_writes_both_artifacts():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        agent_id="raw", final_answer="Hello.", conversation_title="Weekend plans",
    )
    assert wrote is True
    # turn log written...
    assert len(client.calls) == 1
    assert client.calls[0]["payload"]["conversation_title"] == "Weekend plans"
    # ...AND the conversation-title timeline artifact.
    assert len(client.artifact_calls) == 1
    assert client.artifact_calls[0]["kind"] == TIMELINE_KIND


@pytest.mark.asyncio
async def test_minimal_log_without_title_registers_conversation():
    # A turn with no title still registers the conversation (so it lists); each
    # recorded turn refreshes the timeline (advancing recency, mirroring React).
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        agent_id="raw", final_answer="First answer.",
    )
    assert wrote is True
    assert len(client.calls) == 1
    assert "conversation_title" not in client.calls[0]["payload"]
    # The conversation was registered (title-less) so it appears in the list.
    assert len(client.artifact_calls) == 1
    assert client.artifact_calls[0]["kind"] == TIMELINE_KIND
    parsed = json.loads(client.artifact_calls[0]["content_str"])
    assert parsed["conversation_title"] == ""

    # A second no-title turn: turn log written AND the timeline refreshed.
    reset_turn_log_recorded()
    wrote2 = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-2", bundle_id="bundle.demo",
        agent_id="raw", final_answer="Second answer.",
    )
    assert wrote2 is True
    assert len(client.calls) == 2
    assert len(client.artifact_calls) == 2
    assert client.artifact_calls[-1]["turn_id"] == "turn-2"


@pytest.mark.asyncio
async def test_minimal_log_merges_same_turn_sibling_without_losing_user_text():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    attachment = {
        "filename": "keep-the-agent-add-the-runtime.png",
        "mime": "image/png",
        "hosted_uri": "s3://bucket/user/conv/turn/file.png",
        "rn": "rn:file",
    }
    wrote = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        agent_id="raw",
        final_answer="First answer.",
        user_prompt_text="please send this image to Slack",
        user_attachments=[attachment],
        batch_id="batch-1",
    )
    assert wrote is True

    # Defensive regression guard: if a same-turn sibling event ever reaches the
    # fallback again, the newer minimal payload must not replace the earlier one
    # that carried the user's text.
    reset_turn_log_recorded()
    wrote2 = await record_minimal_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        agent_id="raw",
        final_answer="Second answer.",
        user_prompt_text="",
        user_attachments=[attachment],
        batch_id="batch-1",
    )
    assert wrote2 is True

    blocks = client.calls[-1]["payload"]["blocks"]
    prompts = [b for b in blocks if b["type"] == "user.prompt"]
    attachments = [b for b in blocks if b["type"] == "user.attachment.meta"]
    completions = [b for b in blocks if b["type"] == ASSISTANT_COMPLETION_BLOCK_TYPE]
    assert [b["text"] for b in prompts] == ["please send this image to Slack"]
    assert len(attachments) == 1
    assert [b["text"] for b in completions] == ["First answer.", "Second answer."]


# ------------------------------------------------- entrypoint (BaseEntrypoint)
#
# The fallback lives on the app layer, not the orchestrator: BaseEntrypoint.run()
# resets the flag at turn start and calls _record_turn_log_fallback() after
# execute_core, using its own get_ctx_client(). These tests exercise that method
# directly on a minimal stand-in (only get_ctx_client + logger are used).


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def log(self, message, level="INFO"):
        if str(level).upper() == "WARNING":
            self.warnings.append(message)


class _FakeEntrypoint:
    """Minimal stand-in exposing only what _record_turn_log_fallback touches."""

    # Bind the real method under test.
    _record_turn_log_fallback = BaseEntrypoint._record_turn_log_fallback

    def __init__(self, client):
        self._client = client
        self.logger = _FakeLogger()

    async def get_ctx_client(self):
        return self._client


def _fallback_kwargs(*, final_answer="Answer.", steps=None):
    return dict(
        result={"final_answer": final_answer, "step_logs": steps or []},
        tenant="tenant-a", project="project-a", user_id="user-1",
        user_type="registered", thread_id="conv-1", turn_id="turn-1",
        bundle_id="bundle.demo", agent_id="raw",
    )


@pytest.mark.asyncio
async def test_entrypoint_records_for_non_react_turn():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    ep = _FakeEntrypoint(client)
    await ep._record_turn_log_fallback(**_fallback_kwargs(final_answer="Framework-neutral answer."))
    assert len(client.calls) == 1
    assert client.calls[0]["payload"]["blocks"][-1]["text"] == "Framework-neutral answer."
    assert client.calls[0]["agent_id"] == "raw"


@pytest.mark.asyncio
async def test_entrypoint_fallback_records_followup_text_attachments_and_context_events():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    ep = _FakeEntrypoint(client)
    state = {
        "external_events": [
            {
                "id": "fu1",
                "type": "event.user.followup",
                "batch_id": "batch-followup",
                "payload": {"event": {"text": "please send this too"}},
            },
            {
                "id": "file1",
                "type": "event.user.attachment.file",
                "batch_id": "batch-followup",
                "payload": {
                    "event": {
                        "filename": "keep-the-agent-add-the-runtime.png",
                        "mime": "image/png",
                        "hosted_uri": "s3://bucket/user/conv/turn/file.png",
                        "rn": "rn:file",
                    }
                },
            },
            {
                "id": "ctx1",
                "type": "event.user.context.object",
                "batch_id": "batch-followup",
                "payload": {"object_ref": "conv:ar:turn-1.artifact", "label": "runtime note"},
            },
        ]
    }
    kwargs = _fallback_kwargs(final_answer="Framework-neutral answer.")
    kwargs["state"] = state

    await ep._record_turn_log_fallback(**kwargs)

    blocks = client.calls[0]["payload"]["blocks"]
    prompt = [b for b in blocks if b["type"] == "user.prompt"][0]
    assert prompt["text"] == "please send this too"
    assert prompt["meta"]["event_type"] == "event.user.followup"
    assert prompt["meta"]["batch_id"] == "batch-followup"
    assert [b["type"] for b in blocks] == [
        "user.prompt",
        "event.user.context.object",
        "user.attachment.meta",
        ASSISTANT_COMPLETION_BLOCK_TYPE,
    ]


@pytest.mark.asyncio
async def test_entrypoint_no_double_write_when_react_already_recorded():
    # React wrote its own rich log this turn -> flag set -> fallback is a no-op.
    reset_turn_log_recorded()
    mark_turn_log_recorded()
    client = _FakeConversationClient()
    ep = _FakeEntrypoint(client)
    await ep._record_turn_log_fallback(**_fallback_kwargs(final_answer="React answer."))
    assert client.calls == []


@pytest.mark.asyncio
async def test_entrypoint_skips_empty_answer():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    ep = _FakeEntrypoint(client)
    await ep._record_turn_log_fallback(**_fallback_kwargs(final_answer="   "))
    assert client.calls == []


@pytest.mark.asyncio
async def test_entrypoint_no_ctx_client_is_noop():
    reset_turn_log_recorded()
    ep = _FakeEntrypoint(None)  # get_ctx_client() -> None (no pg_pool available)
    await ep._record_turn_log_fallback(**_fallback_kwargs())
    assert ep.logger.warnings == []  # a missing client is not an error


@pytest.mark.asyncio
async def test_entrypoint_recording_failure_never_fails_turn():
    class _BoomClient:
        async def save_turn_log_as_artifact(self, **kwargs):
            raise RuntimeError("store down")

    reset_turn_log_recorded()
    ep = _FakeEntrypoint(_BoomClient())
    # Recording raises internally but is swallowed and logged; no exception escapes.
    await ep._record_turn_log_fallback(**_fallback_kwargs(final_answer="Answer survives."))
    assert ep.logger.warnings, "a recording failure should be logged as a warning"


def test_reset_clears_prior_turn_mark():
    # run() calls reset_turn_log_recorded() at each turn's start; prove the flag
    # scoping the fallback depends on is reset/mark/reset as expected.
    reset_turn_log_recorded()
    assert turn_log_was_recorded() is False
    assert turn_log_recording_kind() == TURN_LOG_RECORDING_NONE
    mark_turn_log_recorded()
    assert turn_log_was_recorded() is True
    assert turn_log_recording_kind() == TURN_LOG_RECORDING_RICH
    assert rich_turn_log_was_recorded() is True
    reset_turn_log_recorded()
    assert turn_log_was_recorded() is False


def test_rich_recording_mark_cannot_be_downgraded_by_minimal_writer():
    reset_turn_log_recorded()
    mark_turn_log_recorded(TURN_LOG_RECORDING_RICH)
    mark_turn_log_recorded(TURN_LOG_RECORDING_MINIMAL)

    assert turn_log_recording_kind() == TURN_LOG_RECORDING_RICH
    assert rich_turn_log_was_recorded() is True


# ------------------------------------------------------- failed-turn recording


def test_build_error_turn_log_payload_marks_error():
    payload = build_error_turn_log_payload(
        error_message="bind_tools not implemented", turn_id="turn-e", error_type="NotImplementedError",
    )
    assert payload["blocks_count"] == len(payload["blocks"]) == 1
    block = payload["blocks"][0]
    assert block["type"] == ASSISTANT_COMPLETION_BLOCK_TYPE
    assert block["text"] == "bind_tools not implemented"
    assert block["meta"] == {"error": True, "error_type": "NotImplementedError"}


def test_build_error_turn_log_payload_defaults_blank_message():
    payload = build_error_turn_log_payload(error_message="   ", turn_id="turn-e")
    assert payload["blocks"][0]["text"] == "An error occurred."


@pytest.mark.asyncio
async def test_records_error_log_for_crashed_turn():
    reset_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_error_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        agent_id="raw", error_message="boom", error_type="RuntimeError",
    )
    assert wrote is True
    block = client.calls[0]["payload"]["blocks"][-1]
    assert block["text"] == "boom"
    assert block["meta"]["error"] is True


@pytest.mark.asyncio
async def test_error_log_no_double_write_when_already_recorded():
    reset_turn_log_recorded()
    mark_turn_log_recorded()
    client = _FakeConversationClient()
    wrote = await record_error_turn_log_if_absent(
        conversation_client=client,
        tenant="t", project="p", user="u", user_type="registered",
        conversation_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo",
        error_message="boom",
    )
    assert wrote is False
    assert client.calls == []


def test_error_surfaced_flag_reset_mark():
    reset_turn_error_surfaced()
    assert turn_error_was_surfaced() is False
    mark_turn_error_surfaced()
    assert turn_error_was_surfaced() is True
    reset_turn_error_surfaced()
    assert turn_error_was_surfaced() is False


# ----------------------------------------- _surface_turn_failure (backstop)


class _CaptureComm:
    def __init__(self):
        self.error_calls = []
        self.step_calls = []

    async def error(self, **kwargs):
        self.error_calls.append(kwargs)

    async def step(self, **kwargs):
        self.step_calls.append(kwargs)


class _FailureEntrypoint:
    """Stand-in binding the real backstop methods over minimal collaborators."""

    _surface_turn_failure = BaseEntrypoint._surface_turn_failure
    _record_failed_turn_log = BaseEntrypoint._record_failed_turn_log
    report_turn_error = BaseEntrypoint.report_turn_error

    def __init__(self, client):
        self._client = client
        self.logger = _FakeLogger()
        self._comm = _CaptureComm()
        self.comm_context = None

    @property
    def comm(self):
        return self._comm

    async def get_ctx_client(self):
        return self._client


def _failure_kwargs():
    return dict(
        tenant="t", project="p", user_id="u1", user_type="registered",
        thread_id="conv-1", turn_id="turn-1", bundle_id="bundle.demo", agent_id="raw",
    )


@pytest.mark.asyncio
async def test_surface_turn_failure_emits_and_records():
    reset_turn_log_recorded()
    reset_turn_error_surfaced()
    client = _FakeConversationClient()
    ep = _FailureEntrypoint(client)
    state = {}
    await ep._surface_turn_failure(state=state, exc=RuntimeError("boom"), **_failure_kwargs())
    # client saw a user-visible chat.error
    assert ep._comm.error_calls and ep._comm.error_calls[0]["message"] == "boom"
    # a FAILED turn log was recorded (error-marked)
    assert len(client.calls) == 1
    assert client.calls[0]["payload"]["blocks"][-1]["meta"]["error"] is True
    # the per-turn flag is set so a second backstop pass is inert
    assert turn_error_was_surfaced() is True


@pytest.mark.asyncio
async def test_surface_turn_failure_inert_when_already_surfaced():
    # React path: the workflow already emitted its chat.error + rolled the turn
    # back and marked the flag -> the backstop must not double-emit or re-record.
    reset_turn_log_recorded()
    reset_turn_error_surfaced()
    mark_turn_error_surfaced()
    client = _FakeConversationClient()
    ep = _FailureEntrypoint(client)
    await ep._surface_turn_failure(state={}, exc=RuntimeError("boom"), **_failure_kwargs())
    assert ep._comm.error_calls == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_surface_turn_failure_skips_economics_limit():
    reset_turn_log_recorded()
    reset_turn_error_surfaced()
    client = _FakeConversationClient()
    ep = _FailureEntrypoint(client)
    await ep._surface_turn_failure(
        state={}, exc=EconomicsLimitException("rate", code="rate"), **_failure_kwargs()
    )
    assert ep._comm.error_calls == []
    assert client.calls == []
    assert turn_error_was_surfaced() is False


def test_a_pinned_object_reloads_as_a_live_chip():
    """LIVE (press, Claude Code lane): a post pinned to the message showed as a
    chip while the turn ran and was gone when the conversation was reopened —
    the recorded turn kept the typed text and dropped the object.

    A context object is an EVENT, not a file and not a sentence: recorded
    verbatim, the reload reader rebuilds the chip from the event's own
    `object_ref`, so clicking it resolves through the same resolver the live
    chip used."""
    from kdcube_ai_app.apps.chat.sdk.solutions.conversation.record import (
        build_minimal_turn_log_payload,
    )
    from kdcube_ai_app.apps.chat.sdk.runtime.user_inputs import (
        iter_turn_user_input_entries,
    )

    ref = "linkedin:kdcube:post:industry/2026.08.13-keep-agent-add-runtime"
    event = {
        "id": "e1",
        "type": "event.user.context.object",
        "batch_id": "b1",
        "ts": "2026-08-14T16:38:00Z",
        "payload": {"object_ref": ref, "label": "2026.08.13-keep-agent-add-runtime"},
    }
    payload = build_minimal_turn_log_payload(
        final_answer="…",
        turn_id="t1",
        user_prompt_text="whats this ?",
        user_events=[event],
        batch_id="b1",
    )
    types = [b["type"] for b in payload["blocks"]]
    assert "event.user.context.object" in types

    entries = iter_turn_user_input_entries(payload["blocks"], turn_id="t1")
    assert len(entries) == 1
    contexts = entries[0].get("contexts") or []
    assert [c.get("ref") for c in contexts] == [ref]
    # the event survives verbatim, so anything the resolver needs is still there
    recorded = [b for b in payload["blocks"] if b["type"] == "event.user.context.object"][0]
    assert recorded["meta"]["event"] == event


def test_the_prompt_and_its_objects_share_one_batch():
    """The reader groups a user's message with its objects by batch id; without
    the shared id the chip records but never attaches to the bubble."""
    from kdcube_ai_app.apps.chat.sdk.solutions.conversation.record import (
        build_minimal_turn_log_payload,
    )

    payload = build_minimal_turn_log_payload(
        final_answer="…",
        turn_id="t9",
        user_prompt_text="look",
        user_events=[{"id": "e", "type": "event.user.context.object",
                      "payload": {"object_ref": "conv:ar:x"}}],
        batch_id="",          # none supplied: one is derived, still shared
    )
    batches = {b["meta"].get("batch_id") for b in payload["blocks"] if b["type"].startswith(("user.prompt", "event."))}
    assert len(batches) == 1 and next(iter(batches))


def test_a_prompt_event_is_not_recorded_twice_as_a_chip():
    """The prompt itself is already the bubble; it must not also render as an
    object attached to itself."""
    from kdcube_ai_app.apps.chat.sdk.solutions.conversation.record import (
        build_minimal_turn_log_payload,
    )

    payload = build_minimal_turn_log_payload(
        final_answer="…",
        turn_id="t2",
        user_prompt_text="hello",
        user_events=[{"id": "p", "type": "event.user.prompt", "payload": {"text": "hello"}}],
        batch_id="b",
    )
    assert [b["type"] for b in payload["blocks"]] == ["user.prompt", "assistant.completion"]


# ── several messages in one turn (2026-08-18) ────────────────────────────────


def test_every_folded_message_reloads_as_its_own_bubble():
    """A turn folds the messages that queued while the previous one was
    running. Each was its own bubble when it was sent, so each has to be its
    own bubble on reload — recording only the first would make them vanish
    after having been read live.
    """
    from kdcube_ai_app.apps.chat.sdk.solutions.conversation.record import (
        user_messages_from_folded_events,
    )

    events = [
        {"type": "event.user.prompt", "payload": {"event": {"text": "do X"}},
         "_kdcube_lane_batch_id": "b1"},
        {"type": "event.user.followup", "payload": {"event": {"text": "actually Y"}},
         "_kdcube_lane_batch_id": "b2"},
        {"type": "event.user.followup", "payload": {"event": {"text": "and Z"}},
         "_kdcube_lane_batch_id": "b3"},
    ]
    messages = user_messages_from_folded_events(events)
    payload = build_minimal_turn_log_payload(
        final_answer="Answer", turn_id="turn-9", user_messages=messages,
    )
    prompts = [b for b in payload["blocks"] if b.get("type") == "user.prompt"]

    assert [b["text"] for b in prompts] == ["do X", "actually Y", "and Z"]
    # Distinct identities, or the reader collapses them into one entry.
    assert [b["meta"]["message_id"] for b in prompts] == ["m0", "m1", "m2"]
    assert len({b["path"] for b in prompts}) == 3
    # A follow-up reloads AS a follow-up, per message and not per turn.
    assert [b["meta"]["event_type"] for b in prompts] == [
        "event.user.prompt", "event.user.followup", "event.user.followup",
    ]
    # And each keeps the send it belonged to, so its chips land on it.
    assert [b["meta"]["batch_id"] for b in prompts] == ["b1", "b2", "b3"]


def test_one_message_records_exactly_as_before():
    """The single-message turn — every turn until now — is untouched: one
    block, `m0`, the unsuffixed artifact path."""
    payload = build_minimal_turn_log_payload(
        final_answer="Answer", turn_id="turn-9",
        user_prompt_text="just the one", batch_id="b1",
    )
    prompts = [b for b in payload["blocks"] if b.get("type") == "user.prompt"]

    assert len(prompts) == 1
    assert prompts[0]["meta"]["message_id"] == "m0"
    assert prompts[0]["path"] == "conv:ar:turn-9.user.prompt"
    assert prompts[0]["meta"]["batch_id"] == "b1"


def test_each_message_keeps_the_time_it_was_sent():
    """A folded turn records several messages in one write. Stamping them all
    with the record moment puts them in the timeline where the ANSWER is,
    rather than where they were typed — so each block carries the arrival time
    the lane recorded, and only falls back to the write moment when the lane
    did not know one.
    """
    from kdcube_ai_app.apps.chat.sdk.solutions.conversation.record import (
        user_messages_from_folded_events,
    )

    events = [
        {"type": "event.user.prompt", "payload": {"event": {"text": "do X"}},
         "_kdcube_lane_batch_id": "b1", "_kdcube_lane_ts": "2026-08-18T00:00:01Z"},
        {"type": "event.user.followup", "payload": {"event": {"text": "actually Y"}},
         "_kdcube_lane_batch_id": "b2", "_kdcube_lane_ts": "2026-08-18T00:00:09Z"},
    ]
    payload = build_minimal_turn_log_payload(
        final_answer="Answer", turn_id="turn-9",
        user_messages=user_messages_from_folded_events(events),
        ts="2026-08-18T00:05:00Z",   # the write moment: much later
    )
    prompts = [b for b in payload["blocks"] if b.get("type") == "user.prompt"]

    assert [b["ts"] for b in prompts] == [
        "2026-08-18T00:00:01Z", "2026-08-18T00:00:09Z",
    ]


def test_a_message_the_lane_never_timed_keeps_the_write_moment():
    """The fallback: an unstamped event records exactly as it always did."""
    payload = build_minimal_turn_log_payload(
        final_answer="Answer", turn_id="turn-9",
        user_prompt_text="just the one", ts="2026-08-18T00:05:00Z",
    )
    prompts = [b for b in payload["blocks"] if b.get("type") == "user.prompt"]

    assert prompts[0]["ts"] == "2026-08-18T00:05:00Z"
