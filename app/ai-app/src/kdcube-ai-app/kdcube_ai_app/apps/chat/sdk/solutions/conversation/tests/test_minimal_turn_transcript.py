# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.conversation.ctx_rag import (
    CONVERSATION_TRANSCRIPT_INDEX_LABEL,
    ContextRAGClient,
    MINIMAL_TURN_TRANSCRIPT_TAG,
)
from kdcube_ai_app.apps.chat.sdk.solutions.conversation.record import (
    TURN_LOG_RECORDING_MINIMAL,
    build_minimal_turn_log_payload,
    reset_turn_log_recorded,
    rich_turn_log_was_recorded,
    turn_log_recording_kind,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.contributions import (
    stage_turn_summary,
    staged_turn_summary,
)


class _Index:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def add_message(self, **kwargs) -> int:
        self.rows.append(kwargs)
        return len(self.rows)


class _Store:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def put_message(self, **kwargs):
        self.rows.append(kwargs)
        return "store://turn.log", "turn-log-message", "turn-log-rn"


class _Model:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts):
        values = list(texts)
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[float(i + 1), 0.5] for i in range(len(values))]


def _payload() -> dict:
    return build_minimal_turn_log_payload(
        final_answer="The cropped image is ready.",
        turn_id="turn-1",
        user_messages=[
            {
                "text": "Crop the image",
                "event_type": "event.user.prompt",
                "batch_id": "batch-1",
                "ts": "2026-08-20T10:00:00Z",
            },
            {
                "text": "Use a square crop",
                "event_type": "event.user.followup",
                "batch_id": "batch-2",
                "ts": "2026-08-20T10:00:01Z",
            },
        ],
        ts="2026-08-20T10:00:02Z",
    )


@pytest.mark.asyncio
async def test_minimal_turn_log_writes_separate_searchable_role_rows() -> None:
    reset_turn_log_recorded()
    idx = _Index()
    store = _Store()
    model = _Model()
    client = ContextRAGClient(conv_idx=idx, store=store, model_service=model)

    await client.save_turn_log_as_artifact(
        tenant="tenant",
        project="project",
        user="user-1",
        conversation_id="conversation-1",
        user_type="registered",
        turn_id="turn-1",
        bundle_id="agent-app@1",
        agent_id="lg-react",
        payload=_payload(),
        recording_kind=TURN_LOG_RECORDING_MINIMAL,
        index_transcript=True,
    )

    assert [row["role"] for row in idx.rows] == ["artifact", "user", "user", "assistant"]
    artifact, first_user, second_user, assistant = idx.rows
    assert artifact["embedding"] is None
    assert "Crop the image" not in artifact["text"]
    assert "Crop the image" in first_user["text"]
    assert "Use a square crop" in second_user["text"]
    assert assistant["text"] == "The cropped image is ready."
    assert first_user["user_type"] == CONVERSATION_TRANSCRIPT_INDEX_LABEL
    assert assistant["user_type"] == CONVERSATION_TRANSCRIPT_INDEX_LABEL
    assert MINIMAL_TURN_TRANSCRIPT_TAG in first_user["tags"]
    assert "event_type:event.user.followup" in second_user["tags"]
    assert all(row["hosted_uri"] == "index_only" for row in idx.rows[1:])
    assert all(row["embedding"] is not None for row in idx.rows[1:])
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 3
    assert turn_log_recording_kind() == TURN_LOG_RECORDING_MINIMAL
    assert rich_turn_log_was_recorded() is False


@pytest.mark.asyncio
async def test_embedding_failure_keeps_lexical_and_trigram_transcript_rows() -> None:
    idx = _Index()
    client = ContextRAGClient(
        conv_idx=idx,
        store=_Store(),
        model_service=_Model(fail=True),
    )

    persisted = await client.save_minimal_turn_transcript_rows(
        user="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        bundle_id="agent-app@1",
        agent_id="press",
        payload=_payload(),
    )

    assert persisted == 3
    assert [row["role"] for row in idx.rows] == ["user", "user", "assistant"]
    assert all(row["embedding"] is None for row in idx.rows)
    assert all(row["text"] for row in idx.rows)


@pytest.mark.asyncio
async def test_staged_turn_summary_enters_turn_log_and_search_projection() -> None:
    state = {"turn_id": "turn-1"}
    first = stage_turn_summary(
        state,
        summary="Initial draft",
        phrases=["square image"],
    )
    second = stage_turn_summary(
        state,
        summary="The image was cropped to a reusable square asset.",
        refs=["conv:fi:conv_c1.turn_1.files/crop.png"],
        phrases=["square crop", "square crop"],
        entities=["crop.png"],
        contributor="langgraph",
    )
    assert first["replaced"] is False
    assert second["replaced"] is True

    contribution = staged_turn_summary(state, turn_id="turn-1")
    payload = build_minimal_turn_log_payload(
        final_answer="The cropped image is ready.",
        turn_id="turn-1",
        user_prompt_text="Crop the image",
        turn_summary_contribution=contribution,
        ts="2026-08-20T10:00:02Z",
    )
    summary_block = next(
        block for block in payload["blocks"]
        if block.get("type") == "conv.working.summary"
    )
    assert summary_block["path"] == "conv:ws:turn-1.conv.working.summary.attempt.1"
    assert "conv:fi:conv_c1.turn_1.files/crop.png" in summary_block["text"]
    assert "Retrieval-anchors:" in summary_block["text"]
    assert summary_block["meta"]["contributor"] == "langgraph"

    idx = _Index()
    model = _Model()
    client = ContextRAGClient(conv_idx=idx, store=_Store(), model_service=model)
    persisted = await client.save_minimal_turn_transcript_rows(
        user="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        bundle_id="agent-app@1",
        agent_id="lg-react",
        payload=payload,
    )

    assert persisted == 3
    assert [row["role"] for row in idx.rows] == ["user", "assistant", "assistant"]
    summary_row = idx.rows[1]
    assert "chat:summary" in summary_row["tags"]
    assert "kind:working.summary" in summary_row["tags"]
    assert "summary_scope:turn" in summary_row["tags"]
    assert summary_row["anchors_text"] == '"square crop" crop.png'
    assert all(row["embedding"] is not None for row in idx.rows)


def test_staged_turn_summary_is_scoped_to_its_trusted_turn() -> None:
    state = {"turn_id": "turn-1"}
    stage_turn_summary(state, summary="Turn one")
    assert staged_turn_summary(state, turn_id="turn-1") is not None
    assert staged_turn_summary(state, turn_id="turn-2") is None
    with pytest.raises(ValueError, match="must not be empty"):
        stage_turn_summary(state, summary="  ")


@pytest.mark.asyncio
async def test_transcript_projection_failure_does_not_undo_persisted_turn_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_turn_log_recorded()
    idx = _Index()
    store = _Store()
    client = ContextRAGClient(conv_idx=idx, store=store, model_service=_Model())

    async def _fail_projection(**_kwargs) -> int:
        raise RuntimeError("projection failed")

    monkeypatch.setattr(client, "save_minimal_turn_transcript_rows", _fail_projection)

    saved = await client.save_turn_log_as_artifact(
        tenant="tenant",
        project="project",
        user="user-1",
        conversation_id="conversation-1",
        user_type="registered",
        turn_id="turn-1",
        bundle_id="agent-app@1",
        agent_id="lg-react",
        payload=_payload(),
        recording_kind=TURN_LOG_RECORDING_MINIMAL,
        index_transcript=True,
    )

    assert saved["hosted_uri"] == "store://turn.log"
    assert [row["role"] for row in idx.rows] == ["artifact"]
    assert turn_log_recording_kind() == TURN_LOG_RECORDING_MINIMAL
