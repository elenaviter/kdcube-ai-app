from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.channels import (
    DirectInputAttachment,
    DirectTurnRequest,
    DirectTurnResult,
    add_direct_input_attachments,
    completed_direct_turn_result,
    prompt_with_attachment_manifest,
    run_terminal_chat,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (
    ConfiguredAgentInput,
)


def test_attachment_from_path_reads_bytes_and_detects_mime(tmp_path) -> None:
    source = tmp_path / "request.txt"
    source.write_bytes(b"hello")

    attachment = DirectInputAttachment.from_path(source)

    assert attachment.filename == "request.txt"
    assert attachment.mime == "text/plain"
    assert attachment.content == b"hello"


@pytest.mark.asyncio
async def test_input_attachments_are_materialized_with_collision_safe_names() -> None:
    turn = SimpleNamespace(
        add_user_attachment_bytes=AsyncMock(side_effect=[{"n": 1}, {"n": 2}])
    )
    workspace = SimpleNamespace(
        current_attachment=lambda name: f"/tmp/current/attachments/{name}"
    )

    rows = await add_direct_input_attachments(
        turn=turn,
        workspace=workspace,
        attachments=(
            DirectInputAttachment("report.txt", "text/plain", b"one"),
            DirectInputAttachment("report.txt", "text/plain", b"two"),
        ),
    )

    assert rows == [{"n": 1}, {"n": 2}]
    first, second = turn.add_user_attachment_bytes.await_args_list
    assert first.kwargs["filename"] == "report.txt"
    assert second.kwargs["filename"] == "report-2.txt"


@pytest.mark.asyncio
async def test_input_attachments_can_be_mirrored_for_provider_native_workspace(
    tmp_path,
) -> None:
    turn = SimpleNamespace(
        add_user_attachment_bytes=AsyncMock(return_value={"ok": True})
    )
    workspace = SimpleNamespace(
        current_attachment=lambda name: tmp_path / "durable" / name
    )

    await add_direct_input_attachments(
        turn=turn,
        workspace=workspace,
        attachments=(DirectInputAttachment("input.png", "image/png", b"image"),),
        mirror_to=tmp_path / "provider" / "attachments",
    )

    assert (
        tmp_path / "provider" / "attachments" / "input.png"
    ).read_bytes() == b"image"


def test_attachment_manifest_names_provider_native_workspace_paths() -> None:
    prompt = prompt_with_attachment_manifest(
        "Inspect the files.",
        (
            {"filename": "input.png", "mime": "image/png"},
            {"filename": "notes.txt", "mime": "text/plain"},
        ),
    )

    assert prompt.startswith("Inspect the files.")
    assert "`attachments/input.png` (image/png)" in prompt
    assert "`attachments/notes.txt` (text/plain)" in prompt


@pytest.mark.asyncio
async def test_completed_result_uses_the_persisted_turn_payload() -> None:
    harness = SimpleNamespace(
        verify_conversation=AsyncMock(
            return_value=[
                {
                    "turn_id": "turn-1",
                    "payload": {"blocks": [{"type": "assistant.completion"}]},
                }
            ]
        )
    )

    result = await completed_direct_turn_result(
        harness=harness,
        conversation_id="conversation-1",
        turn_id="turn-1",
        answer="done",
        metadata={"source": "terminal"},
    )

    assert result.answer == "done"
    assert result.turn_log["blocks"][0]["type"] == "assistant.completion"
    assert result.transport_payload()["source"] == "terminal"


@pytest.mark.asyncio
async def test_terminal_chat_reuses_the_configured_identity() -> None:
    lines = iter(("first", "second", "/exit"))
    writes: list[str] = []
    requests: list[DirectTurnRequest] = []

    async def run_turn(request: DirectTurnRequest) -> DirectTurnResult:
        requests.append(request)
        return DirectTurnResult(
            answer=f"answer-{len(requests)}",
            turn_id=f"turn-{len(requests)}",
            turn_log={"blocks": []},
        )

    await run_terminal_chat(
        agent_input=ConfiguredAgentInput(
            user_id="alice",
            user_type="regular",
            session_id="terminal-1",
            conversation_id="research",
        ),
        run_turn=run_turn,
        read_line=lambda _prompt: next(lines),
        write_line=writes.append,
    )

    assert [request.prompt for request in requests] == ["first", "second"]
    assert {(request.user_id, request.conversation_id) for request in requests} == {
        ("alice", "research")
    }
    assert any("assistant> answer-2" in line for line in writes)
