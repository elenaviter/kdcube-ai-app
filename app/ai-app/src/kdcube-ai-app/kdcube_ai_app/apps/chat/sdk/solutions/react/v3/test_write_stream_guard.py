# SPDX-License-Identifier: MIT

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.react.proto import RuntimeCtx
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.tests.helpers import FakeBrowser
from kdcube_ai_app.apps.chat.sdk.solutions.react.v3.action_overseer import ActionStreamGate
from kdcube_ai_app.apps.chat.sdk.solutions.react.v3.write_stream_guard import ReactWriteStreamGuard
from kdcube_ai_app.apps.chat.sdk.streaming.stream_policy import StreamPolicyViolation


async def _discard_delta(**_kwargs):
    return None


@pytest.mark.asyncio
async def test_write_stream_guard_rejects_visible_current_turn_path():
    runtime = RuntimeCtx(
        turn_id="turn_cur",
        conversation_id="conversation_1",
        outdir="/tmp/out",
        workdir="/tmp/out",
    )
    ctx = FakeBrowser(runtime)
    ctx.contribute(
        [
            {
                "turn": "turn_cur",
                "type": "react.tool.result",
                "path": "conv:fi:conv_conversation_1.turn_cur.files/report.md",
                "text": "first version",
            }
        ]
    )
    action_gate = ActionStreamGate(emit_delta=_discard_delta, action_index=0)
    answer_gate = ActionStreamGate(emit_delta=_discard_delta, action_index=0, lane="final_answer")
    guard = ReactWriteStreamGuard(
        ctx_browser=ctx,
        action_gate=action_gate,
        answer_gate=answer_gate,
        action_index=0,
    )

    with pytest.raises(StreamPolicyViolation) as exc:
        await guard.observe_path("turn_cur/files/report.md")

    assert exc.value.code == "write_path_already_exists"
    assert exc.value.extra["path"] == "turn_cur/files/report.md"
    assert exc.value.extra["artifact_path"] == "conv:fi:conv_conversation_1.turn_cur.files/report.md"
    assert action_gate.status == "denied"
    assert answer_gate.status == "denied"


@pytest.mark.asyncio
async def test_write_stream_guard_allows_new_current_turn_path():
    runtime = RuntimeCtx(
        turn_id="turn_cur",
        conversation_id="conversation_1",
        outdir="/tmp/out",
        workdir="/tmp/out",
    )
    ctx = FakeBrowser(runtime)
    action_gate = ActionStreamGate(emit_delta=_discard_delta, action_index=0)
    answer_gate = ActionStreamGate(emit_delta=_discard_delta, action_index=0, lane="final_answer")
    guard = ReactWriteStreamGuard(
        ctx_browser=ctx,
        action_gate=action_gate,
        answer_gate=answer_gate,
        action_index=0,
    )

    await guard.observe_path("turn_cur/files/report.md")

    assert action_gate.status == "pending"
    assert answer_gate.status == "pending"
