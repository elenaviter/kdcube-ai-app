# SPDX-License-Identifier: MIT

import logging

from kdcube_ai_app.apps.chat.sdk.viz.logging_helpers import log_agent_packet, log_raw_channel_output


def test_log_agent_packet_always_writes_plain_log(caplog):
    packet = {
        "internal_thinking": "thinking text",
        "user_thinking": "user-facing text",
        "agent_response": {
            "action": "call_tool",
            "tool_call": {
                "tool_id": "email.materialize_email_attachments",
                "params": {"message_ids_json": "[\"m1\"]"},
            },
        },
    }

    with caplog.at_level(logging.INFO, logger="agents"):
        log_agent_packet("solver.react.v2.decision.v2.strong", "react.decision.v2", packet)

    text = caplog.text
    assert "[agent.packet] agent=solver.react.v2.decision.v2.strong phase=react.decision.v2" in text
    assert "Internal thinking:" in text
    assert "thinking text" in text
    assert "Structured response:" in text
    assert "email.materialize_email_attachments" in text


def test_log_raw_channel_output_writes_exact_channel_transcript(caplog):
    raw = (
        "<channel:thinking>status</channel:thinking>"
        "<channel:ReactDecisionOutV2>{\"action\":\"call_tool\"}</channel:ReactDecisionOutV2>"
    )

    with caplog.at_level(logging.INFO, logger="agents"):
        log_raw_channel_output("solver.react.v2.decision.v2.strong", "react.decision.v3", raw)

    text = caplog.text
    assert "[agent.raw_channels] agent=solver.react.v2.decision.v2.strong phase=react.decision.v3" in text
    assert "--- raw model output begin ---" in text
    assert raw in text
    assert "--- raw model output end ---" in text
