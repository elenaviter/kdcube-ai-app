# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter
#
# Unit tests for the reusable conversation-title utility.

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from kdcube_ai_app.apps.chat.sdk.tools.backends.summary import conversation_title as ct
from kdcube_ai_app.apps.chat.sdk.streaming.workspace_streamer import ChannelResult


def _channel_result(raw: str, obj=None) -> ChannelResult:
    return ChannelResult(
        raw=raw,
        obj=obj,
        used_sources=[],
        started_at=None,
        finished_at=None,
        error=None,
    )


def _make_streamer(*, output_raw="", output_obj=None, thinking="", service_error=None,
                   capture=None, stop_reason=None):
    """Build a fake stream_with_channels that also fires the thinking emit."""
    async def _fake(svc, *, messages, role, channels, emit, agent, max_tokens, temperature, return_full_raw, **kw):
        if capture is not None:
            capture.update(
                role=role, agent=agent, max_tokens=max_tokens,
                temperature=temperature, messages=messages,
            )
        # Emulate the streamer forwarding a thinking delta to the caller.
        if thinking:
            await emit(channel="thinking", text=thinking, completed=True)
        results = {
            "thinking": _channel_result(thinking),
            "output": _channel_result(output_raw, output_obj),
        }
        meta = {"service_error": service_error} if service_error else {}
        if stop_reason:
            meta["stop_reason"] = stop_reason
        return results, meta
    return _fake


def test_returns_parsed_title(monkeypatch):
    obj = ct.TitleOut(conversation_title="Trip to Rome")
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw='{"conversation_title": "Trip to Rome"}', output_obj=obj))
    title = asyncio.run(ct.generate_conversation_title(object(), user_message="Help me plan a trip to Rome"))
    assert title == "Trip to Rome"


def test_thinking_callback_fires(monkeypatch):
    obj = ct.TitleOut(conversation_title="A Title")
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw='{"conversation_title": "A Title"}', output_obj=obj,
                                       thinking="Thinking about a good name..."))
    seen = []

    async def _on_thinking(*, text, completed):
        seen.append((text, completed))

    title = asyncio.run(ct.generate_conversation_title(
        object(), user_message="hi", on_thinking_delta=_on_thinking))
    assert title == "A Title"
    assert seen == [("Thinking about a good name...", True)]


def test_direct_call_no_ctx_browser_builds_human_message(monkeypatch):
    capture = {}
    obj = ct.TitleOut(conversation_title="X")
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw='{"conversation_title":"X"}', output_obj=obj, capture=capture))
    title = asyncio.run(ct.generate_conversation_title(
        object(), user_message="What is the weather?", answer="It is sunny."))
    assert title == "X"
    # A human message was constructed from user_message (+ answer): no ctx_browser needed.
    human = capture["messages"][-1]
    blocks = human.additional_kwargs["message_blocks"]
    joined = " ".join(b.get("text", "") for b in blocks)
    assert "What is the weather?" in joined
    assert "It is sunny." in joined


def test_empty_output_yields_empty_title(monkeypatch):
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw="", output_obj=None))
    title = asyncio.run(ct.generate_conversation_title(object(), user_message="anything"))
    assert title == ""


def test_malformed_output_fails_open(monkeypatch):
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw="{not json", output_obj=None))
    payload, channels = asyncio.run(ct.run_conversation_title(object(), user_message="anything"))
    assert payload["conversation_title"] == ""
    assert channels["output"] == "{not json"


def test_output_model_override_preserves_extra_fields(monkeypatch):
    class GateOut(BaseModel):
        conversation_title: str | None = Field(default=None)
        route: str | None = Field(default=None)

    obj = GateOut(conversation_title="T", route="plan")
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw='{"conversation_title":"T","route":"plan"}', output_obj=obj))
    payload, _ = asyncio.run(ct.run_conversation_title(
        object(), user_message="q", output_model=GateOut))
    assert payload["conversation_title"] == "T"
    assert payload["route"] == "plan"


def test_defaults_role_and_temperature(monkeypatch):
    capture = {}
    obj = ct.TitleOut(conversation_title="Y")
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw='{"conversation_title":"Y"}', output_obj=obj, capture=capture))
    asyncio.run(ct.generate_conversation_title(object(), user_message="q"))
    assert capture["role"] == "gate.simple"
    assert capture["temperature"] == pytest.approx(0.2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_the_budget_covers_the_thinking_channel_too(monkeypatch):
    """LIVE: press conversations kept listing as "Untitled conversation".

    The title is emitted AFTER the thinking channel the prompt asks for, and the
    two shared a 128-token ceiling — so a chatty model spent the budget thinking
    and the response was cut off before the title. It failed intermittently, on
    the same question, which is what made it read as something that worked and
    then stopped working: the calls that failed spent exactly 128 output tokens.
    """
    capture: dict = {}
    obj = ct.TitleOut(conversation_title="A Title")
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw='{"conversation_title": "A Title"}',
                                       output_obj=obj, capture=capture))
    asyncio.run(ct.generate_conversation_title(object(), user_message="Anything"))
    assert capture["max_tokens"] >= 400


def test_a_title_cut_off_by_the_ceiling_says_so(monkeypatch):
    """An empty title is the same silence whether the model had nothing to say
    or we stopped it mid-sentence. Only one of those is ours to fix, so the log
    has to tell them apart."""
    monkeypatch.setattr(ct, "stream_with_channels",
                        _make_streamer(output_raw="", thinking="Considering names for this",
                                       stop_reason="max_tokens"))
    import logging
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Catch()
    ct.LOGGER.addHandler(handler)
    try:
        title = asyncio.run(ct.generate_conversation_title(object(), user_message="Anything"))
    finally:
        ct.LOGGER.removeHandler(handler)

    assert title == ""
    cut_off = [r for r in records if "CUT OFF" in r.getMessage()]
    assert cut_off and cut_off[0].levelno == logging.WARNING
