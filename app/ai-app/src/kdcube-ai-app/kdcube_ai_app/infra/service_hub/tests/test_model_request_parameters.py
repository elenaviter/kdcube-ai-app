# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from kdcube_ai_app.apps.chat.reg import model_caps, model_request_params
from kdcube_ai_app.infra.accounting.usage import ClientConfigHint
from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase


class _EmptyAnthropicStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    @property
    def text_stream(self):
        async def _empty():
            if False:
                yield ""

        return _empty()

    async def get_final_message(self):
        return SimpleNamespace(content=[], usage=None)


class _RecordingMessages:
    def __init__(self):
        self.stream_kwargs = None

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        return _EmptyAnthropicStream()


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-sonnet-5-20260801",
        "sonnet-5",
    ],
)
def test_claude_5_models_deprecate_temperature(model):
    assert model_caps(model)["temperature"] is False
    assert model_request_params(model, temperature=0.3) == {}


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-4-5"])
def test_earlier_claude_models_keep_temperature(model):
    assert model_request_params(model, temperature=0.3) == {"temperature": 0.3}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5", "claude-fable-5"])
async def test_anthropic_stream_omits_temperature_for_claude_5(model):
    messages = _RecordingMessages()
    async_client = SimpleNamespace(messages=messages)
    service = ModelServiceBase.__new__(ModelServiceBase)
    service.router = SimpleNamespace(_mk_anthropic_async=lambda: async_client)
    client = SimpleNamespace(messages=object())

    events = [
        event
        async for event in service.stream_model_text(
            client,
            [HumanMessage(content="hello")],
            temperature=0.3,
            client_cfg=ClientConfigHint(provider="anthropic", model_name=model),
        )
    ]

    assert messages.stream_kwargs["model"] == model
    assert "temperature" not in messages.stream_kwargs
    assert events[-1]["event"] == "final"


@pytest.mark.asyncio
async def test_anthropic_stream_keeps_temperature_for_supported_model():
    messages = _RecordingMessages()
    async_client = SimpleNamespace(messages=messages)
    service = ModelServiceBase.__new__(ModelServiceBase)
    service.router = SimpleNamespace(_mk_anthropic_async=lambda: async_client)
    client = SimpleNamespace(messages=object())

    _events = [
        event
        async for event in service.stream_model_text(
            client,
            [HumanMessage(content="hello")],
            temperature=0.3,
            client_cfg=ClientConfigHint(
                provider="anthropic",
                model_name="claude-sonnet-4-6",
            ),
        )
    ]

    assert messages.stream_kwargs["temperature"] == 0.3
