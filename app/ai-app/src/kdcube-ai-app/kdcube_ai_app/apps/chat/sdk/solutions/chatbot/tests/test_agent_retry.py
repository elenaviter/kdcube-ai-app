# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.chatbot.agent_retry import retry_with_compaction
from kdcube_ai_app.infra.service_hub.errors import (
    ServiceError,
    ServiceException,
    ServiceKind,
)


class _Timeline:
    def __init__(self, blocks):
        self.blocks = blocks
        self.render_calls = []

    async def render(self, **kwargs):
        self.render_calls.append(kwargs)
        return list(self.blocks)


def _image_error() -> ServiceException:
    return ServiceException(ServiceError(
        kind=ServiceKind.llm,
        service_name="StreamTracker",
        provider="anthropic",
        model_name="claude-haiku",
        error_type="invalid_request_error",
        message="Error code: 400 - Could not process image",
        stage="stream_loop",
        http_status=400,
        code="invalid_request_error",
        retryable=False,
    ))


@pytest.mark.asyncio
async def test_provider_image_rejection_retries_once_with_images_omitted():
    timeline = _Timeline([
        {"type": "text", "text": "inspect the attachment"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "eHh4eA==",
            },
            "cache_control": {"type": "ephemeral"},
        },
    ])
    ctx_browser = SimpleNamespace(timeline=timeline)
    calls = []

    async def _agent_fn(*, blocks):
        calls.append(blocks)
        if len(calls) == 1:
            raise _image_error()
        return {"status": "recovered", "blocks": blocks}

    result = await retry_with_compaction(
        ctx_browser=ctx_browser,
        system_text_fn=lambda: "system",
        agent_fn=_agent_fn,
    )

    assert result["status"] == "recovered"
    assert len(calls) == 2
    assert any(block.get("type") == "image" for block in calls[0])
    assert not any(block.get("type") == "image" for block in calls[1])
    omitted = next(
        block for block in calls[1]
        if "[IMAGE OMITTED FROM MODEL INPUT]" in str(block.get("text") or "")
    )
    assert "reason: provider_rejected_image" in omitted["text"]
    assert omitted["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_provider_image_rejection_without_image_blocks_is_not_retried():
    timeline = _Timeline([{"type": "text", "text": "no image present"}])
    ctx_browser = SimpleNamespace(timeline=timeline)
    calls = 0

    async def _agent_fn(*, blocks):
        nonlocal calls
        calls += 1
        raise _image_error()

    with pytest.raises(ServiceException):
        await retry_with_compaction(
            ctx_browser=ctx_browser,
            system_text_fn=lambda: "system",
            agent_fn=_agent_fn,
        )

    assert calls == 1
