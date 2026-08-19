# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Multimodality passthrough — a multimodal user turn survives the model bridge.

The multimodality seam builds a LangChain ``HumanMessage`` whose ``content`` is a
native block list ``[{"type":"text",...}, {"type":"image","data":<b64>,
"media_type":<mime>}]``. These offline tests prove that shape flows through KDCube's
provider normalizers — the exact functions ``stream_model_text_tracked`` calls —
WITHOUT dropping the image, for BOTH providers:

  - Anthropic (``message_utils.extract_message_blocks`` + ``normalize_blocks``):
    the flat block becomes ``{"type":"image","source":{"type":"base64",...}}``.
  - OpenAI (``openai.normalize_message_for_openai``): it becomes an
    ``image_url`` data-URI block.

Fully offline — pure function calls, no model, no network. This is the contract the
ported agents rely on to be multimodal without touching a provider SDK.
"""
from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from kdcube_ai_app.infra.service_hub.message_utils import extract_message_blocks, normalize_blocks
from kdcube_ai_app.infra.service_hub.openai import normalize_message_for_openai

# A real 1x1 PNG so the image normalizer (which decodes+inspects) keeps it intact.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/"
    "AAX+Av4N70a4AAAAAElFTkSuQmCC"
)


def _multimodal_human() -> HumanMessage:
    return HumanMessage(
        content=[
            {"type": "text", "text": "what is in this image?"},
            {"type": "image", "data": _PNG_B64, "media_type": "image/png"},
        ]
    )


def test_anthropic_normalizer_keeps_the_image_block() -> None:
    msg = _multimodal_human()
    blocks = extract_message_blocks(msg)  # reads .content list
    assert blocks, "content list must be seen as message blocks"

    norm = normalize_blocks(blocks)
    types = [b.get("type") for b in norm]
    assert "text" in types and "image" in types

    image = next(b for b in norm if b.get("type") == "image")
    # The flat {data, media_type} is folded into the Anthropic base64 source block,
    # data intact (not dropped, not truncated to a placeholder).
    src = image.get("source") or {}
    assert src.get("type") == "base64"
    assert src.get("media_type") == "image/png"
    assert src.get("data") == _PNG_B64


def test_openai_normalizer_keeps_the_image_block() -> None:
    msg = _multimodal_human()

    class _FakeClient:  # normalize_message_for_openai only needs a placeholder here
        pass

    out = asyncio.run(normalize_message_for_openai(_FakeClient(), msg))
    parts = out.content
    assert isinstance(parts, list)
    kinds = [p.get("type") for p in parts]
    assert "text" in kinds and "image_url" in kinds

    image = next(p for p in parts if p.get("type") == "image_url")
    url = image["image_url"]["url"]
    # Data URI carrying the base64 verbatim.
    assert url.startswith("data:image/png;base64,")
    assert _PNG_B64 in url


def test_text_only_human_message_is_untouched() -> None:
    # A plain-string HumanMessage (the common, no-attachment case) carries no blocks
    # and passes through both paths unchanged.
    msg = HumanMessage(content="just text")
    assert extract_message_blocks(msg) is None

    class _FakeClient:
        pass

    out = asyncio.run(normalize_message_for_openai(_FakeClient(), msg))
    assert out.content == "just text"
