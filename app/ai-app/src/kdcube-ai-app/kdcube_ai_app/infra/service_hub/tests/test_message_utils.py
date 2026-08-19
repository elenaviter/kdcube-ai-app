# SPDX-License-Identifier: MIT

import base64
import io

from PIL import Image

from kdcube_ai_app.infra.service_hub.message_utils import normalize_blocks
from kdcube_ai_app.infra.service_hub.multimodality import (
    MODALITY_MAX_IMAGE_BYTES,
    MODALITY_MAX_IMAGE_DIMENSION_PX,
)


def _png_base64(size: tuple[int, int], color: str = "white") -> str:
    image = Image.new("RGB", size, color=color)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _decoded_png_size(data_b64: str) -> tuple[int, int]:
    raw = base64.b64decode(data_b64)
    with Image.open(io.BytesIO(raw)) as image:
        return image.size


def test_normalize_blocks_downscales_image_that_only_breaks_dimension_limit():
    original = _png_base64((MODALITY_MAX_IMAGE_DIMENSION_PX + 1001, 60))
    assert len(base64.b64decode(original)) < MODALITY_MAX_IMAGE_BYTES

    blocks = normalize_blocks(
        [
            {
                "type": "image",
                "data": original,
                "media_type": "image/png",
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )

    assert len(blocks) == 1
    image_block = blocks[0]
    assert image_block["type"] == "image"
    assert image_block["cache_control"] == {"type": "ephemeral"}

    normalized = image_block["source"]["data"]
    width, height = _decoded_png_size(normalized)
    assert normalized != original
    assert max(width, height) <= MODALITY_MAX_IMAGE_DIMENSION_PX
    assert width == MODALITY_MAX_IMAGE_DIMENSION_PX


def test_normalize_blocks_leaves_safe_image_untouched():
    original = _png_base64((1200, 800), color="navy")

    blocks = normalize_blocks(
        [
            {
                "type": "image",
                "data": original,
                "media_type": "image/png",
            }
        ]
    )

    assert len(blocks) == 1
    assert blocks[0]["source"]["data"] == original


def test_normalize_blocks_omits_declared_png_with_non_image_bytes():
    invalid = base64.b64encode(b"x" * 32).decode("ascii")

    blocks = normalize_blocks(
        [
            {
                "type": "image",
                "data": invalid,
                "media_type": "image/png",
                "path": "conv:fi:conv_1.turn_1.files/extracted_image.png",
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )

    assert blocks == [{
        "type": "text",
        "text": (
            "[IMAGE OMITTED FROM MODEL INPUT]\n"
            "media_type: image/png\n"
            "path: conv:fi:conv_1.turn_1.files/extracted_image.png\n"
            "reason: invalid_image_data"
        ),
        "cache_control": {"type": "ephemeral"},
    }]
