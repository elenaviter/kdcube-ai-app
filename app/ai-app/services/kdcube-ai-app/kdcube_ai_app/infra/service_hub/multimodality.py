# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# infra/service_hub/multimodality.py

from typing import Dict, Any

MODALITY_IMAGE_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MODALITY_DOC_MIME = {"application/pdf"}

MODALITY_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
MODALITY_MAX_DOC_BYTES = 10 * 1024 * 1024   # 10 MB

MESSAGE_MAX_BYTES = 25 * 1024 * 1024  # total message size cap (text + attachments); keep margin

def estimate_image_tokens_from_base64(base64_data: str) -> int:
    """
    Estimate Anthropic image tokens from base64 size.

    Anthropic pricing tiers:
    - <200KB: ~150 tokens
    - <500KB: ~400 tokens
    - <5MB: ~1600 tokens

    Args:
        base64_data: Base64-encoded image data

    Returns:
        Estimated token cost (150-1600)
    """
    if not base64_data:
        return 0

    size_bytes = len(base64_data) * 3 / 4  # base64 → bytes
    kb = size_bytes / 1024

    if kb < 200:
        return 150
    elif kb < 500:
        return 400
    else:
        return 1600

def estimate_tokens(text: str, *, divisor: int = 4) -> int:
    if not text:
        return 0
    return max(1, len(text) // max(1, divisor))

def estimate_pdf_tokens_from_base64(base64_data: str) -> int:
    """
    Estimate Anthropic PDF tokens from base64 size.

    Anthropic renders PDFs as images: ~10k tokens per page.
    Rough estimate: 50-100KB per page typical.

    Args:
        base64_data: Base64-encoded PDF data

    Returns:
        Estimated token cost (pages × 10k)
    """
    if not base64_data:
        return 0

    size_bytes = len(base64_data) * 3 / 4
    # Conservative estimate: 75KB per page
    estimated_pages = max(1, int(size_bytes / 75_000))
    return estimated_pages * 10_000
