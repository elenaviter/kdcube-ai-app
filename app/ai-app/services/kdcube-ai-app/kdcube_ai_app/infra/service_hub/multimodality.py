# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# infra/service_hub/multimodality.py

MODALITY_IMAGE_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MODALITY_DOC_MIME = {"application/pdf"}

MODALITY_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
MODALITY_MAX_DOC_BYTES = 10 * 1024 * 1024   # 10 MB