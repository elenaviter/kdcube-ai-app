# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Agent-harness timeline identity, projection, and persistence contracts."""

from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.identity import (
    block_event_id,
    block_event_source_id,
    block_matches_event_source,
    event_record_ref,
    event_identity_fields,
    stamp_event_identity,
    stamp_event_identity_many,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.projection import (
    apply_provider_render_patch_results,
    block_owned_by_namespace,
    normalize_provider_render_patches,
    object_ref_from_block,
    object_ref_from_event,
    render_window_blocks,
)

__all__ = [
    "block_event_id",
    "block_event_source_id",
    "block_matches_event_source",
    "event_record_ref",
    "event_identity_fields",
    "stamp_event_identity",
    "stamp_event_identity_many",
    "apply_provider_render_patch_results",
    "block_owned_by_namespace",
    "normalize_provider_render_patches",
    "object_ref_from_block",
    "object_ref_from_event",
    "render_window_blocks",
]
