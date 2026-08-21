# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

"""Framework-neutral identity helpers for harness timeline blocks."""

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    qualify_conversation_ref,
)


def event_record_ref(
    *,
    turn_id: str,
    event_id: str,
    conversation_id: str = "",
    logical_path: str = "",
) -> str:
    """Return the canonical record ref for one accepted event occurrence.

    An ingress/provider may already have stamped a ``conv:ev:`` path. Otherwise
    every adapter uses the same turn + occurrence fallback. The referenced
    object's own locator remains a separate ``object_ref`` in the event body.
    """
    supplied = str(logical_path or "").strip()
    if supplied.startswith("conv:ev:"):
        return qualify_conversation_ref(supplied, conversation_id)
    turn = str(turn_id or "").strip()
    occurrence = str(event_id or "").strip()
    if not turn or not occurrence:
        return ""
    return qualify_conversation_ref(
        f"conv:ev:{turn}.events/{occurrence}",
        conversation_id,
    )


def event_identity_fields(
    *,
    event_source_id: str,
    event_id: str,
    story_id: str | None = None,
) -> dict[str, str]:
    """Build the common source and occurrence identity fields for timeline blocks."""
    fields = {
        "event_source_id": str(event_source_id or "").strip(),
        "event_id": str(event_id or "").strip(),
    }
    story = str(story_id or "").strip()
    if story:
        fields["story_id"] = story
    return {key: value for key, value in fields.items() if value}


def block_event_source_id(
    block: Mapping[str, Any] | None,
    *,
    call_meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Resolve the policy source id for a timeline block.

    Tool-backed blocks use their existing ``tool_id``. Result/file blocks can
    recover that id through ``call_id`` and caller-provided call metadata.
    Non-tool events carry an explicit ``event_source_id``.
    """
    if not isinstance(block, Mapping):
        return ""
    explicit = str(block.get("event_source_id") or "").strip()
    if explicit:
        return explicit
    meta = block.get("meta") if isinstance(block.get("meta"), Mapping) else {}
    explicit = str(meta.get("event_source_id") or "").strip()
    if explicit:
        return explicit
    tool_id = str(block.get("tool_id") or "").strip()
    if tool_id:
        return tool_id
    tool_id = str(meta.get("tool_id") or "").strip()
    if tool_id:
        return tool_id
    call_id = str(block.get("call_id") or meta.get("tool_call_id") or "").strip()
    if call_id and isinstance(call_meta, Mapping):
        row = call_meta.get(call_id)
        if isinstance(row, Mapping):
            return str(row.get("tool_id") or "").strip()
    return ""


def block_event_id(block: Mapping[str, Any] | None) -> str:
    """Resolve the occurrence id for a timeline block."""
    if not isinstance(block, Mapping):
        return ""
    explicit = str(block.get("event_id") or "").strip()
    if explicit:
        return explicit
    meta = block.get("meta") if isinstance(block.get("meta"), Mapping) else {}
    explicit = str(meta.get("event_id") or "").strip()
    if explicit:
        return explicit
    return str(block.get("call_id") or meta.get("tool_call_id") or "").strip()


def block_matches_event_source(
    block: Mapping[str, Any] | None,
    event_source_id: str,
    *,
    call_meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Return true when a block belongs to the given policy source."""
    source_id = str(event_source_id or "").strip()
    return bool(source_id and block_event_source_id(block, call_meta=call_meta) == source_id)


def stamp_event_identity(
    block: MutableMapping[str, Any],
    *,
    event_source_id: str,
    event_id: str,
    story_id: str | None = None,
) -> MutableMapping[str, Any]:
    """Attach source and occurrence identity to an already-shaped timeline block."""
    block.update(
        event_identity_fields(
            event_source_id=event_source_id,
            event_id=event_id,
            story_id=story_id,
        )
    )
    return block


def stamp_event_identity_many(
    blocks: Iterable[MutableMapping[str, Any]],
    *,
    event_source_id: str,
    event_id: str,
    story_id: str | None = None,
) -> list[MutableMapping[str, Any]]:
    """Attach the same source and occurrence identity to a block group."""
    return [
        stamp_event_identity(
            block,
            event_source_id=event_source_id,
            event_id=event_id,
            story_id=story_id,
        )
        for block in blocks
    ]
