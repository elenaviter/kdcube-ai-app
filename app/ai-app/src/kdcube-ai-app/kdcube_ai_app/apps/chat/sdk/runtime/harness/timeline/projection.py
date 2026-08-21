# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Safe, framework-neutral projection of provider-owned timeline blocks."""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from typing import Any

from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.identity import block_event_source_id


def object_ref_from_event(event: Mapping[str, Any] | None) -> str:
    """Return the object locator carried by an event, never its occurrence ref."""
    if not isinstance(event, Mapping):
        return ""
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    nested_event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
    # Accepted event bodies are the most specific projection. Outer envelopes
    # often carry the occurrence's conv:ev ref in their own ref/path fields.
    for source in (nested_event, payload, event):
        for key in (
            "object_ref",
            "event_ref",  # Legacy provider alias; conv:ev remains an occurrence.
            "ref",
            "hosted_uri",
            "logical_path",
            "path",
        ):
            text = str(source.get(key) or "").strip()
            if ":" in text and not text.startswith("conv:ev:"):
                return text
    return ""


def object_ref_from_block(block: Mapping[str, Any]) -> str:
    """Read the canonical owner ref from a timeline block or nested event payload."""
    for value in (
        block.get("object_ref"),
        block.get("logical_path"),
        block.get("hosted_uri"),
    ):
        text = str(value or "").strip()
        if ":" in text and not text.startswith("conv:ev:"):
            return text
    event = block.get("event") if isinstance(block.get("event"), Mapping) else {}
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    nested_event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
    for source in (nested_event, payload, event):
        for key in (
            "object_ref",
            "event_ref",  # Legacy provider alias; conv:ev remains an occurrence.
            "ref",
            "path",
            "logical_path",
            "hosted_uri",
        ):
            text = str(source.get(key) or "").strip()
            if ":" in text and not text.startswith("conv:ev:"):
                return text
    meta = block.get("meta") if isinstance(block.get("meta"), Mapping) else {}
    meta_target = meta.get("target") if isinstance(meta.get("target"), Mapping) else {}
    for source in (meta, meta_target):
        for key in ("object_ref", "ref", "path", "logical_path", "hosted_uri"):
            text = str(source.get(key) or "").strip()
            if ":" in text and not text.startswith("conv:ev:"):
                return text
    return ""


def namespace_from_ref(ref: str) -> str:
    text = str(ref or "").strip()
    if ":" not in text:
        return ""
    return text.split(":", 1)[0].strip().lower().rstrip(":")


def block_owned_by_namespace(
    block: Mapping[str, Any],
    *,
    namespace: str,
    event_source_id: str,
    call_meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Return true only when a provider may patch this block."""
    if block_event_source_id(block, call_meta=call_meta) == event_source_id:
        return True
    object_ref = object_ref_from_block(block)
    return bool(object_ref and namespace_from_ref(object_ref) == namespace)


def render_window_blocks(
    timeline_blocks: list[MutableMapping[str, Any]],
    *,
    owned_indexes: set[int],
    neighbor_radius: int,
    max_blocks: int,
) -> list[dict[str, Any]]:
    """Build a bounded, copied render window around provider-owned blocks."""
    selected: set[int] = {
        index for index in owned_indexes if 0 <= index < len(timeline_blocks)
    }
    radius = max(0, int(neighbor_radius or 0))
    max_count = max(1, int(max_blocks or 1))
    for distance in range(1, radius + 1):
        if len(selected) >= max_count and len(selected) >= len(owned_indexes):
            break
        for index in sorted(owned_indexes):
            for candidate in (index - distance, index + distance):
                if 0 <= candidate < len(timeline_blocks):
                    selected.add(candidate)
                if len(selected) >= max_count and len(selected) >= len(owned_indexes):
                    break
    out: list[dict[str, Any]] = []
    for index in sorted(selected):
        block = copy.deepcopy(timeline_blocks[index])
        if isinstance(block, MutableMapping):
            block["index"] = index
            out.append(dict(block))
    return out


def coerce_block_index(value: Any) -> int | None:
    try:
        index = int(value)
    except Exception:
        return None
    return index if index >= 0 else None


def prepare_provider_render_block(
    block: Mapping[str, Any],
    *,
    namespace: str,
    event_source_id: str,
    fallback_object_ref: str = "",
) -> dict[str, Any] | None:
    """Stamp a provider patch while preserving the owning namespace boundary."""
    out = copy.deepcopy(dict(block or {}))
    out.pop("index", None)
    object_ref = object_ref_from_block(out) or fallback_object_ref
    if object_ref and namespace_from_ref(object_ref) != namespace:
        return None
    if object_ref:
        out.setdefault("object_ref", object_ref)
    out.setdefault("event_source_id", event_source_id)
    meta = out.get("meta") if isinstance(out.get("meta"), Mapping) else {}
    meta = dict(meta or {})
    if object_ref:
        meta.setdefault("object_ref", object_ref)
        meta.setdefault("source_namespace", namespace)
    meta.setdefault("resolved_event_source_id", event_source_id)
    meta["provider_rendered"] = True
    out["meta"] = meta
    return out


def normalize_provider_render_patches(
    response: Any,
    *,
    namespace: str,
    event_source_id: str,
    owned_indexes: set[int],
    timeline_blocks: list[MutableMapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Validate provider patches against ownership and stable timeline indexes."""
    extra = response.extra if hasattr(response, "extra") else {}
    ret = response.ret if hasattr(response, "ret") and isinstance(response.ret, Mapping) else {}
    patches_raw = extra.get("patches") if isinstance(extra, Mapping) else None
    if patches_raw is None and isinstance(ret, Mapping):
        patches_raw = ret.get("patches")
    blocks_raw = extra.get("blocks") if isinstance(extra, Mapping) else None
    if blocks_raw is None and isinstance(ret, Mapping):
        blocks_raw = ret.get("blocks")

    raw_items: list[Any] = []
    if isinstance(patches_raw, list):
        raw_items.extend(patches_raw)
    if isinstance(blocks_raw, list):
        for block in blocks_raw:
            if not isinstance(block, Mapping):
                continue
            index = coerce_block_index(
                block.get("index") or block.get("target_index") or block.get("block_index")
            )
            if index is not None:
                raw_items.append({"op": "replace_block", "index": index, "block": dict(block)})

    patches: list[dict[str, Any]] = []
    rejected = 0
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            rejected += 1
            continue
        op = str(raw.get("op") or raw.get("operation") or "").strip().lower().replace("-", "_")
        if op in {"replace", "replace_block", "set_block"}:
            index = coerce_block_index(
                raw.get("index") or raw.get("target_index") or raw.get("block_index")
            )
            block = raw.get("block") if isinstance(raw.get("block"), Mapping) else {}
            if index is None or index not in owned_indexes or not isinstance(block, Mapping):
                rejected += 1
                continue
            fallback_ref = (
                object_ref_from_block(timeline_blocks[index])
                if index < len(timeline_blocks)
                else ""
            )
            prepared = prepare_provider_render_block(
                block,
                namespace=namespace,
                event_source_id=event_source_id,
                fallback_object_ref=fallback_ref,
            )
            if prepared is None:
                rejected += 1
                continue
            patches.append({"op": "replace_block", "index": index, "block": prepared})
            continue
        if op in {"patch", "patch_block", "update", "update_block"}:
            index = coerce_block_index(
                raw.get("index") or raw.get("target_index") or raw.get("block_index")
            )
            fields = raw.get("fields") if isinstance(raw.get("fields"), Mapping) else raw.get("patch")
            if index is None or index not in owned_indexes or not isinstance(fields, Mapping):
                rejected += 1
                continue
            original = copy.deepcopy(dict(timeline_blocks[index]))
            for key, value in dict(fields).items():
                if key == "meta" and isinstance(value, Mapping):
                    merged_meta = dict(
                        original.get("meta") if isinstance(original.get("meta"), Mapping) else {}
                    )
                    merged_meta.update(dict(value))
                    original["meta"] = merged_meta
                else:
                    original[key] = value
            prepared = prepare_provider_render_block(
                original,
                namespace=namespace,
                event_source_id=event_source_id,
                fallback_object_ref=object_ref_from_block(timeline_blocks[index]),
            )
            if prepared is None:
                rejected += 1
                continue
            patches.append({"op": "replace_block", "index": index, "block": prepared})
            continue
        if op in {"append", "append_block", "append_block_after"}:
            index = coerce_block_index(
                raw.get("index") or raw.get("anchor_index") or raw.get("after_index")
            )
            block = raw.get("block") if isinstance(raw.get("block"), Mapping) else {}
            if index is None or index not in owned_indexes or not isinstance(block, Mapping):
                rejected += 1
                continue
            prepared = prepare_provider_render_block(
                block,
                namespace=namespace,
                event_source_id=event_source_id,
                fallback_object_ref=object_ref_from_block(timeline_blocks[index]),
            )
            if prepared is None:
                rejected += 1
                continue
            patches.append({"op": "append_block_after", "index": index, "block": prepared})
            continue
        rejected += 1
    status = "patches" if patches else "empty"
    if rejected:
        status = f"{status};rejected={rejected}"
    return patches, status


def apply_provider_render_patch_results(
    timeline_blocks: list[MutableMapping[str, Any]],
    results: list[Mapping[str, Any]],
) -> int:
    """Merge validated provider patches deterministically into the timeline."""
    replacements: dict[int, dict[str, Any]] = {}
    appends: dict[int, list[dict[str, Any]]] = {}
    changed = 0
    for result in results or []:
        for patch in result.get("patches") or []:
            if not isinstance(patch, Mapping):
                continue
            op = str(patch.get("op") or "").strip()
            index = coerce_block_index(patch.get("index"))
            block = patch.get("block") if isinstance(patch.get("block"), Mapping) else None
            if index is None or block is None:
                continue
            if op == "replace_block":
                if index not in replacements:
                    replacements[index] = dict(block)
                    changed += 1
                continue
            if op == "append_block_after":
                appends.setdefault(index, []).append(dict(block))
                changed += 1
    for index, block in replacements.items():
        if 0 <= index < len(timeline_blocks):
            timeline_blocks[index] = block
    if appends:
        rebuilt: list[MutableMapping[str, Any]] = []
        for index, block in enumerate(timeline_blocks):
            rebuilt.append(block)
            for appended in appends.get(index, ()):
                rebuilt.append(appended)
        timeline_blocks[:] = rebuilt
    return changed


__all__ = [
    "apply_provider_render_patch_results",
    "block_owned_by_namespace",
    "coerce_block_index",
    "namespace_from_ref",
    "normalize_provider_render_patches",
    "object_ref_from_block",
    "object_ref_from_event",
    "prepare_provider_render_block",
    "render_window_blocks",
]
