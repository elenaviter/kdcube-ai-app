# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Framework-neutral semantic contributions to a completed turn.

Hosted agent frameworks keep their private checkpoints and session summaries.
When an agent deliberately wants one result to become part of KDCube's shared,
searchable conversation record, it stages that result here. Staging mutates the
trusted per-turn state only; the turn recorder is the sole durable writer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


TURN_SUMMARY_CONTRIBUTION_KEY = "_kdcube_turn_summary_contribution"
TURN_SUMMARY_TOOL_NAME = "record_turn_summary"

_MAX_SUMMARY_CHARS = 24_000
_MAX_REFS = 64
_MAX_ANCHORS_PER_KIND = 32
_MAX_REF_CHARS = 2_048
_MAX_ANCHOR_CHARS = 256


def _normalized_values(
    values: Optional[Sequence[Any]],
    *,
    field: str,
    count_limit: int,
    char_limit: int,
) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a list of strings")
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        if "\n" in value or "\r" in value:
            raise ValueError(f"{field} entries must be single-line strings")
        if len(value) > char_limit:
            raise ValueError(f"{field} entries may contain at most {char_limit} characters")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) > count_limit:
            raise ValueError(f"{field} may contain at most {count_limit} entries")
    return out


def render_turn_summary(contribution: Mapping[str, Any]) -> str:
    """Render the staged structured contribution as searchable summary text."""
    text = str(contribution.get("summary") or "").strip()
    refs = list(contribution.get("refs") or [])
    phrases = list(contribution.get("phrases") or [])
    entities = list(contribution.get("entities") or [])
    parts = [text]
    if refs:
        parts.extend(["", "References:", *[f"- {ref}" for ref in refs]])
    if phrases or entities:
        parts.extend([
            "",
            "Retrieval-anchors:",
            f"  phrases: {json.dumps(phrases, ensure_ascii=False)}",
            f"  entities: {json.dumps(entities, ensure_ascii=False)}",
        ])
    return "\n".join(parts).strip()


def stage_turn_summary(
    state: MutableMapping[str, Any],
    *,
    summary: str,
    refs: Optional[Sequence[Any]] = None,
    phrases: Optional[Sequence[Any]] = None,
    entities: Optional[Sequence[Any]] = None,
    contributor: str = "foreign_agent",
) -> Dict[str, Any]:
    """Stage one replaceable summary under trusted current-turn state.

    The model supplies semantic content only. The adapter supplies ``state`` and
    ``contributor``, so the model cannot choose conversation identity or persist
    into another turn. A later call in the same turn replaces the earlier draft.
    """
    if not isinstance(state, MutableMapping):
        raise ValueError("record_turn_summary requires mutable trusted turn state")
    body = str(summary or "").strip()
    if not body:
        raise ValueError("summary must not be empty")
    if len(body) > _MAX_SUMMARY_CHARS:
        raise ValueError(f"summary may contain at most {_MAX_SUMMARY_CHARS} characters")
    turn_id = str(state.get("turn_id") or "").strip()
    if not turn_id:
        raise ValueError("record_turn_summary requires the current turn id")

    contribution = {
        "version": "v1",
        "turn_id": turn_id,
        "summary": body,
        "refs": _normalized_values(
            refs, field="refs", count_limit=_MAX_REFS, char_limit=_MAX_REF_CHARS
        ),
        "phrases": _normalized_values(
            phrases,
            field="phrases",
            count_limit=_MAX_ANCHORS_PER_KIND,
            char_limit=_MAX_ANCHOR_CHARS,
        ),
        "entities": _normalized_values(
            entities,
            field="entities",
            count_limit=_MAX_ANCHORS_PER_KIND,
            char_limit=_MAX_ANCHOR_CHARS,
        ),
        "contributor": str(contributor or "foreign_agent").strip() or "foreign_agent",
    }
    replaced = isinstance(state.get(TURN_SUMMARY_CONTRIBUTION_KEY), Mapping)
    state[TURN_SUMMARY_CONTRIBUTION_KEY] = contribution
    return {
        "status": "staged",
        "replaced": replaced,
        "summary_chars": len(body),
        "refs_count": len(contribution["refs"]),
        "phrases_count": len(contribution["phrases"]),
        "entities_count": len(contribution["entities"]),
        "durable_after_turn_completion": True,
    }


def staged_turn_summary(
    state: Optional[Mapping[str, Any]], *, turn_id: str = ""
) -> Optional[Dict[str, Any]]:
    """Return this turn's staged contribution, excluding stale state."""
    if not isinstance(state, Mapping):
        return None
    raw = state.get(TURN_SUMMARY_CONTRIBUTION_KEY)
    if not isinstance(raw, Mapping):
        return None
    expected = str(turn_id or state.get("turn_id") or "").strip()
    actual = str(raw.get("turn_id") or "").strip()
    if not expected or actual != expected or not str(raw.get("summary") or "").strip():
        return None
    return dict(raw)


def turn_summary_block(
    contribution: Optional[Mapping[str, Any]], *, turn_id: str, ts: str
) -> Optional[Dict[str, Any]]:
    """Build the canonical TurnLog block for a staged summary."""
    if not isinstance(contribution, Mapping):
        return None
    if str(contribution.get("turn_id") or "").strip() != str(turn_id or "").strip():
        return None
    text = render_turn_summary(contribution)
    if not text:
        return None
    return {
        "type": "conv.working.summary",
        "author": "assistant",
        "turn_id": turn_id,
        "turn": turn_id,
        "ts": ts,
        "mime": "text/markdown",
        "path": f"conv:ws:{turn_id}.conv.working.summary.attempt.1",
        "text": text,
        "meta": {
            "kind": "working_summary",
            "summary_scope": "turn",
            "source_tool": TURN_SUMMARY_TOOL_NAME,
            "contributor": str(contribution.get("contributor") or "foreign_agent"),
            "refs": list(contribution.get("refs") or []),
            "phrases": list(contribution.get("phrases") or []),
            "entities": list(contribution.get("entities") or []),
        },
    }


__all__ = [
    "TURN_SUMMARY_CONTRIBUTION_KEY",
    "TURN_SUMMARY_TOOL_NAME",
    "render_turn_summary",
    "stage_turn_summary",
    "staged_turn_summary",
    "turn_summary_block",
]
