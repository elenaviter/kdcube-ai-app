# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The block catalog — every composition unit the constructor can offer.

Built-in blocks come from the two in-tree registries (moderate
``REACT_LITE_*``, extra-lite ``REACT_XLITE_*``). Each entry carries its
MEANING: the curated SIGNALS the block protects/teaches and semantic tags
that reflect them (``block_signals.py``; the signal table in
``docs/sdk/skills/instruction-blocks-and-signals-README.md`` is the long form),
plus a derived text hint, its profile memberships, and the full block text
for the constructor's details view. Stored custom units author their own
signals/tags at save time (served by the store, not here).
"""

from __future__ import annotations

import re
from typing import Any

from kdcube_ai_app.apps.chat.sdk.skills.instructions.shared_instructions_lite import (
    REACT_LITE_PROFILE_BLOCKS,
    list_lite_instruction_blocks,
)
from kdcube_ai_app.apps.chat.sdk.skills.instructions.instructions_extra_lite import (
    list_extra_lite_instruction_blocks,
)
from kdcube_ai_app.apps.chat.sdk.solutions.agentic_config.instructions.block_signals import (
    BLOCK_SIGNALS,
)
from kdcube_ai_app.apps.chat.sdk.util import token_count

_DESCRIPTION_LIMIT = 160


def _derive_description(text: str) -> str:
    """Header tag + first content sentence, picker-sized."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    head = lines[0]
    body = ""
    if re.fullmatch(r"\[[^\]]+\]", head) and len(lines) > 1:
        body = lines[1]
    elif re.match(r"^\[[^\]]+\]", head):
        body = head[head.index("]") + 1 :].strip()
        head = head[: head.index("]") + 1]
    out = f"{head} {body}".strip() if body else head
    if len(out) > _DESCRIPTION_LIMIT:
        out = out[: _DESCRIPTION_LIMIT - 1].rstrip() + "…"
    return out


def builtin_block_catalog() -> list[dict[str, Any]]:
    """Every built-in block as
    ``{name, tier, description, signals, tags, profiles, text, tokens}``.

    ``signals``/``tags`` carry the block's MEANING (curated); ``profiles``
    lists the moderate profiles whose expansion includes it; ``text`` is the
    full block body for the details view; ``tokens`` is its token weight
    (cl100k), so the constructor can show what each block costs.
    """
    profile_membership: dict[str, list[str]] = {}
    for profile, blocks in (REACT_LITE_PROFILE_BLOCKS or {}).items():
        for name in blocks or ():
            profile_membership.setdefault(str(name), []).append(str(profile))

    def _entry(name: str, text: str, tier: str) -> dict[str, Any]:
        meaning = BLOCK_SIGNALS.get(name, {})
        signals = list(meaning.get("signals") or [])
        tags = list(meaning.get("tags") or [tier])
        body = str(text or "").strip()
        return {
            "name": name,
            "tier": tier,
            "description": signals[0] if signals else _derive_description(text),
            "signals": signals,
            "tags": tags,
            "profiles": sorted(profile_membership.get(name, [])),
            "text": body,
            "tokens": token_count(body),
        }

    catalog: list[dict[str, Any]] = []
    for name, text in sorted(list_lite_instruction_blocks().items()):
        catalog.append(_entry(name, text, "moderate"))
    for name, text in sorted(list_extra_lite_instruction_blocks().items()):
        catalog.append(_entry(name, text, "extra-lite"))
    return catalog


__all__ = ["builtin_block_catalog"]
