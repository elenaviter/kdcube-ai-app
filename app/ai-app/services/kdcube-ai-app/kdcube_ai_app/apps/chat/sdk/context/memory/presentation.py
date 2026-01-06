# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/context/memory/presentation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_selected_memories_log(selected_bucket_cards: Optional[List[Dict[str, Any]]]) -> str:
    if not selected_bucket_cards:
        return ""
    lines = ["[selected.memories]"]
    for card in selected_bucket_cards:
        if not isinstance(card, dict):
            continue
        bid = (card.get("bucket_id") or "").strip()
        name = (card.get("name") or "").strip()
        desc = (card.get("short_desc") or "").strip()
        parts: list[str] = []
        if bid:
            parts.append(f"id={bid}")
        if name:
            parts.append(f"name={name}")
        if desc:
            parts.append(f"desc={desc}")
        if parts:
            lines.append("- " + " | ".join(parts))
    return "\n".join(lines) if len(lines) > 1 else ""
