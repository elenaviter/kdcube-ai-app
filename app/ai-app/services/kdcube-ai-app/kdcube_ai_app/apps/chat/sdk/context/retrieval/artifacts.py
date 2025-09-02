# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/retrieval/artifacts.py

from typing import Dict, Any, Tuple
import datetime

def _provenance_from_item(m: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Build a succinct provenance label + metadata for reranker/answer snippets.
    Returns (label, meta)
    label examples: 'conv:user', 'conv:assistant', 'kb:kevc', 'kb:nvd', 'upload:file', 'other'
    meta keys: source, channel/provider, age_days (if ts available), title (<=120 chars), label
    """
    label = "other"
    meta: Dict[str, Any] = {"source": "other"}

    if "role" in m:
        ch = (m.get("role") or "").lower()
        label = f"conv:{ch or 'message'}"
        meta = {"source": "conversation", "channel": ch}
    elif m.get("provider"):
        pv = str(m.get("provider")).lower()
        label = f"kb:{pv}"
        meta = {"source": "kb", "provider": pv}
    elif (m.get("origin") or "").lower() == "upload" or (m.get("source") or "").lower() == "upload":
        label = "upload:file"
        meta = {"source": "upload"}

    # Attach title if any
    title = (m.get("title") or m.get("summary") or m.get("heading") or "").strip()
    if title:
        meta["title"] = (title[:120] + "…") if len(title) > 120 else title

    # Compute age_days from ISO timestamp if present
    ts = m.get("ts") or m.get("timestamp")
    try:
        if ts:
            dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            age = (datetime.datetime.now(datetime.timezone.utc) - dt.astimezone(datetime.timezone.utc)).days
            if age >= 0:
                meta["age_days"] = age
    except Exception:
        pass

    meta["label"] = label
    return label, meta

def _format_snippet_line(idx: int, doc: Dict[str, Any]) -> str:
    """
    Pretty print snippet header + content for the answer generator context.
    Keeps the numeric prefix [n] for citation compatibility and appends a compact provenance label.
    """
    meta = doc.get("meta") or {}
    label = meta.get("label") or "snippet"
    title = meta.get("title")
    age = meta.get("age_days")
    age_str = f"; age={age}d" if isinstance(age, int) else ""
    title_str = f" — {title}" if title else ""
    return f"[{idx}] | {label}{age_str}{title_str}\n{doc.get('content') or ''}"