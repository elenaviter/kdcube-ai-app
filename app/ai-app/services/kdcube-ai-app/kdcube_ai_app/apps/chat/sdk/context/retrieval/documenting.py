# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/retrieval/documenting.py

import datetime as _dt

def _iso(ts: str | None) -> str:
    if not ts: return ""
    try:
        # keep the original Z if present
        return _dt.datetime.fromisoformat(ts.replace("Z","+00:00")).replace(tzinfo=_dt.timezone.utc).isoformat().replace("+00:00","Z")
    except Exception:
        return ts

def _format_context_block(title: str, items: list[dict]) -> str:
    """
    Render context *verbatim* from artifact texts, with light separation.
    No parsing, no KVs, no reformatting — exactly as stored.
    """
    if not items:
        return ""

    out = [
        f"### {title}",
        "_This block is system-provided context related to this message; it was **not** authored by the user._"
    ]

    first = True
    for it in items:
        txt = (it.get("text") or it.get("content") or "").strip()
        if not txt:
            continue
        if not first:
            out.append("\n---\n")
        out.append(txt)
        first = False

    return "\n".join(out)
