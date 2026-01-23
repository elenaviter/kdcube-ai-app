# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/context/memory/objective_memory_log.py

from typing import List

def render_log_for_prompt(entries: List[dict], *, per_bucket_limit: int = 6) -> str:
    """Freshest-first markdown block, tiny and readable."""
    if not entries:
        return ""
    parts = ["[OBJECTIVE MEMORY LOG — freshest first]"]
    for e in list(reversed(entries[-per_bucket_limit:]))[::-1]:  # freshest-first
        lines = [f"objective: {e.get('objective','').strip()}"]
        if e.get("facts"):       lines.append("facts: " + ", ".join([f"{f['key']}=" + (str(f.get('value')) if f.get('value') is not None else "true") for f in e["facts"]]))
        if e.get("assertions"):  lines.append("assertions: " + ", ".join([f"{a['key']}=" + (str(a.get('value')) if a.get('value') is not None else "true") for a in e["assertions"]]))
        if e.get("exceptions"):  lines.append("exceptions: " + ", ".join([e1["rule_key"] for e1 in e["exceptions"]]))
        parts.append("\n".join(lines))
        parts.append("")  # blank line
    return "\n".join(parts).strip()

def _render_selected_memory_buckets_block(objective_memory_section: list[dict],
                                          max_buckets: int = 3,
                                          max_lines_per_bucket: int = 6) -> str:
    """
    objective_memory_section item shape (from build_gate_context_hints):
      { "bucket_card": <card dict>, "timeline": [ {ts_from, ts_to, objective_hint, assertions, exceptions, facts}, ... ] }
    """
    if not objective_memory_section:
        return ""
    lines = ["[OBJECTIVE MEMORY — SELECTED]"]
    for obj in objective_memory_section[:max_buckets]:
        card = (obj or {}).get("bucket_card") or {}
        tl   = (obj or {}).get("timeline") or []
        name = (card.get("name") or card.get("bucket_id") or "(bucket)").strip()
        desc = (card.get("short_desc") or card.get("objective_text") or "").strip()
        updated = (card.get("updated_at") or "").strip()
        head = f"• {name}"
        if desc:   head += f" — {desc}"
        if updated: head += f"  (updated {updated})"
        lines.append(head)

        # compact per-bucket picks coming from the card (already capped when built)
        def _mk_sig(sig: dict) -> str:
            k = sig.get("key") or ""
            v = sig.get("value")
            if isinstance(v, (dict, list)):  # be safe
                try:
                    import json as _json
                    v = _json.dumps(v, ensure_ascii=False)
                except Exception:
                    v = str(v)
            return f"{k} = {v}"

        for sig in (card.get("assertions") or [])[:max_lines_per_bucket]:
            lines.append("   - assertion: " + _mk_sig(sig))
        for sig in (card.get("exceptions") or [])[:max_lines_per_bucket]:
            lines.append("   - exception: " + _mk_sig(sig))
        for sig in (card.get("facts") or [])[:max_lines_per_bucket]:
            lines.append("   - fact:      " + _mk_sig(sig))

        # a couple of timeline hints (oldest->newest feel)
        for s in tl[:2]:
            oh = (s.get("objective_hint") or "").strip()
            tf, tt = (s.get("ts_from") or "").strip(), (s.get("ts_to") or "").strip()
            if oh:
                lines.append(f"     · [{tf}..{tt}] {oh}")

    return "\n".join(lines).strip()


