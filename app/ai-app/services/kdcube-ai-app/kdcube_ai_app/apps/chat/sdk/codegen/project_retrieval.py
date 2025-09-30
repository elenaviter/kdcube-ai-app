# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/codegen/project_retrieval.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

# ---- tiny utils ----

def _first_md_heading(md: str) -> str:
    for ln in (md or "").splitlines():
        t = ln.strip()
        if t.startswith("#"):
            return t.lstrip("# ").strip()
    return ""

def _short(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

EDITABLE_SLOT_NAMES = {
    "editable_md","document_md","draft_md","summary_md","outline_md",
    "body_md","report_md","article_md","content_md","project_canvas", "project_log"
}

_CITATION_OPTIONAL_ATTRS = (
    "provider", "published_time_iso", "modified_time_iso", "expiration",
    # harmless extras we may get from KB
    "mime", "source_type", "rn",
)

CANVAS_SLOTS = { "project_canvas" }
PROJECT_LOG_SLOTS = { "project_log" }

def _is_markdown_mime(m: Optional[str]) -> bool:
    m = (m or "").lower().strip()
    return m in ("text/markdown", "text/x-markdown", "text/md", "markdown")

def _looks_like_markdown(txt: str) -> bool:
    t = (txt or "").strip()
    return ("```" in t) or t.startswith("#") or "\n#" in t

def _is_markdown_from_format_or_text(fmt: Optional[str], mime: Optional[str], txt: str) -> bool:
    f = (fmt or "").lower().strip()
    if f == "markdown":
        return True
    if _is_markdown_mime(mime):
        return True
    return _looks_like_markdown(txt)

# ---- normalized shapes ----
# citations: {url, title, text?}
def _norm_citation(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = (it or {}).get("url") or (it or {}).get("href") or ""
    if not isinstance(url, str) or not url.strip():
        # compat: sometimes older artifacts store under 'value'
        url = str((it or {}).get("value") or "").strip()
    if not url:
        return None

    title = (it or {}).get("title") or (it or {}).get("description") or url
    text  = (it or {}).get("text") or (it or {}).get("body") or (it or {}).get("value_preview") or ""
    sid   = (it or {}).get("sid")
    try:
        sid = int(sid) if sid is not None and str(sid).strip() != "" else None
    except Exception:
        sid = None

    # ⬇️ carry rich attrs if present
    out = {"url": url, "title": title, "text": text}
    if sid is not None:
        out["sid"] = sid
    for k in _CITATION_OPTIONAL_ATTRS:
        if it.get(k) not in (None, ""):
            out[k] = it[k]
    return out

def _pick_canvas_slot(d_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:

    matches = [ d for d in d_items if d.get("slot") in CANVAS_SLOTS ]
    for m in matches:
        v = m.get("value") or {}
        txt = v.get("value") or v.get("value_preview") or ""
        if not txt:
            continue
        fmt  = (v.get("format") or "").lower()
        return { "slot": m, "format": fmt or "markdown", "value": txt }

def _pick_project_log_slot(d_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:

    matches = [ d for d in d_items if d.get("slot") in PROJECT_LOG_SLOTS ]
    for m in matches:
        v = m.get("value") or {}
        txt = v.get("value") or v.get("value_preview") or ""
        if not txt:
            continue
        fmt  = (v.get("format") or "").lower()
        return { "slot": m, "format": fmt or "markdown", "value": txt }

def _history_digest(history: list[dict], limit: int = 3) -> str:
    rows = []
    for h in history[:limit]:
        try:
            exec_id, inner = next(iter(h.items()))
        except Exception:
            continue
        inner = inner or {}
        ts = (inner.get("ts") or "")[:10]
        run = inner.get("codegen_run_id") or exec_id or "?"
        # Prefer project_log if present
        log_txt = ""
        try:
            # project_log is saved as a text inline deliverable or as a string in out_dyn
            log_txt = (inner.get("project_log") or {}).get("text") or ""
        except Exception:
            log_txt = ""
        if not log_txt:
            # fallback to headings from canvas/presentation
            pres = inner.get("program_presentation") or inner.get("solver_failure") or ""
            canvas_txt = ((inner.get("project_canvas") or {}).get("text") or "")
            title = _first_md_heading(pres) or _first_md_heading(canvas_txt) or _short(canvas_txt, 60) or "(no title)"
            rows.append(f"{ts} — {title} [run:{run}]")
        else:
            rows.append(f"{ts} — { _short(log_txt, 800) } [run:{run}]")
    return "; ".join(rows) if rows else "none"


async def _build_program_history_from_turn_ids(self, *,
                                               turn_ids: List[str],
                                               scope: str = "track", days: int = 365) -> List[Dict[str, Any]]:
    """
    For each turn_id, materialize: program presentation (if present), project_canvas / project_log
    from deliverables, and citations tied to the run. Returns the same shape as _build_program_history().
    """
    if not self.context_rag_client:
        return []
    out = []
    seen_runs = set()

    for tid in turn_ids:
        mat = await self.context_rag_client.materialize_turn(
            turn_id=tid, scope=scope, days=days, with_payload=True
        )

        # Unpack rich envelopes (payload + ts + tags)
        prez_env = (mat.get("presentation") or {})
        dels_env = mat.get("deliverables") or {}
        assistant_env = mat.get("assistant") or {}
        user_env = mat.get("user") or {}
        solver_failure_env = mat.get("solver_failure") or {}
        citables_env = mat.get("citables") or {}

        prez = ((prez_env or {}).get("payload") or {}).get("payload") or {}
        dels = ((dels_env or {}).get("payload") or {}).get("payload") or {}
        citables = ((citables_env or {}).get("payload") or {}).get("payload") or {}

        assistant = ((((assistant_env or {}).get("payload") or {}).get("payload") or {})).get("completion") or ""
        user = (((user_env or {}).get("payload") or {}).get("payload") or {}).get("prompt") or ""

        d_items = list((dels or {}).get("items") or [])
        cite_items =  list((citables or {}).get("items") or [])
        round_reason = (dels or {}).get("round_reasoning") or ""

        # Prefer assistant ts, else user ts
        ts_val = assistant_env.get("ts") or user_env.get("ts") or ""

        # codegen_run_id priority: deliverables.payload -> tags -> presentation markdown
        codegen_run_id = (dels or {}).get("execution_id") or ""

        # Presentation markdown (if present)
        pres_md = (prez.get("markdown") or "") if isinstance(prez, dict) else ""

        # Citations bundle (if we have run id)
        cites = {"items": cite_items}

        # Extract canvas/log from deliverables items
        canvas = _pick_canvas_slot(d_items) or {}
        project_log = _pick_project_log_slot(d_items) or {}

        exec_id = codegen_run_id
        if exec_id in seen_runs:
            continue
        seen_runs.add(exec_id)

        # Solver failure (markdown, if any)
        solver_failure = ((solver_failure_env or {}).get("payload") or {}).get("payload") or {}
        solver_failure_md = (solver_failure.get("markdown") or "") if isinstance(solver_failure, dict) else ""

        ret = {
            **({"program_presentation": pres_md} if pres_md else {}),
            **({"project_canvas": {"format": canvas.get("format","markdown"), "text": canvas.get("value","")}} if canvas else {}),
            **({"project_log": {"format": project_log.get("format","markdown"), "text": project_log.get("value","")}} if project_log else {}),
            **({"solver_failure": solver_failure_md} if solver_failure_md else {}),
            **({"web_links_citations": {"items": [_norm_citation(c) for c in cites["items"] if _norm_citation(c)]}}),
            **{"media": []},
            "ts": ts_val,
            **({"codegen_run_id": codegen_run_id} if codegen_run_id else {}),
            **({"round_reasoning": round_reason} if round_reason else {}),
            "assistant": assistant,
            "user": user,
        }
        out.append({exec_id: ret})

    # newest first
    out.sort(key=lambda e: next(iter(e.values())).get("ts","") or "", reverse=True)
    return out

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
_UTM_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id","gclid","fbclid"}

def _normalize_url(u: str) -> str:
    try:
        if not u: return ""
        s = urlsplit(u.strip())
        scheme = (s.scheme or "https").lower()
        netloc = s.netloc.lower().rstrip(":80").rstrip(":443")
        path = s.path or "/"
        # drop anchors
        fragment = ""
        # drop tracking params & keep stable order
        q = [(k,v) for k,v in parse_qsl(s.query, keep_blank_values=True) if k.lower() not in _UTM_PARAMS]
        query = urlencode(q, doseq=True)
        # strip trailing slash for canonicalization (except root)
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        return urlunsplit((scheme, netloc, path, query, fragment))
    except Exception:
        return (u or "").strip()

_CITABLE_TOOL_IDS = {
    "generic_tools.web_search", "generic_tools.browsing",
    "ctx_tools.merge_sources",

}

def _extract_citable_items_from_out_items(out_items: list[dict]) -> list[dict]:
    rows = []
    for r in (out_items or []):
        if not isinstance(r, dict):
            continue
        if r.get("type") != "inline" or not bool(r.get("citable")):
            continue
        tid = (r.get("tool_id") or "").lower()
        if not (tid in _CITABLE_TOOL_IDS or tid.endswith(".kb_search") or tid.endswith(".kb_search_advanced")):
            continue
        out = r.get("output")
        pack = out if isinstance(out, list) else ([out] if isinstance(out, dict) else [])
        for c in pack:
            if isinstance(c, dict):
                url = _normalize_url(str(c.get("url") or c.get("href") or ""))
                if not url:
                    continue
                item = {
                    "url": url,
                    "title": c.get("title") or c.get("description") or url,
                    "text": c.get("text") or c.get("body") or c.get("content") or "",
                    "sid": c.get("sid"),
                    "_tool_id": r.get("tool_id") or "",
                    "_resource_id": r.get("resource_id") or "",
                }
                for k in _CITATION_OPTIONAL_ATTRS:
                    if c.get(k):
                        item[k] = c[k]
                rows.append(item)
    return rows

def _latest_merge_sources_row(out_items: list[dict]) -> list[dict]:
    """Return the output list from the latest ctx_tools.merge_sources (if any)."""
    import re
    latest_idx, latest_row = -1, None
    for r in (out_items or []):
        if not (isinstance(r, dict) and r.get("type") == "inline" and r.get("citable") is True):
            continue
        if (r.get("tool_id") or "") != "ctx_tools.merge_sources":
            continue
        rid = str(r.get("resource_id") or "")
        m = re.search(r":(\d+)$", rid)
        idx = int(m.group(1)) if m else -1
        if idx > latest_idx:
            latest_idx, latest_row = idx, r
    if latest_row and isinstance(latest_row.get("output"), list):
        return latest_row["output"]
    return []

def _rewrite_md_citation_tokens(md: str, sid_map: dict[int,int]) -> str:
    """
    Replace [[S:1,2]] with [[S:a,b]] using sid_map.
    If none of the numbers in a token map, drop the token entirely.
    """
    if not md or not sid_map:
        return md or ""

    def repl(m):
        body = m.group(1)
        nums = []
        for p in body.split(","):
            p = p.strip()
            if not p.isdigit():
                continue
            old = int(p)
            new = sid_map.get(old)
            if new:
                nums.append(str(new))
        if not nums:
            return ""  # drop token
        return f"[[S:{','.join(nums)}]]"

    return re.sub(r"\[\[S:([0-9,\s]+)\]\]", repl, md)

def reconcile_citations_for_context(history: list[dict], *, max_sources: int = 60, rewrite_tokens_in_place: bool = True):
    """
    Input: output of _build_program_history_from_turn_ids (a list of {exec_id: {...}} sorted newest first).
    Output:
      {
        "canonical_sources": [ { sid, url, title, text } ... ],
        "sid_maps": { run_id -> { old_sid -> sid } }
      }
    Side-effect (optional): rewrites [[S:n]] in project_canvas/project_log of each history entry to global sid.
    """
    # 1) Flatten citable items across turns and collect per-run tool outputs
    flat: list[tuple[str, dict, dict]] = []  # (run_id, meta, item)
    per_run_rows: dict[str, list[dict]] = {}
    per_run_out_items: dict[str, list[dict]] = {}

    for rec in (history or []):
        run_id, meta = next(iter(rec.items()))
        per_run_out_items[run_id] = meta.get("out_items") or []  # might not exist; we’ll derive below if needed

    # If out_items not stored, reconstruct from our stored material:
    # We didn't ship out_items in your current _build_program_history_from_turn_ids,
    # so just re-extract from 'citables' and treat them as inline packs.
    for rec in (history or []):
        run_id, meta = next(iter(rec.items()))
        # synthesize an out_items-like list from web_links_citations, so extraction works the same
        citations_list = ((meta.get("web_links_citations") or {}).get("items") or [])
        faux_out_item = {
            "type": "inline",
            "citable": True,
            "tool_id": "ctx_tools.merge_sources",  # make them act as a merged pack for this run
            "resource_id": f"tool:ctx_tools.merge_sources:0",
            "output": citations_list
        }
        out_items = [faux_out_item]
        per_run_out_items[run_id] = out_items

        # Extract items
        ext = _extract_citable_items_from_out_items(out_items)
        per_run_rows[run_id] = ext
        for it in ext:
            flat.append((run_id, meta, it))

    # 2) Determine base ordering: prefer latest run’s merge_sources order if present; else newest-first appearance order.
    latest_run_id = next(iter(history[0].keys())) if history else None
    base_order_urls: list[str] = []
    if latest_run_id:
        base_merge = _latest_merge_sources_row(per_run_out_items.get(latest_run_id, []))
        if base_merge:
            base_order_urls = [_normalize_url(c.get("url") or "") for c in base_merge if isinstance(c, dict)]
            base_order_urls = [u for u in base_order_urls if u]

    # 3) Build canonical map (first seen wins after base order), assign global SIDs
    by_url: dict[str, dict] = {}
    ordered_urls: list[str] = []

    # Seed with base order
    for u in base_order_urls:
        if u not in by_url:
            by_url[u] = {}
            ordered_urls.append(u)

    # Add remaining (newest-first by history order)
    for run_id, meta, it in flat:
        u = it["url"]
        if u not in by_url:
            by_url[u] = {
                "url": u,
                "title": it["title"],
                "text": it.get("text",""),
            }
            # ⬇️ copy rich attrs if present
            for k in  _CITATION_OPTIONAL_ATTRS:
                if it.get(k):
                    by_url[u][k] = it[k]
            ordered_urls.append(u)

    # Limit length
    ordered_urls = ordered_urls[:max_sources]

    # Assign global SIDs (1..N) deterministically via current order
    global_sid_of_url: dict[str,int] = {u: i+1 for i,u in enumerate(ordered_urls)}

    canonical_sources = []
    for u in ordered_urls:
        src = by_url[u]
        row = {
            "sid": global_sid_of_url[u],
            "url": u,
            "title": src.get("title") or u,
            "text": src.get("text",""),
        }
        for k in _CITATION_OPTIONAL_ATTRS:
            if src.get(k):
                row[k] = src[k]
        canonical_sources.append(row)

    # 4) Build per-run sid maps (old → global)
    sid_maps: dict[str, dict[int,int]] = {}
    for run_id, rows in per_run_rows.items():
        m: dict[int,int] = {}
        for it in rows:
            u = it["url"]
            old_sid = it.get("sid")
            if old_sid is None:
                continue
            try:
                old_sid = int(old_sid)
            except Exception:
                continue
            new_sid = global_sid_of_url.get(u)
            if new_sid:
                m[old_sid] = new_sid
        if m:
            sid_maps[run_id] = m

    # 5) Optionally rewrite [[S:n]] tokens inside each run’s markdown blobs
    if rewrite_tokens_in_place:
        for rec in history:
            run_id, meta = next(iter(rec.items()))
            sid_map = sid_maps.get(run_id, {})
            if not sid_map:
                continue
            # rewrite project_canvas / project_log if present
            if "project_canvas" in meta and isinstance(meta["project_canvas"], dict):
                val = meta["project_canvas"].get("text") or meta["project_canvas"].get("value") or ""
                if val:
                    new_val = _rewrite_md_citation_tokens(val, sid_map)
                    if "text" in meta["project_canvas"]:
                        meta["project_canvas"]["text"] = new_val
                    else:
                        meta["project_canvas"]["value"] = new_val
            if "project_log" in meta and isinstance(meta["project_log"], dict):
                val = meta["project_log"].get("text") or meta["project_log"].get("value") or ""
                if val:
                    new_val = _rewrite_md_citation_tokens(val, sid_map)
                    if "text" in meta["project_log"]:
                        meta["project_log"]["text"] = new_val
                    else:
                        meta["project_log"]["value"] = new_val

    return {
        "canonical_sources": canonical_sources,
        "sid_maps": sid_maps,
    }
