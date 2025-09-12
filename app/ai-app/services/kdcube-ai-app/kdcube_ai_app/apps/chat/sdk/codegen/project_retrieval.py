# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/codegen/project_retrieval.py

from __future__ import annotations
from typing import Any, Dict, List, Optional

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
def _norm_citation(it: Dict[str, Any]) -> Optional[Dict[str, str]]:
    url = (it or {}).get("url") or ""
    if not isinstance(url, str) or not url.strip():
        # compat: sometimes older artifacts store under 'value'
        url = str((it or {}).get("value") or "").strip()
    if not url:
        return None
    title = (it or {}).get("title") or (it or {}).get("description") or url
    text = (it or {}).get("text") or (it or {}).get("body") or (it or {}).get("value_preview") or ""
    return {"url": url, "title": title, "text": text}

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

async def _build_program_history(self,
                                 user_text: str,
                                 scope: str = "track",
                                 days: int = 365,
                                 top_k: int = 6,
                                 prefer_last_runs: bool = True) -> List[Dict[str, Any]]:
    """
    Collect up to top_k recent codegen runs and their artifacts with minimal passes.

    Strategy:
      1) Pull the LAST N program presentations in-track (recency; no embeddings).
      2) For each run, fetch *both* deliverables and citations in a single call.
      3) Normalize citations to {url,title,text?}. Deliverables remain single-value per slot.

    Returns a list:
      [
        {
          "<exec_id>": {
            "program_presentation": <md>,
            "project_canvas": { "format": "markdown"|"text", "text": <str> },
            "project_log": { "format": "markdown"|"text", "text": <str> },
            "web_links_citations": { "items": [ {url,title,text?}, ... ] },
            "media": [],
            "ts": <iso>,
            "codegen_run_id": <run>,
            "round_reasoning": <str>
          }
        },
        ...
      ]
    """
    if not self.context_rag_client:
        return []

    # 1) recent N presentations (fast; recency)
    if prefer_last_runs:
        pres = await self.context_rag_client.recent(
            kinds=("codegen.program.presentation",),
            scope=scope, days=days, limit=top_k,
            with_payload=True
        )
        pres_items = pres.get("items") or []
        # fallback if none found: do semantic as backup
        if not pres_items:
            search_res = await self.context_rag_client.search(
                query=(user_text or " "),
                kinds=("codegen.program.presentation",),
                scope=scope, days=days, top_k=top_k,
                include_deps=False, with_payload=True, sort="recency"
            )
            pres_items = search_res.get("items") or []
    else:
        search_res = await self.context_rag_client.search(
            query=(user_text or " "),
            kinds=("codegen.program.presentation",),
            scope=scope, days=days, top_k=top_k,
            include_deps=False, with_payload=True, sort="recency"
        )
        pres_items = search_res.get("items") or []

    out: List[Dict[str, Any]] = []
    for hit in pres_items:
        paydoc = hit.get("payload") or {}
        meta = paydoc.get("meta") or {}
        payl = paydoc.get("payload") or {}
        md   = payl.get("markdown") or ""
        run  = (meta.get("codegen_run_id") or "").strip() or "cg-unknown"
        ts   = hit.get("ts")

        # 2) one-pass fetch for this run (deliverables + citables)
        bundle = await self.context_rag_client.get_run_artifacts(codegen_run_id=run, scope=scope, days=days, with_payload=True)
        ddoc = bundle.get("deliverables") or {}
        cdoc = bundle.get("citables") or {}

        # Deliverables shape: {"items":[{"slot","description","value":{...}},...], "round_reasoning":...}
        d_items = list((ddoc.get("items") or []))
        round_reason = ddoc.get("round_reasoning") or ""

        # Citations shape: {"items":[{url,title,text?},...]}
        cites_raw = list((cdoc.get("items") or []))
        cites: List[Dict[str, str]] = []
        for it in cites_raw:
            nc = _norm_citation(it)
            if nc:
                cites.append(nc)

        canvas_slot = _pick_canvas_slot(d_items) or {}
        project_log_slot = _pick_project_log_slot(d_items) or {}

        media: List[Dict[str, Any]] = []  # hook for future image/media bundles

        exec_id = run  # stable key
        out.append({
            exec_id: {
                "program_presentation": md,
                "project_canvas": { "format": canvas_slot.get("format","markdown"), "text": canvas_slot.get("text","") },
                "project_log": { "format": project_log_slot.get("format","markdown"), "text": project_log_slot.get("text","") },
                "web_links_citations": { "items": cites },
                "media": media,
                "ts": ts,
                "codegen_run_id": run,
                "round_reasoning": round_reason
            }
        })

    # newest first by ts
    out.sort(key=lambda e: next(iter(e.values())).get("ts",""), reverse=True)
    return out

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
            pres = inner.get("program_presentation") or ""
            canvas_txt = ((inner.get("project_canvas") or {}).get("text") or "")
            title = _first_md_heading(pres) or _first_md_heading(canvas_txt) or _short(canvas_txt, 60) or "(no title)"
            rows.append(f"{ts} — {title} [run:{run}]")
        else:
            rows.append(f"{ts} — { _short(log_txt, 80) } [run:{run}]")
    return "; ".join(rows) if rows else "none"
