# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/codegen/project_retrieval.py

def _first_md_heading(md: str) -> str:
    for ln in (md or "").splitlines():
        t = ln.strip()
        if t.startswith("#"):
            return t.lstrip("# ").strip()
    return ""

def _short(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

EDITABLE_SLOT_NAMES = {"editable_md","document_md","draft_md","summary_md","outline_md","body_md","report_md","article_md","content_md"}
def _is_markdown_mime(m):
    return (m or "").lower() in ("text/markdown","text/x-markdown","text/md","markdown")
def _pick_editable_from_deliverables(d_items: list[dict]) -> dict | None:
    # prefer explicit editable slots; else first inline markdown; else first inline text
    cand = None
    for it in d_items or []:
        slot = (it or {}).get("slot") or ""
        vals = list((it or {}).get("values") or [])
        for v in vals:
            if v.get("type") != "inline":
                continue
            mime = (v.get("mime") or "").lower()
            txt  = v.get("value") or v.get("value_preview") or ""
            if slot in EDITABLE_SLOT_NAMES and txt:
                return {"slot": slot, "format": ("markdown" if _is_markdown_mime(mime) else "text"), "text": txt}
            if _is_markdown_mime(mime) and txt and cand is None:
                cand = {"slot": slot or "editable_md", "format": "markdown", "text": txt}
            elif txt and cand is None:
                cand = {"slot": slot or "editable_md", "format": "text", "text": txt}
    return cand


async def _build_program_history(self, user_text: str, scope: str = "track", days: int = 365, top_k: int = 6) -> list[dict]:
    if not self.context_rag_client:
        return []
    # 1) Find relevant program presentations (with payloads)
    pres = await self.context_rag_client.search(
        query=(user_text or " "),
        kinds=("codegen.program.presentation",),
        scope=scope, days=days, top_k=top_k,
        include_deps=False, with_payload=True
    )
    out: list[dict] = []
    for hit in (pres.get("items") or []):
        full = hit.get("payload") or {}
        meta = full.get("meta") or {}
        payl = full.get("payload") or {}
        md   = payl.get("markdown") or ""
        run  = (meta.get("codegen_run_id") or "").strip() or "cg-unknown"
        ts   = hit.get("ts")

        # 2) Pull deliverables for this run (for editable)
        d_items, round_reason = [], ""
        try:
            dlv = await self.context_rag_client.search(
                query=" ",
                kinds=("codegen.program.out.deliverables", f"codegen_run:{run}"),
                scope=scope, days=days, top_k=1,
                include_deps=False, with_payload=True
            )
            if dlv.get("items"):
                ddoc = dlv["items"][0].get("payload") or {}
                dp   = (ddoc.get("payload") or {})
                d_items = list(dp.get("items") or [])
                round_reason = dp.get("round_reasoning") or ""
        except Exception:
            pass

        editable = _pick_editable_from_deliverables(d_items) or {}

        # 3) Citations for this run → normalized to sid/title/url/text/…
        cites = []
        try:
            cit = await self.context_rag_client.search(
                query=" ",
                kinds=("codegen.program.citables", f"codegen_run:{run}"),
                scope=scope, days=days, top_k=1,
                include_deps=False, with_payload=True
            )
            if cit.get("items"):
                cdoc = cit["items"][0].get("payload") or {}
                cp   = (cdoc.get("payload") or {})
                items = list(cp.get("items") or [])
                sid = 1
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    url = it.get("value") or it.get("url") or ""
                    title = it.get("description") or it.get("title") or (url[:80] if url else "source")
                    body = it.get("body") or it.get("text") or it.get("value_preview") or ""
                    cites.append({
                        "sid": sid, "title": title, "url": url, "text": body,
                        "tool_id": it.get("tool_id"), "description": it.get("description") or "",
                        "resource_id": it.get("resource_id") or ""
                    })
                    sid += 1
        except Exception:
            pass

        # 4) Media (images) — optional; leave empty unless you already mirrored assets per run
        media = []

        # 5) Build exec entry keyed by exec_id (prefer an execution_id if you persist it; else run id)
        exec_id = run  # fallback; if you persist a dedicated 'execution_id', substitute it here
        out.append({
            exec_id: {
                "program_presentation": md,
                "project_canvas": { "format": editable.get("format","markdown"), "text": editable.get("text","") },
                "web_links_citations": { "items": cites },
                "media": media,
                "ts": ts,
                "codegen_run_id": run,
                "round_reasoning": round_reason
            }
        })
    # newest first by their inner ts
    out.sort(key=lambda e: next(iter(e.values())).get("ts",""), reverse=True)
    return out


def _history_digest(history: list[dict], limit: int = 3) -> str:
    rows = []
    for h in history[:limit]:
        ts = (h.get("ts") or "")[:10]
        run = h.get("codegen_run_id") or "?"
        slots = sum(len(s.get("values") or []) for s in (h.get("deliverables") or []))
        cites = len(h.get("citations") or [])
        rows.append(f"{ts} — {h.get('title','')} [run:{run}] slots:{slots} cites:{cites}")
    return "; ".join(rows) if rows else "none"

