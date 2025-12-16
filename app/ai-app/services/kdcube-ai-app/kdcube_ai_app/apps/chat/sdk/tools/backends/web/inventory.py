# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/tools/backends/web/presentation.py

from typing import Dict, Optional, Iterable, Any, List
from dataclasses import dataclass
import urllib.parse
from collections import deque

from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import SOURCE_ID_CV

UNKNOWN = "unknown"
DUCKDUCKGO = "duckduckgo"
BRAVE = "brave"
EXA = "exa"
SERPER = "serper"

PROVIDERS_AUTHORITY_RANK = {
    UNKNOWN: 0,
    DUCKDUCKGO: 1,
    BRAVE: 2,
    EXA: 3,
    SERPER: 4
}
def compose_search_results_html(
        *,
        objective: str,
        queries: list[str],
        initial_rows: list[dict],
        reconciled_rows: list[dict] | None = None,
        filtered_rows: list[dict] | None = None,
        title: str = "Web Search Results",
) -> str:
    """
    Build a self-contained HTML page showing:
      - Filtered survivors (top, with ✓ tick)
      - Reconciled (kept) results, with relevance badges
      - All initially found items (deduped), at the bottom

    Expected row shape (dict keys):
      - sid: int
      - title: str
      - url: str
      - body: str (snippet)
      - objective_relevance: float   # optional (reconciled)
      - query_relevance: float       # optional (reconciled)
      - content: str                 # optional (fetched content)
    """
    import html

    def esc(s: str) -> str:
        return html.escape(s or "")

    def badge(label: str) -> str:
        return f'<span class="badge">{esc(label)}</span>'

    def score_badge(v: float | None, label: str) -> str:
        if v is None:
            return ""
        pct = max(0, min(100, round(float(v)*100))) if v <= 1.0 else max(0, min(100, round(float(v))))
        # Color by band
        cls = "score-low"
        if pct >= 90:
            cls = "score-hi"
        elif pct >= 70:
            cls = "score-med"
        return f'<span class="score {cls}" title="{esc(label)}">{esc(label)} {pct}%</span>'

    def row_html(r: dict, *, tick=False, show_scores=False) -> str:
        sid = r.get("sid")
        title = r.get("title") or r.get("url") or f"Result {sid}"
        url = r.get("url") or ""
        body = (r.get("body") or r.get("text") or "").strip()
        # badges
        b = []
        if tick:
            b.append('<span class="tick" title="Survived content filter">✓</span>')
        if show_scores:
            b.append(score_badge(r.get("objective_relevance"), "Objective"))
            b.append(score_badge(r.get("query_relevance"), "Query"))
        badges = "".join(b)

        return f"""
        <article class="card" data-sid="{sid}">
          <header class="card-hd">
            <a class="title" href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(title)}</a>
            <div class="badges">{badges}</div>
          </header>
          <div class="meta">
            <code class="sid">SID: {sid}</code>
            <a class="link" href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(url)}</a>
          </div>
          <p class="snippet">{esc(body)}</p>
        </article>
        """

    def section(title: str, rows: list[dict], *, tick=False, show_scores=False) -> str:
        if not rows:
            return ""
        items = "\n".join(row_html(r, tick=tick, show_scores=show_scores) for r in rows)
        return f"""
        <section class="section">
          <h2>{esc(title)}</h2>
          {items}
        </section>
        """

    # To avoid duplicate repetition across sections, keep each list as provided
    filtered_rows = filtered_rows or []
    reconciled_rows = reconciled_rows or []
    initial_rows = initial_rows or []

    # Inline CSS (printable & dark-mode friendly enough)
    css = """
    :root { --bg:#fff; --fg:#111; --muted:#666; --line:#e5e7eb; --brand:#2563eb; --ok:#16a34a; --lo:#f59e0b; --med:#06b6d4; --hi:#10b981; }
    @media (prefers-color-scheme: dark) {
      :root { --bg:#0b0b0c; --fg:#e9e9ea; --muted:#9aa0a6; --line:#26282c; --brand:#60a5fa; --ok:#22c55e; --lo:#fbbf24; --med:#22d3ee; --hi:#34d399; }
    }
    *{box-sizing:border-box} html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,Segoe UI,Roboto,Helvetica,Arial}
    main{max-width:980px;margin:0 auto;padding:24px}
    header.page h1{font-size:1.4rem;margin:0 0 6px}
    header.page .sub{color:var(--muted);font-size:.95rem}
    .pill{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:999px;margin-right:6px;color:var(--muted);font-size:.85rem}
    .section{margin-top:28px}
    .section>h2{font-size:1.1rem;border-bottom:1px solid var(--line);padding-bottom:6px;margin:0 0 12px}
    .card{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:12px 0;background:rgba(0,0,0,.02)}
    .card-hd{display:flex;gap:10px;align-items:baseline;justify-content:space-between}
    .title{font-weight:600;text-decoration:none;color:var(--fg)}
    .title:hover{color:var(--brand)}
    .badges{display:flex;gap:6px;align-items:center}
    .tick{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:999px;background:var(--ok);color:#fff;font-weight:700}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;background:transparent;border:1px solid var(--line);color:var(--muted);font-size:.8rem}
    .score{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:600;font-size:.8rem;border:1px solid var(--line)}
    .score-low{background:rgba(245,158,11,.15)}
    .score-med{background:rgba(6,182,212,.15)}
    .score-hi{background:rgba(16,185,129,.15)}
    .meta{display:flex;gap:10px;align-items:center;margin:6px 0 8px}
    .sid{background:transparent;color:var(--muted);border:1px dashed var(--line);padding:2px 6px;border-radius:6px}
    .link{color:var(--brand);text-decoration:none;font-size:.92rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70%}
    .link:hover{text-decoration:underline}
    .snippet{margin:.2rem 0 0;color:var(--fg)}
    .topline{margin:10px 0 0;display:flex;flex-wrap:wrap;gap:6px}
    """

    q_pills = "".join(f'<span class="pill">Q: {esc(q)}</span>' for q in (queries or []))
    header_html = f"""
    <header class="page">
      <h1>{esc(title)}</h1>
      <div class="sub">Objective: {esc(objective or '')}</div>
      <div class="topline">{q_pills}</div>
    </header>
    """

    html_doc = f"""<!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>{esc(title)}</title>
        <style>{css}</style>
      </head>
      <body>
        <main>
          {header_html}
          {section("Filtered (Survivors ✓)", filtered_rows, tick=True, show_scores=True)}
          {section("Reconciled (Kept)", reconciled_rows, tick=False, show_scores=True)}
          {section("Initially Found", initial_rows, tick=False, show_scores=False)}
        </main>
      </body>
    </html>
    """
    return html_doc.strip()

def _claim_sid_block(n: int) -> int:
    """Reserve a contiguous block of n SIDs, return the base (first) SID."""
    n = max(int(n or 0), 0)
    st = SOURCE_ID_CV.get() or {}
    try:
        base = int(st.get("next", 1))
    except (TypeError, ValueError):
        base = 1
    st["next"] = base + n
    SOURCE_ID_CV.set(st)
    return base

def _normalize_url(u: str) -> str:
    # best-effort normalization for dedup
    try:
        p = urllib.parse.urlsplit(u.strip())
        p = p._replace(fragment="", query=p.query)  # keep query, drop fragment
        # normalize scheme/host only if present
        scheme = (p.scheme or "http").lower()
        netloc = p.netloc.lower()
        # remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        return urllib.parse.urlunsplit((scheme, netloc, p.path, p.query, ""))
    except Exception:
        return (u or "").strip()


# ----------------------------- Errors -----------------------------
class SearchBackendError(RuntimeError):
    pass

# ----------------------------- Canonical enums (string-based) -----------------------------

CANON_FRESHNESS = {None, "day", "week", "month", "year"}
CANON_SAFESEARCH = {"off", "moderate", "strict"}


def normalize_freshness(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    # allow a couple of friendly aliases
    aliases = {
        "d": "day",
        "w": "week",
        "m": "month",
        "y": "year",
        "today": "day",
        "24h": "day",
        "7d": "week",
        "30d": "month",
        "365d": "year",
        "any": None,
        "all": None,
    }
    v = aliases.get(v, v)
    if v not in CANON_FRESHNESS:
        return None
    return v

def normalize_country(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    # Canonical policy: prefer ISO-3166 alpha-2 uppercase.
    # If someone passes "de-DE" or "us-en" we degrade to the first 2 letters.
    v = v.replace("_", "-")
    if len(v) >= 2:
        cc = v[:2].upper()
        if cc.isalpha():
            return cc
    return None


def normalize_safesearch(value: Optional[str]) -> str:
    if value is None:
        return "moderate"
    v = str(value).strip().lower()
    if not v:
        return "moderate"
    aliases = {
        "safe": "strict",
        "on": "strict",
        "true": "strict",
        "false": "off",
        "none": "off",
        "medium": "moderate",
        "default": "moderate",
    }
    v = aliases.get(v, v)
    if v not in CANON_SAFESEARCH:
        return "moderate"
    return v


def clamp_max_results(n: int, *, lo: int = 1, hi: int = 100) -> int:
    try:
        x = int(n)
    except Exception:
        x = 10
    return max(lo, min(hi, x))


# ----------------------------- Data shapes -----------------------------

@dataclass(frozen=True)
class SearchRequest:
    query: str
    max_results: int = 10
    freshness: Optional[str] = None
    country: Optional[str] = None
    safesearch: str = "moderate"

    def normalized(self) -> "SearchRequest":
        q = (self.query or "").strip()
        if not q:
            raise ValueError("Query cannot be empty")
        return SearchRequest(
            query=q,
            max_results=clamp_max_results(self.max_results, lo=1, hi=100),
            freshness=normalize_freshness(self.freshness),
            country=normalize_country(self.country),
            safesearch=normalize_safesearch(self.safesearch),
        )


# Normalized hit notation for your pipeline
def make_hit(title: str, url: str, text: str, vendor: str) -> Dict[str, str]:
    return {
        "title": (title or "").strip(),
        "url": (url or "").strip(),
        "text": (text or "").strip(),
        "vendor": vendor,
    }

def dedup_round_robin_ranked(
        per_query_results: List[Iterable[Dict[str, Any]]],
        n: int,
        default_provider_name: Optional[str] = UNKNOWN,
        authority_rank: Optional[Dict[str, int]] = None,
):
    """
    Merges multiple result streams into a single list of unique rows, using:

    1) **Round-robin sampling** across streams (fair interleaving)
    2) **URL-based deduplication** (after `_normalize_url(...)`)
    3) **Provider authority ranking** to decide which duplicate “wins”

    It returns up to `n` unique items, each with an ephemeral `sid` (1..n).

    :param per_query_results: A list of result collections (one per provider/query). Each element is iterated once.
    Each hit may contain:
    - url (required to dedupe; empty/invalid URLs are skipped)
    - title (optional)
    - body (optional; mapped to output text)
    - vendor (optional; used as provider name)
    :param n: Maximum number of unique URLs to return.
    :param default_provider_name: Default provider label if a hit has no vendor
    :param authority_rank: Provider → rank mapping. Higher number = higher authority.
    Providers not present in this mapping are treated as rank 0.
    :return:
    """
    if not authority_rank:
        authority_rank = PROVIDERS_AUTHORITY_RANK
    # Normalize rank keys once
    rank_map = {k.lower(): int(v) for k, v in authority_rank.items()}

    def get_rank(vendor: str | None) -> int:
        if not vendor:
            return 0
        return rank_map.get(vendor.lower(), 0)

    streams = [iter(res) for res in per_query_results]
    active = deque(range(len(streams)))  # indices still in rotation

    unique: Dict[str, Dict[str, Any]] = {}   # url -> row
    best_rank: Dict[str, int] = {}           # url -> rank of chosen provider

    while active and len(unique) < n:
        idx = active.popleft()

        try:
            hit = next(streams[idx])
            active.append(idx)  # still active, put it back into rotation
        except StopIteration:
            continue

        url = _normalize_url(hit.get("url", ""))
        if not url:
            continue

        vendor = hit.get("vendor") or default_provider_name
        r = get_rank(vendor)

        incoming = {
            "title": hit.get("title") or "",
            "url": url,
            "text": hit.get("body") or hit.get("text") or "",
            "provider": vendor,
        }

        if url not in unique:
            unique[url] = incoming
            best_rank[url] = r
            continue

        # Duplicate: keep the entry from the higher-rank provider
        cur = unique[url]
        cur_r = best_rank[url]

        if r > cur_r:
            # Higher-rank wins; keep it as primary, but don't throw away info if it lacks fields.
            unique[url] = {
                "title": incoming["title"] or cur.get("title", ""),
                "url": url,
                "text": incoming["text"] or cur.get("text", ""),
                "provider": incoming["provider"],  # winning provider
            }
            best_rank[url] = r
        else:
            # Current stays primary; optionally backfill missing fields from lower rank
            if not cur.get("title") and incoming["title"]:
                cur["title"] = incoming["title"]
            if not cur.get("text") and incoming["text"]:
                cur["text"] = incoming["text"]
            if not cur.get("provider") and incoming["provider"]:
                cur["provider"] = incoming["provider"]

    # Build rows (stable insertion order; replacing a url keeps its position)
    rows = []
    for sid, row in enumerate(unique.values(), 1):
        rows.append({
            "sid": sid,
            "title": row.get("title", ""),
            "url": row["url"],
            "text": row.get("text", ""),
            "provider": row.get("provider") or default_provider_name,
        })
        if len(rows) >= n:
            break

    return rows