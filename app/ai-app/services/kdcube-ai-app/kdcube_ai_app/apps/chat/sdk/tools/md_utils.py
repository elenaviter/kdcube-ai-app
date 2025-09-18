# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/md_utils.py

import re, json
from typing import Optional, Dict, List


def _superscript_num(n: int) -> str:
    _map = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}
    return "".join(_map.get(ch, ch) for ch in str(n))

def build_citation_map(citations: List[Dict]) -> Dict[int, Dict]:
    """Build citation map from sr.citations() format"""
    by_id = {}
    for c in citations:
        sid = c.get("sid")
        if sid is not None:
            by_id[int(sid)] = {
                "url": c.get("url", ""),
                "title": c.get("title", ""),
                "text": c.get("text", "")
            }
    return by_id

def _normalize_sources(sources_json: Optional[str]) -> tuple[dict[int, dict], list[int]]:
    """
    Accepts:
      - JSON array of objects: [{sid?, title, url, ...}, ...] (sid is 1-based; if missing, index+1 is used)
      - or JSON object: { "1": {title,url}, "2": {...}, ... }
    Returns:
      (by_id, order_ids) where by_id: {sid:int -> {title,url,...}}, order_ids is the ordered list of sids.
    """
    if not sources_json:
        return {}, []
    try:
        src = json.loads(sources_json)
    except Exception:
        return {}, []
    by_id: dict[int, dict] = {}
    order: list[int] = []

    if isinstance(src, list):
        for i, row in enumerate(src):
            if not isinstance(row, dict):
                continue
            sid = row.get("sid")
            if sid is None:
                sid = i + 1
            try:
                sid = int(sid)
            except Exception:
                continue
            by_id[sid] = row
            order.append(sid)
    elif isinstance(src, dict):
        for k, row in src.items():
            try:
                sid = int(k)
            except Exception:
                continue
            if isinstance(row, dict):
                by_id[sid] = row
                order.append(sid)
    return by_id, order

def _replace_citation_tokens(md: str, by_id: dict[int, dict]) -> str:
    """
    Replace [[S:1]] or [[S:1,4]] or [[S:1-15]] with inline links:
      [[S:3]] -> [³](https://example "Title")
      [[S:1,4]] -> [¹](url1 "Title1") [⁴](url4 "Title4")
      [[S:1-15]] -> [¹](url1 "Title1") [²](url2 "Title2") ... [¹⁵](url15 "Title15")
    Unknown ids are dropped from the replacement; if none are known, the token is removed.
    """
    if not by_id:
        return md

    pat = re.compile(r"\[\[S:([0-9,\s\-]+)]]")

    def _one(m: re.Match) -> str:
        ids_str = m.group(1)
        ids = []

        # Handle both comma-separated and ranges
        parts = ids_str.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                # Handle range like "1-15"
                try:
                    start, end = part.split("-", 1)
                    start_num = int(start.strip())
                    end_num = int(end.strip())
                    ids.extend(range(start_num, end_num + 1))
                except ValueError:
                    continue
            elif part.isdigit():
                ids.append(int(part))

        pieces = []
        for i in ids:
            meta = by_id.get(i)
            if not meta:
                continue
            url = meta.get("url") or meta.get("href")
            title = (meta.get("title") or url or "").replace('"', "'")
            if not url:
                continue
            sup = _superscript_num(i)
            pieces.append(f"[{sup}]({url} \"{title}\")")
        return " " + " ".join(pieces) if pieces else ""

    return pat.sub(_one, md)

def _append_sources_section(md: str, by_id: dict[int, dict], order: list[int]) -> str:
    """
    If the doc doesn't already contain a '## Sources' header, append one with numbered links.
    """
    if not by_id or not order:
        return md
    # rough check to avoid duplicating a section the caller already added
    if re.search(r"^##\s+Sources\b", md, flags=re.IGNORECASE | re.MULTILINE):
        return md
    lines = ["", "---", "", "## Sources", ""]
    for sid in order:
        meta = by_id.get(sid) or {}
        url = meta.get("url") or meta.get("href") or ""
        title = meta.get("title") or url
        if not url:
            continue
        lines.append(f"{sid}. [{title}]({url})")
    return md + "\n".join(lines) + "\n"

def replace_citation_tokens_streaming(text: str, citation_map: Dict[int, Dict]) -> str:
    if not citation_map:
        return text

    pat = re.compile(r"\[\[S:([0-9,\s\-]+)]]")

    def _replace_one(m: re.Match) -> str:
        ids_str = m.group(1)
        ids = []

        parts = ids_str.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    start_num = int(start.strip())
                    end_num = int(end.strip())
                    ids.extend(range(start_num, end_num + 1))
                except ValueError:
                    continue
            elif part.isdigit():
                ids.append(int(part))

        pieces = []
        for i in ids:
            meta = citation_map.get(i)
            if not meta:
                continue
            url = meta.get("url") or ""
            title = (meta.get("title") or url or "").replace('"', "'")
            if not url:
                continue
            pieces.append(f"[{title}]({url})")
        return " " + " ".join(pieces) if pieces else ""

    return pat.sub(_replace_one, text)

def has_incomplete_citation_token(text: str) -> bool:
    """Check if text ends with an incomplete citation token that might be cut off"""
    # Look for partial patterns at the end that might be incomplete citation tokens
    patterns = [
        r'\[\[?$',           # [[
        r'\[\[S:?$',         # [[S:
        r'\[\[S:[0-9,\s]*(?!\]\])$',  # [[S:1,2 (but no closing ]]
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False