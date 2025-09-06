# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/md_utils.py

import re, json
from typing import Optional


def _superscript_num(n: int) -> str:
    _map = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}
    return "".join(_map.get(ch, ch) for ch in str(n))

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
    Replace [[S:1]] or [[S:1,4]] with inline links:
      [[S:3]] -> [³](https://example "Title")
      [[S:1,4]] -> [¹](url1 "Title1") [⁴](url4 "Title4")
    Unknown ids are dropped from the replacement; if none are known, the token is removed.
    """
    if not by_id:
        return md

    pat = re.compile(r"\[\[S:([0-9,\s]+)\]\]")

    def _one(m: re.Match) -> str:
        ids_str = m.group(1)
        ids = []
        for part in ids_str.split(","):
            part = part.strip()
            if part.isdigit():
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