# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/retrieval/turn_retriever.py

from typing import List, Dict, Any, Optional, Tuple

from kdcube_ai_app.apps.chat.sdk.context.policy.policy import PolicyResult
from kdcube_ai_app.apps.chat.sdk.util import slug

def _slug_tokens(s: str) -> List[str]:
    s = slug(s)
    parts = [p for p in s.split("-") if p]
    toks = set(parts)

    # Optional: add lightweight bigrams to catch short phrases like "cloud posture"
    for i in range(len(parts) - 1):
        toks.add(f"{parts[i]}_{parts[i+1]}")

    return list(toks)

def _item_tokens(item: Dict[str,Any]) -> List[str]:
    toks = set()
    # tags/metadata/provider/title are common carriers
    for t in (item.get("tags") or []):
        toks.update(_slug_tokens(str(t)))
    prov = item.get("provider")
    if prov: toks.update(_slug_tokens(str(prov)))
    title = item.get("title") or item.get("summary") or item.get("heading") or ""
    if title: toks.update(_slug_tokens(title))
    # light-touch from content (first 10 words)
    content = (item.get("content") or item.get("text") or "")
    if content:
        head = " ".join(str(content).split()[:10])
        toks.update(_slug_tokens(head))
    return list(toks)

def _key_hits_tokens(key: str, tokens: List[str]) -> bool:
    ks = _slug_tokens(key)
    for k in ks:
        for t in tokens:
            # exact match first
            if k == t:
                return True
            # allow anchored prefix/suffix matches for reasonably specific strings
            if len(k) >= 4 and (t.startswith(k) or t.endswith(k) or k.startswith(t) or k.endswith(t)):
                return True
    return False

def apply_policy_rerank(policy: PolicyResult, conv_hits: List[Dict[str,Any]], kb_hits: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Generic policy-aware reranker:
      - Drop items that fuzzy-match any avoid[key]
      - Boost items that fuzzy-match any do[key]
      - allow_if acts as a small boost (document passes if matched elsewhere)
    """
    avoid_keys = list((policy.get("avoid") or {}).keys())
    do_keys    = list((policy.get("do") or {}).keys())
    allow_keys = list((policy.get("allow_if") or {}).keys())

    def score_item(item: Dict[str,Any]) -> float:
        base = float(item.get("score") or item.get("semantic_score") or 0.5)
        toks = _item_tokens(item)

        # hard drop if any avoid matches
        for k in avoid_keys:
            if _key_hits_tokens(k, toks):
                return -1.0

        # boosts
        boost = 0.0
        for k in do_keys:
            if _key_hits_tokens(k, toks):
                boost += 0.10
        for k in allow_keys:
            if _key_hits_tokens(k, toks):
                boost += 0.05

        return base + boost

    pool = []
    for x in conv_hits:
        y = dict(x); y["__origin"] = "conv"; y["__s"] = score_item(y); pool.append(y)
    for x in kb_hits:
        y = dict(x); y["__origin"] = "kb";  y["__s"] = score_item(y); pool.append(y)

    pool = [p for p in pool if p["__s"] >= 0.0]
    pool.sort(key=lambda z: (z["__s"], z.get("created_at") or 0), reverse=True)
    return pool[:12]

