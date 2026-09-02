# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Query-side term shaping for the lexical and trigram arms of conversation search.

Both arms take the agent's query string as-is. The lexical arm hands it to
Postgres ``websearch_to_tsquery``, which ANDs every unquoted word, so one
conversational filler word ("last time i worked with excel") empties the
result. The trigram arm averages ``word_similarity`` over the query tokens, so
the same filler words drag a real match below the threshold.

The instructions tell the model to send content words only. These helpers make
the arms tolerant when it does not:

- ``lexical_or_fallback_query`` rewrites the query into ``websearch_to_tsquery``
  OR syntax (quoted phrases kept intact) for a second pass that runs only when
  the AND pass returned nothing.
- ``trigram_query_tokens`` drops stop words and short tokens before the
  similarity average, so filler cannot dilute the score of the words that
  carry the topic.
"""

from __future__ import annotations

import re
from typing import List

# Words that carry no retrieval signal in a memory query. English stop words
# plus the conversational-recall framing the model is told not to send.
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "on", "in",
    "at", "to", "for", "from", "by", "with", "about", "into", "over", "under",
    "again", "further", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must", "not",
    "no", "nor", "this", "that", "these", "those", "there", "here", "where",
    "when", "which", "who", "whom", "what", "how", "why", "all", "any", "some",
    "such", "very", "just", "also", "than", "too", "own", "same", "other",
    "each", "few", "more", "most", "only", "both", "i", "me", "my", "mine", "we",
    "us", "our", "ours", "you", "your", "yours", "he", "him", "his", "she",
    "her", "hers", "it", "its", "they", "them", "their", "theirs",
    # conversational recall framing
    "time", "last", "earlier", "before", "previously", "ago", "recently",
    "yesterday", "today", "week", "month", "year", "worked", "working", "work",
    "discussed", "discuss", "talked", "talk", "mentioned", "mention", "said",
    "asked", "told", "remember", "recall", "thing", "things", "stuff",
    "something", "anything", "everything", "please", "want", "need", "like",
    "find", "look", "looking", "show", "get", "got", "make", "made", "use",
    "used", "using",
})

_QUOTED_RE = re.compile(r'"([^"]+)"')
_OPERATOR_TOKENS = frozenset({"or", "and", "not"})


def trigram_query_tokens(query_text: str, *, min_len: int = 3) -> List[str]:
    """Tokens worth averaging word_similarity over.

    Splits on non-word characters, keeps tokens of ``min_len`` or more
    characters, drops stop words and the query operators (``OR``). If that
    leaves nothing, falls back to the raw length filter alone so a query made
    entirely of short or stop-listed words still searches something.
    """
    raw = [t for t in re.split(r"\W+", query_text or "") if t and len(t) >= min_len]
    kept = [t for t in raw if t.lower() not in STOP_WORDS and t.lower() not in _OPERATOR_TOKENS]
    return kept or raw


def lexical_or_fallback_query(query_text: str) -> str:
    """The OR form of a query for ``websearch_to_tsquery``.

    Quoted phrases stay quoted and intact (a phrase is one unit), stop words
    are dropped, ``-word`` exclusions are dropped too (in websearch syntax a
    negation binds to its neighbour, which an OR chain would misplace), and
    the remaining units are joined with ``OR``. Returns "" when fewer than two
    units remain, or when the query already uses OR, in which case a second
    pass would add nothing over the first.
    """
    text = (query_text or "").strip()
    if not text:
        return ""
    if re.search(r"\bOR\b", text):
        return ""
    units: List[str] = []
    for m in _QUOTED_RE.finditer(text):
        phrase = m.group(1).strip()
        if phrase:
            units.append(f'"{phrase}"')
    rest = _QUOTED_RE.sub(" ", text)
    for tok in rest.split():
        tok = tok.strip()
        if not tok or (tok.startswith("-") and len(tok) > 1):
            continue
        word = re.sub(r"^\W+|\W+$", "", tok)
        if not word or word.lower() in STOP_WORDS or word.lower() in _OPERATOR_TOKENS:
            continue
        units.append(word)
    seen: set[str] = set()
    ordered: List[str] = []
    for u in units:
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(u)
    if len(ordered) < 2:
        return ""
    return " OR ".join(ordered)


__all__ = ["STOP_WORDS", "lexical_or_fallback_query", "trigram_query_tokens"]
