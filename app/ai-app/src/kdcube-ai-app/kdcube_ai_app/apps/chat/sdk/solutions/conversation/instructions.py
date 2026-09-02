# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Model-facing instruction text for the conversation memory realm.

Two texts live here on purpose, side by side, because they are two halves of
one mechanism:

- ``turn_summary_writing_guide`` teaches the model how to WRITE the turn
  summary that later becomes a searchable row (``conv.working.summary``).
- ``CONVERSATION_QUERY_GUIDE`` teaches the model how to QUERY the store that
  those rows land in.

Both are host-neutral. The native ReAct agent writes the summary through
``<channel:summary>`` and searches with ``react.memsearch``; a hosted foreign
agent writes it through ``record_turn_summary`` and searches through the
``conv`` named-service namespace; an external MCP client only searches. Every
one of those surfaces renders the same core text and adds one line that names
its own tool, so the writing side and the reading side cannot drift apart per
host. ``SUMMARY_GUIDE_SENTINEL`` and ``QUERY_GUIDE_SENTINEL`` are the
sentences a regression test looks for on every carrier.

This module imports nothing from the SDK so any surface can import it.
"""

from __future__ import annotations


CONVERSATION_NAMED_SERVICE_NAMESPACE = "conv"

# Realm-trait intro, coherent with the mem (`MEMORY_NAMESPACE_INTRO`) and canvas
# (`CANVAS_NAMESPACE_INTRO`) intros: conversations are one of the user's memory
# realms — what was actually said. Positive framing: it says what the realm IS
# and what it searches (user text, assistant text, the user's uploaded
# attachment summaries).
CONVERSATION_NAMESPACE_INTRO = "Conversations — what was actually said in chat, this conversation or across earlier ones. Search them to recover what the user said (their prompts and follow-ups), what the assistant said (replies and working summaries), and the user's uploaded attachments (their indexed summaries); results come back as turn-level handles you can read or pull. Reach for it whenever a look back would help: an explicit recall request, or when the user refers to something from before, says it was clearer earlier, can't re-locate something, or resumes a dropped thread. Default scope is the current conversation; widen to the user's other conversations with you for cross-conversation recall. It's one of the user's memory realms — the said-aloud kind — alongside their durable memories (`mem`) and context boards (`cnv`)."


# --- Reading side -----------------------------------------------------------

# The sentence every query-side carrier must contain (regression sentinel).
QUERY_GUIDE_SENTINEL = "each such word is a lexical AND-term that fails and dilutes the other arms"

# Host-neutral: names no tool. The carrier (react.memsearch spec, the `conv`
# named-service description, the hosted-agent recovery guide) adds its own
# tool name, scope semantics, and result paths around this text.
CONVERSATION_QUERY_GUIDE = (
    "Stored per turn: what the USER said (prompts, follow-ups, attachment labels), the user's ATTACHMENT summaries (filename, topics, lookup keys), what the ASSISTANT said (replies), the assistant's turn SUMMARIES with their retrieval anchors, internal NOTES. "
    "Ranking, so you can shape the query: three arms run on the one query string and are fused. Semantic: the query embedded against each stored text as a whole. Lexical: every unquoted word must occur in the text (stemmed); a \"quoted phrase\" must occur verbatim; OR separates alternatives; -word excludes; retrieval anchors carry the highest weight. Fuzzy: character similarity averaged over the query tokens (typos yes, synonyms no). Then a recency lift: at equal match, a turn from today outranks a month-old one about 2x. "
    "So the query is a compact bag of content words as they would appear in the stored text, one topic per call: names, file names, identifiers, domain nouns, the user's own wording, exact strings in double quotes, synonyms joined with OR. No conversational framing (\"last time I worked with\", \"the thing we discussed\"): " + QUERY_GUIDE_SENTINEL + ". No time words in the query: time goes to from/to; for old material with no time clue, widen days. "
    "Pick targets by who is being recalled and use that side's words: \"I said/uploaded\" -> user/attachment (their wording, the filename); \"you said/made\" -> assistant/summary (the assistant's wording, artifact names, anchors). Catalog lookups take no query: ordinal (the n-th turn), from/to alone (a period), neither with targets summary (an ordered overview). "
    "On a miss: fewer, surer words; quote the exact phrase; switch targets; widen the window; or browse the overview and read by ordinal. Hits are turn-level: read the turn's summary path first, then the exact refs."
)


# --- Writing side -----------------------------------------------------------

# The sentence every summary-writing carrier must contain (regression sentinel).
SUMMARY_GUIDE_SENTINEL = "Make it distinguishable from the earlier turns you can see"

_DEFAULT_SEARCH_REF = "the conversation search's `summary` target"

_TURN_SUMMARY_WRITING_GUIDE = (
    "This text is what stays of the turn once it leaves the visible window, and what a later compaction reads. It is embedded whole for semantic search and its words are indexed for lexical search, so {search_ref} matches THIS text and nothing else about the turn. Write it for a future reader that holds only a query.\n"
    "- Name things by their searchable names, not by reference: the user's words for the task, file names, artifact titles, identifiers, dates, numbers, domain nouns. Not \"the file\", \"the request\", \"as discussed\".\n"
    "- " + SUMMARY_GUIDE_SENTINEL + ": say what is NEW in this turn (which sub-topic, artifact, state change, error), so a query meant for this turn does not also match the turns before it. Ten turns of \"updated the forecast spreadsheet\" are ten unfindable turns.\n"
    "- Carry the outcome state and the terms a future query would use, including the user's wording when it differs from yours. Refs are logical paths (prompt, decisive tool results, produced artifacts, completion), each with a short human name.\n"
    "- Retrieval anchors are indexed as top-weight tokens for the lexical arm. phrases = verbatim strings someone might re-quote (exact filenames, error messages, titles, the user's exact wording; never paraphrases; searched in double quotes they match exactly this text). entities = proper nouns that identify this turn among hundreds (product/tool/project/person/bundle ids; never generic nouns like \"file\"/\"data\"/\"report\"). Both optional; omit for trivial turns.\n"
    "Example, a turn that built a Q2 forecast spreadsheet and hit an openpyxl error while renaming a column: phrases [\"Forecast-Q2-2026.xlsx\", \"openpyxl IndexError\", \"rename ARR contribution column\"]; entities [\"Forecast-Q2-2026.xlsx\", \"openpyxl\", \"ARR contribution\"]."
)


def turn_summary_writing_guide(search_ref: str | None = None) -> str:
    """The host-neutral rules for writing a searchable turn summary.

    ``search_ref`` names the search surface of the host in its own words
    (for example ``react.memsearch(targets=["summary"])`` for the native agent,
    or "the `conv` search with targets summary" for a hosted agent). When the
    host binds no search tool, leave it unset and the text speaks of the
    conversation search in class vocabulary.
    """
    ref = str(search_ref or "").strip() or _DEFAULT_SEARCH_REF
    return _TURN_SUMMARY_WRITING_GUIDE.format(search_ref=ref)


__all__ = [
    "CONVERSATION_NAMED_SERVICE_NAMESPACE",
    "CONVERSATION_NAMESPACE_INTRO",
    "CONVERSATION_QUERY_GUIDE",
    "QUERY_GUIDE_SENTINEL",
    "SUMMARY_GUIDE_SENTINEL",
    "turn_summary_writing_guide",
]
