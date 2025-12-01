# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/web/content_filters.py

# Core filtering policy - understanding-based

# ----------------------------- Light-weight span slicer (post-processing) -----------------------------
import re as _re
_WORD_RE = _re.compile(r"\w+", _re.U)

def _find_phrase_ci(text: str, phrase: str, start_from: int = 0) -> int:
    """Case-insensitive exact substring first; fallback to word-separated regex match."""
    if not text or not phrase:
        return -1
    low = text.lower()
    ph = phrase.lower()

    # Try exact substring match first
    if start_from:
        idx = low.find(ph, start_from)
        if idx != -1:
            return idx
    else:
        idx = low.find(ph)
        if idx != -1:
            return idx

    # Fallback: allow non-word gaps between words (handles formatting differences)
    words = _WORD_RE.findall(ph)
    if not words:
        return -1
    pat = r"\b" + r"\W+".join(map(_re.escape, words)) + r"\b"
    m = _re.compile(pat, _re.I | _re.U).search(text, pos=start_from)
    return m.start() if m else -1


def trim_with_spans(
        content: str,
        spans: list[dict],
        *,
        ctx_before: int = 600,      # increased for wider capture
        ctx_after: int = 600,       # increased for wider capture
        min_gap: int = 400,         # increased to merge nearby spans
        max_joined: int = 30000,    # increased to allow more content
) -> str:
    """
    Cut 1–2 slices around (s,e) anchors with context; merge close slices; join with ellipses.

    Large context windows to capture complete cognitive blocks with surrounding context:
    - ctx_before/after: 600 - wide context around boundaries
    - min_gap: 400 - aggressively merge nearby spans
    - max_joined: 30000 - allow substantial content
    """
    if not content or not spans:
        return ""

    intervals: list[tuple[int, int]] = []

    for sp in spans[:2]:
        s = (sp.get("s") or "").strip()
        e = (sp.get("e") or "").strip()
        if not s or not e:
            continue

        s_idx = _find_phrase_ci(content, s)
        if s_idx == -1:
            continue

        e_search_from = s_idx + max(1, len(s))
        e_idx = _find_phrase_ci(content, e, start_from=e_search_from)

        if e_idx == -1:
            # Fallback: take substantial content after start anchor
            e_idx = min(len(content) - 1, e_search_from + 2500)

        # Expand to include context
        a = max(0, s_idx - max(0, ctx_before))
        b = min(len(content), e_idx + len(e) + max(0, ctx_after))

        if b > a:
            intervals.append((a, b))

    if not intervals:
        return ""

    # Sort and merge close intervals
    intervals.sort()
    merged = [intervals[0]]

    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + max(0, min_gap):
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    # Extract and join
    parts = [content[s:e] for s, e in merged]
    out = "\n\n…\n\n".join(parts)

    if len(out) > max_joined:
        out = out[:max_joined] + "\n\n…"

    return out


def apply_spans_to_rows(
        rows: list[dict],
        spans_map: dict[int, list[dict]],
        *,
        ctx_before: int = 600,
        ctx_after: int = 600,
        min_gap: int = 400,
) -> list[dict]:
    """Apply span-based trimming to source rows."""
    if not rows or not spans_map:
        return []

    survivors: list[dict] = []

    for r in rows:
        sid = int(r.get("sid", -1))
        spans = spans_map.get(sid) or spans_map.get(str(sid))
        if not spans:
            continue

        original = r.get("content", "") or ""
        pruned = trim_with_spans(
            original,
            spans,
            ctx_before=ctx_before,
            ctx_after=ctx_after,
            min_gap=min_gap
        )

        if not pruned:
            continue

        nr = dict(r)
        nr["content_original_length"] = len(original)
        nr["content"] = pruned
        nr["content_length"] = len(pruned)
        nr["seg_spans"] = spans
        survivors.append(nr)

    return survivors

def FILTER_AND_SEGMENT_GUIDE_with_high_precision(now_iso: str) -> str:

    SPANS_EMPHASIS = """
BOUNDARY & RECALL RULES (NO CUTTING IN THE MIDDLE)

- The text between 's' and 'e' MUST NOT cut through the middle of a logical block: DO NOT cut through
  • a paragraph; 
  • a list item
  • a table row
  • a code/config block
  • a Q&A pair / FAQ item

- Treat each block as atomic:
  • If any part of a block is relevant, the entire block MUST be inside the span.
  • Never place 's' or 'e' in the middle of a block. They should conceptually sit
    on boundaries between blocks (even though you anchor them on a short phrase).

HIGH-RECALL BIAS

- When deciding where to stop a span:
  • If you are unsure whether the boundary block is still relevant, TREAT IT AS RELEVANT.
  • Extend 'e' forward until you are clearly past the last informative block.
  • It is BETTER to include some neutral or slightly redundant text than to
    risk cutting off relevant content.

- When the page has a long continuous relevant section:
  • Prefer ONE large span that covers the whole section.
  • Only introduce a second span if there is a large non-informative chunk
    (e.g., a big promo or unrelated section) splitting two relevant clusters.

- Think in terms of "sections":
  • Identify the first heading/paragraph where truly useful content starts.
  • Identify the last heading/paragraph where truly useful content ends.
  • Your span(s) should cover that entire region, not just isolated sentences.

ANCHOR PLACEMENT AND BOUNDARIES TOGETHER

- 's' should come from the FIRST clearly relevant block:
  • Choose a short phrase (3–8 words) from the first heading or sentence
    of the main relevant region, but conceptually the span begins at that block.

- 'e' should come from the LAST clearly relevant block:
  • Choose a short phrase (3–8 words) from the last sentence of the last
    relevant block (or immediately after it), so that everything before it
    stays inside the span.

- If the relevant section extends further than you expected:
  • Move 'e' later so that all relevant blocks are inside.
  • Do NOT leave a tail of relevant content outside the span.    
"""

    return f"""
GENERAL PURPOSE FILTER + SEGMENT (OBJECTIVE-AWARE; SEMANTIC DEDUP)

INPUTS
- objective: what we're trying to achieve (may be empty)
- queries: search queries used to find sources
- sources: list of {{sid, content, published_time_iso?, modified_time_iso?}}

GOAL
1) FILTER: keep the minimal set of sources that truly contain useful, substantive information for the objective.
2) SEGMENT: for each kept source, return **1–2 spans** that cover the page's main informative content ("fruit"), excluding site chrome/boilerplate ("envelope").
3) **SEMANTIC DEDUP (content-only):** if two sources cover the same useful content, keep only one unless each adds **substantial, non-overlapping** useful content.

----------------------------------------------------------------
FILTER (objective-aware, no URL heuristics)
- Relevance: keep pages with substantive information (not just navigation, promos, link hubs).
- Substance: prefer coherent sections with paragraphs/lists/tables/code/Q&A/specs/instructions.
- Freshness: use only as a tie-breaker when content is otherwise indistinguishable.
- Safeguard: if at least one page has substance, keep at least one SID.
TODAY: {now_iso}

----------------------------------------------------------------
SEGMENT (objective-aware; wide coverage; max 2 spans)
INTENT
- Capture the **majority of objective-relevant content** in **one large span** when possible.
- Use a **second span** only if a substantial non-informative block splits two large relevant clusters.

INCLUDE (FRUIT)
- Coherent informative regions: headings + paragraphs, lists, tables, code/config blocks, figures with captions, Q&A.

EXCLUDE (ENVELOPE)
- Site-wide nav, hero/marketing banners, promos/tiles, carousels, social/share bars, cookie/legal notices, footers, logo walls, repetitive link grids, pure link hubs.

COVERAGE
- Combined span(s) should contain **≥ ~70% of all relevant content** on the page.
- **Anti-tiny rule:** each span encloses **≥ 1,000 characters or ≥ 3 full blocks (paragraphs/rows/items)** unless the page is truly short.
- Keep whole structures intact (no mid-table/list/code cuts).

SPAN COUNT & ORDER
- Default **1 span**. Use **2** only for separated large clusters.
- Spans must be **non-overlapping** and in **document order**.

{SPANS_EMPHASIS}

----------------------------------------------------------------
ANCHORS (CRITICAL: EXACT VERBATIM COPY REQUIRED)

**FUNDAMENTAL RULE:** 
Anchor text MUST be **EXACT CHARACTER-FOR-CHARACTER COPY** from the source content.
- Copy the text EXACTLY as it appears, including ALL punctuation, symbols, capitalization, and spacing
- DO NOT paraphrase, simplify, or "clean up" the anchor text
- DO NOT remove quotes, apostrophes, hyphens, parentheses, or any other characters
- DO NOT change capitalization
- Think of it as Ctrl+C / Ctrl+V - an exact copy

**WHAT THIS MEANS:**
✓ CORRECT: "Section 2.1: Overview" (if source has "Section 2.1: Overview")
✗ WRONG: "Section 2.1 Overview" (removed the colon)

✓ CORRECT: "What's the difference between" (if source has "What's the difference between")
✗ WRONG: "What is the difference between" (changed apostrophe)

✓ CORRECT: "API Reference (v2.0)" (if source has "API Reference (v2.0)")
✗ WRONG: "API Reference v2.0" (removed parentheses)

✓ CORRECT: "Step #3 - Configure settings" (if source has "Step #3 - Configure settings")
✗ WRONG: "Step 3 Configure settings" (removed symbols)

**Each span uses 's' (start) and 'e' (end), both MUST be:**
  • **VERBATIM EXACT COPY** - character-for-character identical to source text, preserving ALL punctuation and symbols
  • **Compact & distinctive** - aim for 3-8 words (shorter is better if unique)
  • **Unique on the page** - if not unique, extend the phrase until it becomes unique
  • **Ordered** - 's' must appear before 'e' in the document
  • **Non-ENVELOPE** - don't anchor in nav/footer/legal/cookie/social/promotional blocks
  • **From actual content text** - NOT from URLs, image filenames, or alt text

**CHOOSING 's' (START):**
- Locate the first heading or sentence that begins the **first relevant cluster** (skip envelope above)
- Copy EXACTLY 3-8 consecutive words from that point
- Include ALL punctuation/symbols exactly as they appear
- If the phrase is not unique on the page, extend it word-by-word until unique

**CHOOSING 'e' (END):**
- Locate the point immediately after the **last relevant block** to include
- Copy EXACTLY the last 3-8 words of that block (or first 3-8 words of the next sentence/heading)
- Include ALL punctuation/symbols exactly as they appear
- Do **not** end on a heading that introduces still-relevant content; include that content and move 'e' after it
- When unsure, **bias toward inclusion** to avoid amputating useful content

**VERIFICATION CHECKLIST (before outputting):**
For each anchor ('s' and 'e'), verify:
1. Can I find this EXACT string (including all punctuation) in the source with Ctrl+F? → Must be YES
2. Does it include ALL symbols, punctuation, capitalization from the original? → Must be YES
3. Is it unique on this page? → Must be YES (if no, extend the phrase)
4. Is it 3-8 words? → Preferred (can be slightly longer if needed for uniqueness)
5. Does it avoid envelope sections (nav/footer/legal/social)? → Must be YES

**ROBUSTNESS:**
- Small amounts of envelope may remain if needed to keep the body intact
- Short decorative separators between relevant sections should remain **inside** the span

----------------------------------------------------------------
SEMANTIC DEDUP (content-only; performed **after** proposing spans)
GOAL
- Keep the **minimum** number of SIDs whose span-enclosed content **together** covers the useful material.  
- Dedup is based **solely on the text enclosed by the proposed spans** (normalized), not on URLs/titles.

NORMALIZATION (for overlap checks)
- From each SID, concatenate text inside its proposed span(s).
- Lowercase; collapse whitespace; remove obvious boilerplate tokens that slipped in (e.g., repeated nav/cta phrases). Keep numbers and units.

OVERLAP METRICS
- Compute:
  • Overlap ratio O = overlap_chars / min(chars_A, chars_B)  
  • Jaccard on token n-grams (e.g., 5-grams) over span text  
  • Structural overlap (same headings/table rows/FAQ questions detected by text match)
- Define **near-duplicate** if any strong signal holds (suggested thresholds):
  • O ≥ 0.90  OR  5-gram Jaccard ≥ 0.85  OR  structural overlap ≥ 0.90.

UNIQUE CONTRIBUTION
- For each source, compute unique_A = (chars in A not in B) / union_chars; similarly unique_B.
- Treat "unique" as **substantial** if unique_X ≥ 0.15 **or** it introduces **new informative structures** (new headings/sections, new table rows/parameters, new Q&A items, new constraints/notes) not present in the other.

KEEP/DROP RULES
- If A and B are near-duplicates **and neither** has substantial unique contribution:
  → **Keep one**; **drop the other** (do not output spans for the dropped SID).
- If both have substantial unique contribution (each adds useful, non-overlapping content):
  → **Keep both**.
- If only one adds substantial unique contribution:
  → **Keep that one**, drop the other.

TIE-BREAKER WHEN DROPPING ONE (content-only)
- Prefer the candidate whose spans cover **more of the union** of relevant content (greater coverage length + more distinct headings/rows/items).  
- If still tied, prefer the one with clearer organization (fewer envelope tokens inside).  
- If still tied, either is acceptable.

CLUSTERING
- Apply the above pairwise to form duplicate clusters. From each cluster, keep the minimal subset that covers all unique contributions; drop the rest.

POST-HOC COLLISION GUARD
- If two kept SIDs end up with **identical ('s','e') anchors** or **≥ 95% identical span text**, re-apply the KEEP/DROP RULES and remove redundancies.

----------------------------------------------------------------
VALIDATION
- Only **kept** SIDs are in the output.
- Each kept SID has **1–2 spans** meeting coverage and anti-tiny rules; whole structures enclosed.
- **CRITICAL:** 's' and 'e' are **EXACT VERBATIM COPIES** from source (preserving ALL punctuation/symbols), **compact** (3-8 words preferred), **unique**, **ordered**, **non-ENVELOPE**.
- Span(s) capture the **majority** of relevant material on that page.

OUTPUT PROTOCOL
**THINKING (user-visible):** 1-2 short sentences summarizing observations and decisions.

**MAIN OUTPUT:** ONLY valid JSON. No text before/after.
- If sources kept:  a JSON object mapping kept SID → array of 1–2 spans
  {{ "<sid>": [ {{ "s": "...", "e": "..." }}, ... ], ... }}
- If all dropped: {{}}  

No extra keys and never return free text explanations in main output. Empty object {{}} if no results.
"""

"""
IMPROVED FILTER_AND_SEGMENT_GUIDE - HIGH RECALL VERSION

Key changes:
1. Objective is a GUIDE, not a FILTER for content within pages
2. Increased coverage from 70% to 80-95%
3. Bias toward inclusion: "when in doubt, include it"
4. Focus on "substantively useful" not "objective-matching"
5. Clearer definition of what counts as useful content
"""

def FILTER_AND_SEGMENT_GUIDE_high_recall_short(now_iso: str) -> str:
    return f"""
GENERAL PURPOSE FILTER + SEGMENT (STRICT JSON • HIGH-RECALL • EXCLUSIVE 'e' • SEMANTIC DEDUP)

INPUTS
- objective: what we’re trying to achieve (may be empty)
- queries: search queries used to find sources
- sources: list of {{sid, content, published_time_iso?, modified_time_iso?}}

GOAL
1) FILTER: keep only sources that truly contain useful, substantive information for the objective; remove near-duplicates by content.
2) SEGMENT: for each kept source, return **1–2 spans** that cover the page’s main informative content (“fruit”) while excluding site chrome/boilerplate (“envelope”).
3) DEDUP: content-only, performed **after** spans are proposed.

========================================================================
STRICT OUTPUT FORMAT (NO FREE TEXT)

- Return **ONLY** a single JSON object: {{ "<sid>": [{{"s":"...", "e":"..."}}, ...], ... }}
- **No** explanations, **no** prose, **no** code fences, **no** markdown.
- If nothing kept, return **{{}}** (empty object).

The JSON **must** be directly parseable (UTF-8, double quotes, no trailing commas).

========================================================================
FILTER (objective-aware, semantic, no URL heuristics)
- Relevance: keep pages that directly help the objective (or, if objective is empty, match the queries) **and** contain substantive content.
- Substance: coherent informational blocks (paragraphs, lists, tables, code/config, Q&A, specs, instructions).
- Drop: pure navigation/link hubs, promo-only, cookie/legal pages, error pages.
- Freshness: tie-breaker only when content is otherwise indistinguishable.
- Safeguard: if at least one page has substance + relevance, keep at least one.
TODAY: {now_iso}

========================================================================
SEGMENT (high recall; 1–2 spans; exclusive 'e')

INTENT
- Capture **80–95%** of the page’s substantive content in spans.
- Prefer **one large span** covering the main body. Use a **second span** only if a large non-informative block splits two big relevant clusters.

INCLUDE (FRUIT)
- Expository/reference content: headings+paragraphs, lists, tables, code/config, figures with captions, Q&A.

EXCLUDE (ENVELOPE)
- Site-wide nav/breadcrumbs, hero/marketing banners, tiles/carousels, CTAs, social bars, cookie/legal banners, footers, logo walls, repetitive link grids.

NO MID-BLOCK CUTS
- A span must not cut through the middle of:
  • a paragraph • a list item • a table row • a code/config block • a Q&A item.
- Treat each block as **atomic**: if any part is useful, include the whole block.

ANTI-TINY RULE
- Each span should enclose **≥ 1,000 characters or ≥ 3 full blocks** (unless the page itself is short).

SPAN COUNT & ORDER
- Default **1 span**. Use **2** only for clearly separated large clusters.
- Spans must be **non-overlapping** and in **document order**.

========================================================================
ANCHORS (exclusive 'e', verbatim, unique, deterministic)

BASICS
- Each span is defined by 's' (start) and 'e' (end).
- Anchors must be:
  • **VERBATIM** substrings from the page text (not URLs/filenames/alt text)
  • **Compact & distinctive** (~3–12 words; extend as needed)
  • **UNIQUE** on the page (see uniqueness rules)
  • **ORDERED**: 's' must appear before 'e'
  • **NON-ENVELOPE**: do not anchor in nav/footer/legal/cookie/social/promo blocks

EXCLUSIVE 'e' (critical)
- The span covers **from the first character of 's' up to but NOT including the first character of 'e'**.
- Therefore, **'e' must come from the first phrase in the first block that is NOT included** (e.g., the next section heading, a footer/utility heading, “Was this helpful?”, etc.).
- Do **not** take 'e' from inside the last included block—doing so would cut mid-block.

CHOOSING 's'
- Pick a short phrase from the **first useful block** (heading or intro sentence) after top-of-page envelope.
- If the phrase is not unique, extend word-by-word until it becomes unique.

CHOOSING 'e'
- Pick a short phrase from the **first boundary block immediately after the last useful block you intend to include** (e.g., next major heading or clear footer/utility delimiter).
- Because 'e' is **exclusive**, this guarantees the last useful block remains fully enclosed.
- If the phrase is not unique, extend word-by-word until unique.

UNIQUENESS & DISAMBIGUATION (must satisfy, in order)
1) Try to choose a phrase that occurs **exactly once** on the page.
2) If it occurs multiple times, **extend it** with following words (or prepend preceding words) until **unique**.
3) If it still occurs multiple times (e.g., common footer text), then:
   • For 's': use the **first** occurrence in the document (or the first after the previous span’s 'e').
   • For 'e': use the **first occurrence strictly after 's'** that lies in the **first non-included block** after the last included block.
   This yields a deterministic, nearest-boundary selection.

BIAS TOWARD INCLUSION
- If unsure where the useful region ends, place 'e' **later** (exclusive) so that all useful blocks are inside.
- Small amounts of envelope may slip in to preserve continuity—acceptable.

========================================================================
SEMANTIC DEDUP (content-only; after spans are proposed)

SCOPE
- Dedup uses **only the text enclosed by each SID’s proposed span(s)**, after normalization. URLs/titles are ignored.

NORMALIZE span text for overlap:
- Lowercase, collapse whitespace; strip obvious boilerplate tokens that slipped in.
- Keep numbers/units and technical tokens.

OVERLAP SIGNALS (near-duplicate if any strong signal holds)
- Character overlap ratio O ≥ 0.90 (on min length), or
- 5-gram Jaccard ≥ 0.85, or
- Structural overlap ≥ 0.90 (same headings/table rows/FAQ questions by text match).

UNIQUE CONTRIBUTION
- For A vs B, compute unique_A = (chars in A not in B)/union_chars.
- Treat as **substantial** if unique_X ≥ 0.15 **or** A introduces new informative structures (new headings/rows/FAQ items/notes).

KEEP/DROP
- Near-duplicates with **no substantial unique contribution** → **keep one**, drop the other(s).
- If both add substantial unique content → **keep both**.
- If only one adds substantial unique content → **keep that one**.
- Tie-breaker when dropping: keep the candidate whose spans cover **more of the union** of useful content; if still tied, keep the cleaner one (less envelope).

CLUSTERING
- Form duplicate clusters; from each cluster, keep the minimal subset that covers all unique contributions; drop the rest.

POST-HOC COLLISION GUARD
- If two kept SIDs have **identical anchors** or **≥95% identical span text**, re-apply KEEP/DROP and remove redundancies.

========================================================================
VALIDATION (all must pass)
- Output contains only kept SIDs.
- Each kept SID has **1–2 spans**; spans enclose whole structures (no mid-block cuts).
- **'e' is exclusive**; anchors are verbatim, compact, unique, ordered, and non-envelope.
- Spans capture the **majority** of the page’s substantive content (80–95% target).
- **Output is raw JSON only** (no prose, no fences). If nothing kept: **{{}}**.
"""


def FILTER_AND_SEGMENT_GUIDE_with_high_recall(now_iso: str) -> str:

    SPANS_EMPHASIS = """
BOUNDARY & RECALL RULES (NO CUTTING IN THE MIDDLE)

- The text between 's' and 'e' MUST NOT cut through the middle of a logical block: DO NOT cut through
  • a paragraph; 
  • a list item
  • a table row
  • a code/config block
  • a Q&A pair / FAQ item

- Treat each block as atomic:
  • If any part of a block is relevant, the entire block MUST be inside the span.
  • Never place 's' or 'e' in the middle of a block. They should conceptually sit
    on boundaries between blocks (even though you anchor them on a short phrase).

HIGH-RECALL BIAS

- When deciding where to stop a span:
  • If you are unsure whether the boundary block is still useful, TREAT IT AS USEFUL.
  • Extend 'e' forward until you are clearly past the last informative block.
  • It is BETTER to include some neutral or slightly redundant text than to
    risk cutting off useful content.

- When the page has a long continuous useful section:
  • Prefer ONE large span that covers the whole section.
  • Only introduce a second span if there is a large non-informative chunk
    (e.g., a big promo or unrelated section) splitting two useful clusters.

- Think in terms of "sections":
  • Identify the first heading/paragraph where truly useful content starts.
  • Identify the last heading/paragraph where truly useful content ends.
  • Your span(s) should cover that entire region, not just isolated sentences.

ANCHOR PLACEMENT AND BOUNDARIES TOGETHER

- 's' should come from the FIRST clearly useful block:
  • Choose a short phrase (3–8 words) from the first heading or sentence
    of the main useful region, but conceptually the span begins at that block.

- 'e' should come from the LAST clearly useful block:
  • Choose a short phrase (3–8 words) from the last sentence of the last
    useful block (or immediately after it), so that everything before it
    stays inside the span.

- If the useful section extends further than you expected:
  • Move 'e' later so that all useful blocks are inside.
  • Do NOT leave a tail of useful content outside the span.    
"""

    return f"""
GENERAL PURPOSE FILTER + SEGMENT (HIGH RECALL WITHIN PAGES; SEMANTIC DEDUP)

INPUTS
- objective: what we're trying to achieve (may be empty - used as GUIDE, not strict filter)
- queries: search queries used to find sources
- sources: list of {{sid, content, published_time_iso?, modified_time_iso?}}

GOAL
1) FILTER: keep ONLY sources that **directly help achieve the objective** and have substantive content. Drop duplicates. This is a STRICT filter based on objective relevance.
2) SEGMENT: for each kept source, return **1–2 spans** that capture **ALL substantively useful content** on that page ("fruit"), excluding only site chrome/boilerplate ("envelope"). This is GENEROUS - keep all useful info on kept pages.
3) **SEMANTIC DEDUP (content-only):** if two sources cover substantially the same content, keep only one unless each adds unique useful information.

----------------------------------------------------------------
FILTER (STRICT: objective-focused + substance + dedup)

**PRIMARY CRITERION - OBJECTIVE MATCH (STRICT):**
- Keep pages that **directly address** the objective or queries
- Keep pages with information that **directly helps** achieve the objective
- Drop pages that are only tangentially related, even if substantive
- If objective is empty/generic, keep pages that match the queries

**SECONDARY CRITERION - SUBSTANCE:**
- Among objective-matching pages, prefer those with substantive content:
  • Coherent paragraphs, detailed explanations
  • Lists, tables, code examples
  • Q&A, specifications, instructions, tutorials
- Drop navigation-only pages, pure link hubs, promo-only pages, error pages

**TERTIARY CRITERION - FRESHNESS (tie-breaker only):**
- Use published_time_iso/modified_time_iso only when two pages are otherwise equivalent
- Prefer more recent content when quality is equal

**SAFEGUARD:**
- If at least one page has substance AND matches objective, keep at least one SID
- If NO pages match objective well, keep nothing (return empty {{}})

TODAY: {now_iso}

----------------------------------------------------------------
SEGMENT (HIGH RECALL; substantive content focus; max 2 spans)

**CRITICAL: FILTER vs SEGMENT SEPARATION**

The FILTER phase already decided this page is objective-relevant.
Your job in SEGMENT: capture ALL useful content on this page.

**DO NOT re-filter based on objective here!**
- ❌ WRONG: "This paragraph doesn't directly match objective, skip it"
- ✅ CORRECT: "This paragraph has useful information, include it"

**THE TWO-PHASE LOGIC:**
1. **FILTER (strict):** "Does this PAGE help with the objective?" → Keep or drop entire page
2. **SEGMENT (generous):** "What CONTENT on this kept page is useful?" → Keep all substantive content, skip only boilerplate

If a page passed FILTER, it's already objective-relevant. Now extract ALL the useful information from it.

INTENT
- Capture **80-95% of all substantive content** on the page in spans.
- Default to **one large span** covering the main content body.
- Use **second span** only if a large non-informative block (ads, promos, unrelated widgets) splits content into separated clusters.

WHAT IS "SUBSTANTIVE CONTENT" (INCLUDE - FRUIT)
Content with real information value:
- Explanatory paragraphs (concepts, processes, how things work)
- Technical details (parameters, configurations, specifications)
- Examples (code blocks, sample configs, use cases)
- Structured data (tables, lists of features/options/parameters)
- Q&A sections (common questions, troubleshooting)
- Instructions, tutorials, step-by-step guides
- Definitions, glossaries, reference material
- Comparisons, tradeoffs, recommendations
- Prerequisites, dependencies, related concepts
- Caveats, warnings, limitations, notes

**BIAS TOWARD INCLUSION:**
- If unsure whether content is useful → **INCLUDE IT**
- Related topics, background info, context → **INCLUDE**
- Examples even if not directly matching objective → **INCLUDE**
- Prerequisites, setup, configuration → **INCLUDE**
- Tangentially related sections → **INCLUDE** (unless clearly off-topic marketing)

WHAT IS "ENVELOPE" (EXCLUDE - BOILERPLATE)
Non-informative site chrome:
- Site-wide navigation menus, breadcrumbs
- Hero banners, marketing taglines, CTAs ("Sign up now!")
- Carousels, product tiles, promotional blocks
- Social sharing buttons, comment sections
- Cookie notices, legal disclaimers, privacy policy links
- Footers with company info, site links
- Logo walls, partner lists (unless informative)
- Repetitive link grids with no explanatory text

**ENVELOPE IS MINIMAL:**
- Only exclude truly non-informative chrome
- A section with both envelope (nav) and fruit (content) → keep the whole section, let trim_with_spans handle context

COVERAGE REQUIREMENTS
- Combined span(s) should capture **≥ 80% of substantive content** on the page.
- **Anti-tiny rule:** each span encloses **≥ 1,000 characters or ≥ 3 full blocks** unless page is genuinely short.
- Keep whole structures intact (no mid-table/list/code cuts).
- When in doubt about boundaries → **extend the span** (bias toward inclusion).

SPAN COUNT & ORDER
- Default **1 span**. Use **2** only for separated large clusters.
- Spans must be **non-overlapping** and in **document order**.

{SPANS_EMPHASIS}

----------------------------------------------------------------
ANCHORS (CRITICAL: EXACT VERBATIM COPY REQUIRED)

**FUNDAMENTAL RULE:** 
Anchor text MUST be **EXACT CHARACTER-FOR-CHARACTER COPY** from the source content.
- Copy the text EXACTLY as it appears, including ALL punctuation, symbols, capitalization, and spacing
- DO NOT paraphrase, simplify, or "clean up" the anchor text
- DO NOT remove quotes, apostrophes, hyphens, parentheses, or any other characters
- DO NOT change capitalization
- Think of it as Ctrl+C / Ctrl+V - an exact copy

**WHAT THIS MEANS:**
✓ CORRECT: "Section 2.1: Overview" (if source has "Section 2.1: Overview")
✗ WRONG: "Section 2.1 Overview" (removed the colon)

✓ CORRECT: "What's the difference between" (if source has "What's the difference between")
✗ WRONG: "What is the difference between" (changed apostrophe)

✓ CORRECT: "API Reference (v2.0)" (if source has "API Reference (v2.0)")
✗ WRONG: "API Reference v2.0" (removed parentheses)

✓ CORRECT: "Step #3 - Configure settings" (if source has "Step #3 - Configure settings")
✗ WRONG: "Step 3 Configure settings" (removed symbols)

**Each span uses 's' (start) and 'e' (end), both MUST be:**
  • **VERBATIM EXACT COPY** - character-for-character identical to source text, preserving ALL punctuation and symbols
  • **Compact & distinctive** - aim for 3-8 words (shorter is better if unique)
  • **Unique on the page** - if not unique, extend the phrase until it becomes unique
  • **Ordered** - 's' must appear before 'e' in the document
  • **Non-ENVELOPE** - don't anchor in nav/footer/legal/cookie/social/promotional blocks
  • **From actual content text** - NOT from URLs, image filenames, or alt text

**CHOOSING 's' (START):**
- Locate the first heading or sentence that begins the **first useful cluster** (skip envelope above)
- Copy EXACTLY 3-8 consecutive words from that point
- Include ALL punctuation/symbols exactly as they appear
- If the phrase is not unique on the page, extend it word-by-word until unique

**CHOOSING 'e' (END):**
- Locate the point immediately after the **last useful block** to include
- Copy EXACTLY the last 3-8 words of the INCLUDED content
- "e" must be FROM the content you want to keep, not from the next section
- Do NOT use the first words of the next heading/section in "e"
- Include ALL punctuation/symbols exactly as they appear
- Do **not** end on a heading that introduces still-useful content; include that content and move 'e' after it
- When unsure, **bias toward inclusion** to avoid amputating useful content

**VERIFICATION CHECKLIST (before outputting):**
For each anchor ('s' and 'e'), verify:
1. Can I find this EXACT string (including all punctuation) in the source with Ctrl+F? → Must be YES
2. Does it include ALL symbols, punctuation, capitalization from the original? → Must be YES
3. Is it unique on this page? → Must be YES (if no, extend the phrase)
4. Is it 3-8 words? → Preferred (can be slightly longer if needed for uniqueness)
5. Does it avoid envelope sections (nav/footer/legal/social)? → Must be YES

**ROBUSTNESS:**
- Small amounts of envelope may remain if needed to keep the body intact
- Short decorative separators between useful sections should remain **inside** the span

----------------------------------------------------------------
SEMANTIC DEDUP (content-only; performed **after** proposing spans)
GOAL
- Keep the **minimum** number of SIDs whose span-enclosed content **together** covers the useful material.  
- Dedup is based **solely on the text enclosed by the proposed spans** (normalized), not on URLs/titles.

NORMALIZATION (for overlap checks)
- From each SID, concatenate text inside its proposed span(s).
- Lowercase; collapse whitespace; remove obvious boilerplate tokens that slipped in (e.g., repeated nav/cta phrases). Keep numbers and units.

OVERLAP METRICS
- Compute:
  • Overlap ratio O = overlap_chars / min(chars_A, chars_B)  
  • Jaccard on token n-grams (e.g., 5-grams) over span text  
  • Structural overlap (same headings/table rows/FAQ questions detected by text match)
- Define **near-duplicate** if any strong signal holds (suggested thresholds):
  • O ≥ 0.90  OR  5-gram Jaccard ≥ 0.85  OR  structural overlap ≥ 0.90.

UNIQUE CONTRIBUTION
- For each source, compute unique_A = (chars in A not in B) / union_chars; similarly unique_B.
- Treat "unique" as **substantial** if unique_X ≥ 0.15 **or** it introduces **new informative structures** (new headings/sections, new table rows/parameters, new Q&A items, new constraints/notes) not present in the other.

KEEP/DROP RULES
- If A and B are near-duplicates **and neither** has substantial unique contribution:
  → **Keep one**; **drop the other** (do not output spans for the dropped SID).
- If both have substantial unique contribution (each adds useful, non-overlapping content):
  → **Keep both**.
- If only one adds substantial unique contribution:
  → **Keep that one**, drop the other.

TIE-BREAKER WHEN DROPPING ONE (content-only)
- Prefer the candidate whose spans cover **more of the union** of useful content (greater coverage length + more distinct headings/rows/items).  
- If still tied, prefer the one with clearer organization (fewer envelope tokens inside).  
- If still tied, either is acceptable.

CLUSTERING
- Apply the above pairwise to form duplicate clusters. From each cluster, keep the minimal subset that covers all unique contributions; drop the rest.

POST-HOC COLLISION GUARD
- If two kept SIDs end up with **identical ('s','e') anchors** or **≥ 95% identical span text**, re-apply the KEEP/DROP RULES and remove redundancies.

----------------------------------------------------------------
VALIDATION
- Only **kept** SIDs are in the output.
- Each kept SID has **1–2 spans** meeting coverage and anti-tiny rules; whole structures enclosed.
- **CRITICAL:** 's' and 'e' are **EXACT VERBATIM COPIES** from source (preserving ALL punctuation/symbols), **compact** (3-8 words preferred), **unique**, **ordered**, **non-ENVELOPE**.
- Span(s) capture **≥ 80% of substantive content** on that page.

OUTPUT PROTOCOL
**THINKING (user-visible):** 1-2 short sentences summarizing observations and decisions.

**MAIN OUTPUT:** ONLY valid JSON. No text before/after.
- If sources kept:  a JSON object mapping kept SID → array of 1–2 spans
  {{ "<sid>": [ {{ "s": "...", "e": "..." }}, ... ], ... }}
- If all dropped: {{}}  

No extra keys and never return free text explanations in main output. Empty object {{}} if no results.
"""

def FILTER_AND_SEGMENT_GUIDE_with_high_recall(now_iso: str) -> str:

    SPANS_EMPHASIS = """
BOUNDARY & RECALL RULES (NO CUTTING IN THE MIDDLE)

- The text between 's' and 'e' MUST NOT cut through the middle of a logical block: DO NOT cut through
  • a paragraph;
  • a list item;
  • a table row;
  • a code/config block;
  • a Q&A pair / FAQ item.

- Treat each block as atomic:
  • If any part of a block is useful, the entire block MUST be inside the span.
  • Never place 's' or 'e' inside a block. They should conceptually sit on boundaries between blocks (even though you anchor them on a short phrase).

HIGH-RECALL BIAS

- When deciding where to stop a span:
  • If you are unsure whether the boundary block is still useful, TREAT IT AS USEFUL.
  • Extend the end boundary forward until you are clearly past the last informative block.
  • It is BETTER to include some neutral or slightly redundant text than to risk cutting off useful content.

- When the page has a long continuous useful section:
  • Prefer ONE large span that covers the whole section.
  • Only introduce a second span if there is a large non-informative chunk (e.g., promo/partner logo wall/unrelated widget) splitting two useful clusters.

- Think in terms of "sections":
  • Identify the first heading/paragraph where truly useful content starts.
  • Identify the last heading/paragraph where truly useful content ends.
  • Your span(s) should cover that entire region, not just isolated sentences.

ANCHOR PLACEMENT AND BOUNDARIES TOGETHER

- 's' should come from the FIRST clearly useful block:
  • Choose a short phrase (3–8 words) from the first heading or first sentence of the main useful region, but conceptually the span begins at that block.

- 'e' should come from the LAST clearly useful block:
  • Choose a short phrase (3–8 words) from the last sentence of the last useful block (or immediately after it), so that everything before it stays inside the span.

- If the useful section extends further than you expected:
  • Move 'e' later so that all useful blocks are inside.
  • Do NOT leave useful tail content outside the span.
"""

    return f"""
GENERAL PURPOSE FILTER + SEGMENT (HIGH RECALL WITHIN PAGES; SEMANTIC DEDUP)

INPUTS
- objective: what we're trying to achieve (may be empty – used as GUIDE for FILTER only)
- queries: search queries used to find sources
- sources: list of {{sid, content, published_time_iso?, modified_time_iso?}}

GOAL
1) FILTER: keep ONLY sources that directly help achieve the objective and have substantive content. Drop near-duplicates by content.
2) SEGMENT (objective-agnostic): for each kept source, return 1–2 spans that capture ALL substantively useful content on that page ("fruit"), excluding only site chrome/boilerplate ("envelope"). Be generous.
3) SEMANTIC DEDUP (content-only): if two sources cover substantially the same content, keep only one unless each adds unique useful information.

----------------------------------------------------------------
FILTER (STRICT: objective-focused + substance + semantic dedup)

PRIMARY — OBJECTIVE MATCH (STRICT):
- Keep pages that directly address the objective or queries.
- Keep pages with information that directly helps achieve the objective.
- Drop pages that are only tangentially related, even if substantive.
- If objective is empty/generic, keep pages that match the queries.

SECONDARY — SUBSTANCE:
- Among objective-matching pages, prefer those with substantive content:
  • Coherent paragraphs, detailed explanations
  • Lists/tables/specs
  • Code examples/configs
  • Q&A/FAQ
  • Instructions/tutorials.
- Drop navigation-only pages, pure link hubs, promo-only pages, error pages.

TERTIARY — FRESHNESS (tie-breaker only):
- Use published_time_iso/modified_time_iso when two pages are otherwise equivalent.
- Prefer the more recent when quality is equal.

SAFEGUARD:
- If at least one page has substance AND matches the objective, keep at least one SID.
- If no pages match the objective well, keep nothing (return empty {{}}).

TODAY: {now_iso}

----------------------------------------------------------------
SEGMENT (objective-agnostic; high recall; 1–2 spans; exclusive 'e')

CRITICAL: FILTER vs SEGMENT SEPARATION
- FILTER (strict): "Does this PAGE help with the objective?" → Keep or drop entire page.
- SEGMENT (generous & objective-agnostic): "What CONTENT on this kept page is useful?" → Keep all substantive content; skip only boilerplate.

INTENT
- Capture 80–95% of all substantive content on the page in spans.
- Prefer ONE large span covering the main content body.
- Use a second span only if a large non-informative block splits content into separated clusters.

WHAT TO INCLUDE (FRUIT)
- Explanatory paragraphs; technical details/parameters/specs; examples (code/config); structured data (tables/lists); Q&A/FAQ; instructions/tutorials; definitions/glossaries; reference material; comparisons/tradeoffs; prerequisites/dependencies; caveats/limitations/notes.
- Bias toward inclusion: if unsure whether content is useful, INCLUDE IT.

WHAT TO EXCLUDE (ENVELOPE)
- Site-wide navigation, breadcrumbs, hero/marketing taglines, CTAs, carousels/product tiles/promos, social/share bars, cookie/legal banners, footers/company link grids, partner/logo walls, repetitive link hubs without explanations.

COVERAGE REQUIREMENTS
- Combined span(s) should capture ≥ 80% of substantive content on the page.
- Anti-tiny rule: each span encloses ≥ 1,000 characters or ≥ 3 full blocks (paragraphs/list-items/table-rows/code/FAQ) unless the page is genuinely short.
- Keep whole structures intact (no mid-table/list/code cuts).
- When in doubt about boundaries → extend the span (favor inclusion).

SPAN COUNT & ORDER
- Default 1 span. Use 2 only for clearly separated clusters.
- Spans must be non-overlapping and in document order.

{SPANS_EMPHASIS}

----------------------------------------------------------------
ANCHORS (EXACT VERBATIM; EXCLUSIVE END; MULTI-OCCURRENCE POLICY)

FUNDAMENTAL RULE
- Anchor text MUST be EXACT CHARACTER-FOR-CHARACTER COPY from the source content:
  • Preserve punctuation, symbols, capitalization, and spacing.
  • Do NOT paraphrase, normalize, or drop characters.
  • Think Ctrl+C / Ctrl+V.

CONSTRAINTS FOR BOTH 's' AND 'e'
- Verbatim exact copy; compact & distinctive (aim 3–8 words; extend if needed for uniqueness).
- Unique on the page (if not unique, extend word-by-word until unique).
- Ordered: 's' must appear before 'e'.
- Non-ENVELOPE: do not select from global nav/footer/legal/cookie/social/promotional blocks.
- From real text content (not URLs, not image filenames/alt text).

CHOOSING 's' (START)
- Locate the first heading/sentence that begins the first useful cluster (skip envelope above).
- Copy exactly 3–8 consecutive words from there; extend to ensure uniqueness.

CHOOSING 'e' (END) — **EXCLUSIVE**
- The span is **[s_pos, e_pos)** — it ENDS BEFORE the first character of 'e'.
- Choose 'e' from the **last useful block** (typically its last sentence). 'e' text itself is **not included** in the span.
- Do NOT use the first words of the next heading/section as 'e'.
- If the last useful block’s sentence contains phrases that appear elsewhere:
  • Start with a 3–8 word phrase near the end of that sentence.
  • If multiple occurrences exist, **uniquify-in-place**: extend the phrase by adding adjacent words from the same sentence until it becomes unique for that position.
  • If still ambiguous, pick a different short phrase from the same last useful sentence and repeat.
- Never place 'e' on a heading that introduces still-useful content; include that content and move 'e' after it.

VERIFICATION CHECKLIST (for each anchor)
1) Exact match (including punctuation/case) exists in the page.  
2) Anchor keeps whole blocks intact (no mid-block cuts).  
3) Unique on the page (extend if needed).  
4) 3–8 words preferred (can be longer only to secure uniqueness).  
5) Not taken from envelope sections.

ROBUSTNESS
- Small amounts of envelope may remain if necessary to keep the body intact.
- Decorative separators between useful sections should remain inside the span.

----------------------------------------------------------------
SEMANTIC DEDUP (content-only; after proposing spans)

GOAL
- Keep the minimum number of SIDs whose span-enclosed content together covers the useful material.
- Dedup is based solely on the text enclosed by the proposed spans (normalized). Ignore URLs/titles.

NORMALIZATION (for overlap checks)
- For each SID, concatenate text inside its span(s).
- Lowercase; collapse whitespace; strip obvious boilerplate tokens (generic nav/cta stubs). Keep numbers/units and meaningful tokens.

OVERLAP METRICS
- Compute:
  • Character overlap ratio O = overlap_chars / min(chars_A, chars_B)
  • 5-gram Jaccard over tokens
  • Structural overlap: same headings/table rows/FAQ questions by text match
- Define near-duplicate if ANY holds:
  • O ≥ 0.90  OR  5-gram Jaccard ≥ 0.85  OR  structural overlap ≥ 0.90.

UNIQUE CONTRIBUTION
- For each source, compute unique_X = (chars unique to X) / union_chars.
- Treat as **substantial** if unique_X ≥ 0.15 OR it introduces **new informative structures** (new headings/sections, table rows/parameters, Q&A items, constraints/notes) not present in the other.

KEEP/DROP RULES
- If A and B are near-duplicates and neither has substantial unique contribution:
  → Keep ONE; drop the other (no spans for dropped SID).
- If both have substantial unique contribution:
  → Keep BOTH.
- If only one adds substantial unique contribution:
  → Keep THAT one; drop the other.

TIE-BREAKERS WHEN KEEPING ONE (content-only)
- Prefer the candidate whose spans cover more of the union of useful content (longer coverage + more distinct structures).
- If still tied within ~2%, prefer the one with fewer envelope tokens inside.
- If still tied, prefer the smaller SID as a deterministic fallback.

CLUSTERING
- Apply pairwise to form duplicate clusters; keep the minimal subset that covers all unique contributions; drop the rest.

POST-HOC COLLISION GUARD
- If two kept SIDs end up with identical ('s','e') anchors OR ≥ 95% identical span text, re-apply KEEP/DROP and remove redundancies (use tie-breakers above).

----------------------------------------------------------------
VALIDATION
- Only kept SIDs appear in output.
- Each kept SID has 1–2 spans meeting coverage and anti-tiny rules; whole structures enclosed.
- 's' and 'e' are exact verbatim, unique, ordered, non-ENVELOPE.
- 'e' is **exclusive** (span ends before 'e').
- Combined spans capture ≥ 80% of substantive content on that page.

----------------------------------------------------------------
OUTPUT PROTOCOL
- Output **ONLY** a single JSON object. No prose, no code fences, no keys other than SIDs.
- If sources kept:  {{ "<sid>": [ {{ "s": "...", "e": "..." }}, ... ], ... }}
- If all dropped:  {{}}
- Do not add explanations, markdown fences, or any extra text before/after the JSON.
"""

FILTER_AND_SEGMENT_GUIDE = lambda iso: FILTER_AND_SEGMENT_GUIDE_with_high_recall(iso)