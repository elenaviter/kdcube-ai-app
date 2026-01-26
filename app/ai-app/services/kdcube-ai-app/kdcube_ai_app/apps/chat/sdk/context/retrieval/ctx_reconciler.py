# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/retrieval/ctx_reranker.py

from __future__ import annotations
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from kdcube_ai_app.apps.chat.sdk.tools.backends.summary.user_input_summary_generator import \
    user_input_summary_instruction
from kdcube_ai_app.apps.chat.sdk.util import _now_str, _today_str, _now_up_to_minutes
from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase, create_cached_system_message
from kdcube_ai_app.apps.chat.sdk.streaming.streaming import  \
    _stream_agent_two_sections_to_json as _stream_agent_sections_to_json, _get_2section_protocol, \
    _stream_simple_structured_json
import kdcube_ai_app.apps.chat.sdk.viz.logging_helpers as logging_helpers

from kdcube_ai_app.apps.chat.sdk.skills.instructions.shared_instructions import URGENCY_SIGNALS, CLARIFICATION_QUALITY, TECH_EVOLUTION_CAVEAT, ELABORATION_NO_CLARIFY, USER_GENDER_ASSUMPTIONS

# Helper function for prefix matching
def _resolve_id_by_prefix(partial_id: str, valid_pool: set) -> Optional[str]:
    """
    Resolve a potentially partial ID to its full form from the pool.

    Rules:
    - If exact match exists → return it
    - If exactly one prefix match → return the full ID
    - If zero or multiple matches → return None (ambiguous/invalid)

    Examples:
        pool = {"turn_1762184903332_iablmc", "turn_1762184903332_xyzkl"}
        _resolve_id_by_prefix("turn_1762184903332_iablmc", pool) → "turn_1762184903332_iablmc" (exact)
        _resolve_id_by_prefix("turn_1762184903332", pool) → None (ambiguous: 2 matches)

        pool = {"turn_1762184903332_iablmc", "turn_9999999999999_other"}
        _resolve_id_by_prefix("turn_1762184903332", pool) → "turn_1762184903332_iablmc" (unique prefix)
    """
    if not partial_id or not isinstance(partial_id, str):
        return None

    # Fast path: exact match
    if partial_id in valid_pool:
        return partial_id

    # Prefix matching
    matches = [full_id for full_id in valid_pool if full_id.startswith(partial_id)]

    # Return only if exactly one match (unambiguous)
    if len(matches) == 1:
        return matches[0]

    return None  # Zero or multiple matches

def _get_2section_protocol_ctx(json_shape_hint: str) -> str:
    """
    Strict 2-part protocol for the context reranker:

      1) THINKING CHANNEL  (user-facing, very short, non-technical)
      2) STRUCTURED JSON CHANNEL (CtxRerankOut JSON, machine-readable)
    """
    return (
        "\n\n[CRITICAL OUTPUT PROTOCOL — TWO SECTIONS, IN THIS ORDER]:\n"
        "• You MUST produce EXACTLY TWO SECTIONS (two channels) in this order.\n"
        "• Use EACH START marker below EXACTLY ONCE.\n"
        "• NEVER write any END markers like <<< END ... >>>.\n"
        "• The SECOND section must be a fenced JSON block and contain ONLY JSON.\n\n"

        "CHANNEL 1 — THINKING CHANNEL (user-facing status):\n"
        "Marker:\n"
        "<<< BEGIN INTERNAL THINKING >>>\n"
        "Immediately after this marker, write a VERY SHORT, non-technical status for the user.\n"
        "- 1–2 short sentences or up to 3 brief bullets.\n"
        "- Plain language only (no JSON, no field names, no schemas, no `turn_ids` etc.).\n"
        "- Explain what you’re focusing on and how you’ll use prior context, e.g.:\n"
        "  • which earlier parts of the conversation you’re going to reuse,\n"
        "  • whether you will rely on past notes/memories,\n"
        "  • whether you might need to ask a follow-up question.\n"
        "- Do NOT mention CtxRerankOut, keys like `turn_ids`, or any API details.\n"
        "- Do NOT emit any other BEGIN/END markers inside this channel.\n\n"

        "Examples of GOOD thinking channel snippets:\n"
        "- \"I’ll reuse our recent discussion about your project and a couple of earlier notes.\"\n"
        "- \"I’m focusing on your last question and the earlier requirements you shared.\"\n"
        "- \"The context looks sufficient, so I’ll pick the most relevant past messages for you.\"\n"
        "If you truly have nothing to add, output a single line with \"…\".\n\n"

        "CHANNEL 2 — STRUCTURED JSON CHANNEL (CtxRerankOut decision):\n"
        "Marker:\n"
        "<<< BEGIN STRUCTURED JSON >>>\n"
        "Immediately after this marker, output ONLY a ```json fenced block with a single\n"
        "CtxRerankOut object that matches the JSON shape hint below:\n"
        "```json\n"
        f"{json_shape_hint}\n"
        "```\n\n"

        "[STRICT RULES FOR CHANNEL 2 (JSON)]:\n"
        "1. Channel 2 MUST contain ONLY a single JSON object.\n"
        "2. JSON MUST be inside the ```json fenced block shown above.\n"
        "3. DO NOT write any text, markdown, or comments before ```json.\n"
        "4. DO NOT write anything after the closing ``` (no prose, no markers).\n"
        "5. DO NOT write any other code fences (```python, ```text, etc.).\n"
        "6. The JSON must match the CtxRerankOut schema (types and keys) and be valid JSON.\n"
        "7. Do NOT mention the two-channel protocol, markers, or tools INSIDE the JSON.\n\n"

        "CORRECT (structure only; example is illustrative):\n"
        "<<< BEGIN INTERNAL THINKING >>>\n"
        "I’ll reuse your recent messages and a couple of earlier notes to guide the answer.\n"
        "<<< BEGIN STRUCTURED JSON >>>\n"
        "```json\n"
        "{\n"
        "  \"turn_ids\": [\"turn_abc\"],\n"
        "  \"memory_bucket_ids\": [],\n"
        "  \"local_memories_turn_ids\": [],\n"
        "  \"assertions\": [\n"
        "    {\"key\":\"format|source|topic|delivery|safety\", \"value\":\"string|number|boolean|{min,max}|{options:[...]}\", \"severity\":\"must|prefer|allow\", \"scope\":\"conversation|objective|artifact|thread\", \"applies_to\":\"label\"}\n"
        "  ],\n"
        "  \"exceptions\": [\n"
        "    {\"key\":\"format|source|topic|delivery|safety\", \"value\":\"string|number|boolean|{exclude:[...]}|{rule:...}\", \"severity\":\"avoid|do_not\", \"scope\":\"conversation|objective|artifact|thread\", \"applies_to\":\"label\"}\n"
        "  ],\n"
        "  \"facts\": [\n"
        "    {\"key\":\"domain|constraint|fact\", \"value\":\"string|number|boolean|{...}\", \"severity\":\"high|medium|low\", \"scope\":\"conversation|objective|artifact|thread\", \"applies_to\":\"label\"}\n"
        "  ],\n"
        "  \"user_input_summary\": \"...\",\n"
        "  \"objective\": \"...\",\n"
        "  \"clarification_questions\": [],\n"
        "}\n"
        "```\n\n"

        "WRONG (DO NOT DO THIS):\n"
        "<<< BEGIN INTERNAL THINKING >>>\n"
        "Reasoning… I will now output JSON with turn_ids.\n"  # mentions internals → forbidden\n"
        "<<< BEGIN STRUCTURED JSON >>>\n"
        "```json\n"
        "{ \"turn_ids\": [\"turn_abc\"] }\n"
        "```\n"
        "Here is some extra explanation after the JSON.\n"
    )

def _agents_debug_enabled() -> bool:
    value = os.getenv("KDCUBE_AGENTS_DEBUG", "")
    return str(value).strip().lower() not in ("", "0", "false", "no")


# -------------------- Output model (unchanged) --------------------

class CtxRerankOut(BaseModel):
    turn_ids: List[str] = Field(default_factory=list, description="Ordered, unique, ≤ limit_ctx (turns to materialize)")
    memory_bucket_ids: List[str] = Field(default_factory=list)
    local_memories_turn_ids: List[str] = Field(default_factory=list)
    assertions: List[Dict[str, Any]] = Field(default_factory=list)
    exceptions: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    user_input_summary: str = ""
    objective: str = ""
    clarification_questions: List[str] = Field(default_factory=list)

# -------------------- Context and Local (turn) Memories Reconciler --------------------

async def ctx_reconciler_stream(
        svc: ModelServiceBase,
        *,
        guess_package_json: str,
        current_context_str: str,
        search_hits_json: str,
        bucket_cards_json: str,
        timezone: str,
        gate_decision: dict = None,
        limit_ctx: int = 10,
        max_buckets: int = 3,
        max_delta_keep: int = 5,
        on_thinking_delta: Optional[Any] = None,
        max_tokens: Optional[int] = 2000
) -> Dict[str, Any]:
    """
    Responsibilities:
      - Select turn_ids: minimal materialized prior turns needed for origin + governing specs.
      - Select memory_bucket_ids: relevant long-lived thematic buckets from candidate cards.
      - Select local_memories_turn_ids: earlier-turn insights useful without loading full turns.
      - Produce objective and user_input_summary for downstream routing/embedding.
      - Emit clarification questions only when context selection is ambiguous.

    Selection pools:
      - turn_ids: ONLY from [CURRENT_CONTEXT] prior turns + [SEARCH_HITS].
      - memory_bucket_ids: ONLY from [CANDIDATE_MEMORY_BUCKET_CARDS] (already cosine-ranked upstream).
    - local_memories_turn_ids: ONLY from [TURN MEMORIES — CHRONOLOGICAL].

    Route-aware behavior:
      - If route ∈ tools_* → Solver will run. Prefer not to ask clarifications. If user refers
        to prior content vaguely (this/that/it/the file), assume the most recent relevant artifact or
        do NOT ask "which X?".
      - Honor CONTEXT_QUERIES by prioritizing turns likely matching those targets.

    Deixis (this/that/similar to/you said/that doc/schema) → prioritize assistant_artifact-like recent turns.
    """

    today = _today_str()
    now = _now_up_to_minutes()
    # Toggle: if True, allow delta_fps to be in the selection pool; else only historic turn logs
    allow_delta_fp_pool = False

    gate_decision = gate_decision or {}
    route = gate_decision.get("route")

    thinking_budget_tokens = min(180, max(80, int(0.1 * max_tokens)))
    sys_1 = (
        "[ROLE]\n"
        "You are a light reranker for conversation context and memory-bucket picker.\n"
        "User prompt is in the current turn log in [user.prompt]. Semantic summaries of the user attachments, if any, is in [user.attachments.<attachment name>].\n"
        "You DO NOT solve the user request. Read your role in this instruction and follow it.\n"
        "\n"
        "[HARD LIMITS]\n"
        "• THINKING soft budget ≤ {thinking_budget_tokens} tokens. If near budget, output a single '…' and proceed to JSON.\n"
        "• THINKING is user-visible and must be plain, non-technical language.\n"
        "• THINKING must be telegraphic, concise, and sharp.\n"
        "  - Do NOT mention JSON, field names, schemas, or internal keys (like `turn_ids`).\n"
        "  - Briefly describe how you are using prior conversation/memories to help the user.\n"
        "• JSON MUST be complete & valid; never truncate JSON; never output text after JSON.\n"
        "• MINIMAL JSON: if a field is empty/default (null, \"\", [], {}, 0), do NOT emit the key.\n"
        "\n"
        "[RESPONSIBILITIES]\n"
        "- Select turn_ids: the minimal set of materialized prior turns needed to complete the task (origin + governing specs).\n"
        "- Select memory_bucket_ids: long-lived thematic memory buckets to activate (few, high-fit).\n"
        "- Select local_memories_turn_ids: earlier turn insights (facts/assertions/preferences) relevant without loading full turns.\n"
        "- Extract CURRENT turn preferences (assertions/exceptions) using visible context.\n"
        "- Extract CURRENT turn facts (facts) using visible context.\n"
        "- Produce objective and user_input_summary for downstream routing/embedding.\n"
        "- Emit clarification questions only when context selection is ambiguous (not task-level gaps).\n"
        "- All extracted fields (assertions/exceptions/facts/objective/summary) must be telegraphic, concise, and sharp.\n"
        "- HARD BAN: Do NOT ask how the work should be done. This agent is recall-only.\n"
        "\n"
        "[INPUTS]\n"
        "• [CURRENT_CONTEXT]: includes current turn log plus recent/earlier turns.\n"
        "  The current turn log already includes the user message and attachments summaries.\n"
        "  Use the [TURNS CANDIDATES TABLE] to see why each turn is included (recent_turn vs context_hit + queries).\n"
        "  If present, [USER FEEDBACK — CHRONOLOGICAL] (near bottom) summarizes feedback for nearby turns.\n"
        "  [TURN MEMORIES — CHRONOLOGICAL] (near bottom) lists memories captured from recent turns (newest→oldest).\n"
        "  These include user preferences and assistant-originated signals that may matter for downstream agents.\n"
        "  Include ALL turn_ids whose memories are relevant to the current task in local_memories_turn_ids so those memories are preserved.\n"
        "  If preferences conflict, prioritize the most recent turn that states the preference (newer overrides older).\n"
        "• [CONTEXT_QUERIES]: the semantic queries that were run against history (from gate ctx_retrieval_queries).\n"
        "• [SEARCH_HITS]: semantic/keyword search results with turn_ids and scores (primary source for turn_ids).\n"
        "  Each hit includes source_query that links it back to a context query.\n"
        "• [CANDIDATE_MEMORY_BUCKET_CARDS]: memory buckets (selectable pool for memory_bucket_ids).\n"
        "• Route: the gate routing decision (tools_* vs general_*), already applied to clarification policy below.\n"
        "\n"
        "[YOUR TASK]\n"
        "Return STRICT JSON with:\n"
        f"- turn_ids (≤ {limit_ctx}) — chosen from [SEARCH_HITS] + prior turns in [CURRENT_CONTEXT] (ordered by relevance, not just recency). These are the turns to materialize.\n"
        f"- memory_bucket_ids (≤ {max_buckets}) — subset of [CANDIDATE_MEMORY_BUCKET_CARDS] bucket_id that are relevant to activate for the current turn.\n"
        "- clarification_questions (≤4) — ONLY if [CLARIFICATION POLICY] allows this and user intent is still under-specified OR key constraints are missing and cannot be inferred from the selected turns/buckets.\n"
        f"- local_memories_turn_ids (≤ {max_delta_keep}) — subset of turn ids listed under [TURN MEMORIES — CHRONOLOGICAL] that are relevant to the current turn.\n"
        "- assertions — CURRENT turn preferences to include (see [PREFS SPEC]).\n"
        "- exceptions — CURRENT turn preferences to avoid/exclude (see [PREFS SPEC]).\n"
        "- facts — CURRENT turn facts to remember (see [FACTS SPEC]).\n"
        "- facts — CURRENT turn facts to remember (see [FACTS SPEC]).\n"
        "- user_input_summary — contextual, embedding-friendly summary of the current user input (see [USER_INPUT_SUMMARY SPEC]).\n"
        "- objective — one short sentence describing the current turn objective.\n"
        "• Apply the **Context Completeness** principle: select turn_ids that jointly cover origin + unique, non-derivable constraints (e.g., output format/schema, scope, audience, routing hints).\n"
        "• If scope is ambiguous, include more turn_ids (max limit) instead of asking questions.\n"
        "• If multiple targets might fit the user reference, include ALL turns that contain those targets (up to limit).\n"
        "• Your goal is to resolve all references in the user message to prior artifacts/objects using the visible context (matched turns + recent turns).\n"
        "  This resolution must inform turn selection and user_input_summary/objective.\n"
        "• Clarifications are ONLY about choosing between different documents/turns; never ask about sections/tabs/parts within a single identified document.\n"
        "• If the user says \"I said/I provided/I shared/in my example/in my data\" it ALWAYS refers to user messages (text or attachments), never assistant replies or assistant-created artifacts. I=User.\n"
        "• Scope signals: \"I said/I mentioned/I provided/I gave/I shared\" always refer to user-provided content (scope=user), even if an assistant-created artifact also exists.\n"
        "• Scope signals: \"I gave/provided/shared\", \"my data\", \"example I provided\" → user-provided origin. \"You said/explained\" → assistant statements. \"You created/made the file/report/spreadsheet\" → assistant artifact. If multiple are plausible, include both origin turns.\n"
        "\n"
        "[HOW TO DECIDE]\n"
        "1) Determine the 'working objective' using the composite lens.\n"
        "   Infer the intent from:\n"
        "     • the current turn log (user message)\n"
        "     • the dominant theme across the turns you select on turn_ids\n"
        "   • Identify missing dimensions of the objective (content origin, format/schema, scope/filters). Ensure at least one relevant turn covers each dimension. Don’t drop the only turn that specifies format/schema.\n"
        "2) Use [TURN MEMORIES — CHRONOLOGICAL] to better understand the current context and improve your choice\n"
        "3) Choose memory buckets from [CANDIDATE_MEMORY_BUCKET_CARDS] (quality > quantity) that best support the working objective:\n"
        "   - Compare the working objective and [CURRENT_CONTEXT] themes against each bucket card's:\n"
        "     name, short_desc, objective_text, topic_centroid, and top_signals (facts/assertions/exceptions).\n"
        "   - Prefer fewer, highly relevant memory buckets.\n"
        "   - Store at memory_bucket_ids\n"
        "   - If nothing pass well, leave memory_bucket_ids blank!\n"
        "\n"
        "4) Choose turns (selection):\n"
        "   \n"
        "   **Step A - Select turn_ids:**\n"
        "   • PRIMARY: [SEARCH_HITS] (semantically matched turn logs)\n"
        "   • SECONDARY: recent turns in [CURRENT_CONTEXT]\n"
        "   • Prefer: high-scoring search hits > recent turns; if [SEARCH_HITS] is empty, select from [CURRENT_CONTEXT] only\n"
        "   • If [SEARCH_HITS] and recent turns overlap (same turn_id), treat as ONE turn (do not double count).\n"
        
        "   • Always include turns that introduce or refine **task-defining specifications** (e.g., output structure/schema, required fields, contracts/interfaces, acceptance criteria, thresholds/units, naming/paths).\n"
        "   • If a clarification or resolution turn supplies previously missing **specs/constraints** you (or the system) asked for, mark it **relevant** even if the artifact content originated elsewhere.\n"
        "   • Do not drop a turn when it is the **sole source of a non-derivable requirement**; when uncertain, keep the minimal set that covers artifact origin **and** governing specifications.\n"        
        
        "   \n"
        f"   Choose up to {limit_ctx} turns for full materialization.\n"
        "   \n"
        "   **Which turns to include - read turn logs for origin signals:**\n"
        "   \n"
        "   Include turns where content ORIGINATED:\n"
        "   ✓ Turn shows solver generated/created the artifact\n"
        "   ✓ Turn shows user provided/pasted data in their message\n"
        "   ✓ Turn shows assistant wrote the content/answer\n"
        "   \n"
        "   Don't include turns that only MENTION:\n"
        "   ✗ Turn retrieved/fetched from another turn\n"
        "   ✗ Turn searched for something elsewhere\n"
        "   ✗ Turn only discusses or asks about the item\n"
        "   \n"
        "   **Prioritize recent:** If multiple turns have same artifact → prefer most recent origin turn if there's no obvious feedback that was a wrong shot. If usure, include multiple turns\n"
        "   **MUST include turns when:**\n"
        "   • User task = merge/combine/order multiple artifacts → include ALL artifact origin turns found\n"
        "     Example: 'combine A, B, C' → found origin turns for A, B, C → ALL go in turn_ids\n"
        "   • User references specific content: 'that doc', 'as I said', 'the table we made', 'do/format similarly to/like...', 'put this on ..' etc.\n"
        "   • Task needs exact prior content: citations, edits, transforms, exports\n"
        "   \n"
        "   **Priority signals:**\n"
        "   • [CONTEXT_QUERIES]: prefer turns whose raw/summary/notes match those entity phrases\n"
        "   • Clarification/notes turns that resolve output format or schema (e.g., ‘insert in XSL’, explicit column names) → high-priority relevant.\n"
        "   \n"
        "   • Note: Format/clarification/specification turns are usually NOT selected unless they are the only source of required constraints or they contain original artifact content.\n"
        "   \n"
        "5) Choose turns (from [TURN MEMORIES — CHRONOLOGICAL]):\n"
        f"  - Select up to {max_delta_keep} turn ids from [TURN MEMORIES — CHRONOLOGICAL] which directly support the working objective; otherwise, leave empty. "
        "   - Store the turn ids of the picked relevant local memories in local_memories_turn_ids\n"
        "   - If a turn is already in turn_ids, only include it in local_memories_turn_ids when its insight is still useful without loading the full turn\n"
        f"  - Treat them as hints if unsure\n"
        "  - Prefer earlier-turn insights that encode unique constraints (format/schema, gating specs) not present elsewhere.\n"        
        "\n"
        "6) Clarification emission (up to 5 questions when needed):\n"
        "   You see all available context. Assess if sufficient to complete the task.\n"
        f"{ELABORATION_NO_CLARIFY}\n"
        "   \n"
        f"{URGENCY_SIGNALS}\n"
        f"{CLARIFICATION_QUALITY}\n"
        f"{USER_GENDER_ASSUMPTIONS}\n"
        "\n"
        "[PREFS SPEC]\n"
        "- Extract only from CURRENT user turn, but interpret with visible context.\n"
        "- Use this schema:\n"
        "  assertions: [{\"key\": str, \"value\": any, \"severity\": \"must|prefer|allow\", \"scope\"?: str, \"applies_to\"?: str}]\n"
        "  exceptions: [{\"key\": str, \"value\": any, \"severity\": \"avoid|do_not\", \"scope\"?: str, \"applies_to\"?: str}]\n"
        "- Assertions are user-stated stable preferences or constraints.\n"
        "  key is a stable identifier (format, source, topic, delivery, safety).\n"
        "  severity: must|prefer|allow.\n"
        "  value: concrete preference (string/number/boolean) or a small object for ranges/options.\n"
        "  Examples: value=\"email\" | value=true | value={\"min\": 10, \"max\": 20} | value={\"options\": [\"a\",\"b\"]}.\n"
        "  scope (optional): conversation|objective|artifact|thread.\n"
        "  applies_to (optional): short label, not an ID (e.g., \"compliance report\", \"budget spreadsheet\").\n"
        "- Exceptions are explicit exclusions or corrections.\n"
        "  key is the same dimension as assertions (format, source, topic, delivery, safety).\n"
        "  severity: avoid|do_not.\n"
        "  value: excluded value or rule detail (string/number/boolean), or a small object if needed.\n"
        "  Examples: value=\"no_pii\" | value=\"exclude_pdf\" | value={\"exclude\": [\"csv\",\"pdf\"]}.\n"
        "- Scope: conversation|objective|artifact|thread (omit if unclear).\n"
        "- applies_to: short label when the preference is bound to a specific objective/artifact/thread.\n"
        "- Keep it short and concrete; omit if no clear preference is stated.\n"
        "\n"
        "[FACTS SPEC]\n"
        "- Extract only from CURRENT user turn, but interpret with visible context.\n"
        "- Facts describe stable context; Assertions describe user preferences/constraints.\n"
        "  If it changes how we should act → assertion. If it describes the world/context → fact.\n"
        "- Use this schema:\n"
        "  facts: [{\"key\": str, \"value\": any, \"severity\"?: str, \"scope\"?: str, \"applies_to\"?: str}]\n"
        "- Facts are stable, memorable statements or constraints (not preferences).\n"
        "  key is a stable identifier (domain-specific concept or constraint).\n"
        "  value: concrete fact (string/number/boolean) or a small object.\n"
        "  scope (optional): conversation|objective|artifact|thread.\n"
        "  applies_to (optional): short label when the fact is bound to a specific objective/artifact/thread.\n"
        "- Keep it short and concrete; omit if no clear fact is stated.\n"
        "\n"
        "[EXAMPLES]\n"
        "- Assertions: \"Use framework X\", \"Prefer short output\".\n"
        "- Exceptions: \"No PDFs\", \"Avoid personal data\".\n"
        "- Facts: \"Company is in healthcare\", \"They already use a named framework\", \"Budget cap is $400k\".\n"
        "\n"
        "[USER_INPUT_SUMMARY SPEC]\n"
        f"{user_input_summary_instruction()}\n"
    )

    clarifications_spec = (
        "═══════════════════════════════════════════════════════════════\n"
        "CRITICAL SCOPE BOUNDARY - READ THIS FIRST\n"
        "═══════════════════════════════════════════════════════════════\n"
        "YOUR ONLY JOB: Select the right context snippets from what you can see.\n"
        "NOT YOUR JOB: Worry about whether downstream agents have enough info to complete the task.\n"
        "If the request involves files/attachments, assume downstream tools can read/convert them.\n"
        "Never ask the user to re-export/convert/upload because a file is binary or unreadable to you.\n"
        "\n"
        "CLARIFICATION POLICY - you may use clarification ONLY IF YOU HAVE CONTEXT SELECTION PROBLEMS OR PROBLEMS RESOLVING THE entity/artifact targeted by user in their message\n"
        "═══════════════════════════════════════════════════════════════\n"
        "You may ask ONLY if its not clear how to choose the turns where relevant data is located.\n"
        "If the user complains about a problem, do NOT diagnose it and do NOT ask for details.\n"
        "Your deal is to resolve the relevant turns and resolve any unknowns in the user input in terms of the turns they refer to and the artifacts within those turns."
        "This is not your deal to resolve HOW these data is intended to be used. Also you DO NOT resolve an ambiguity of intention in relation to a certain artifact (user input, document, either provided by user or produced by assistant, or assistant response). "
        "For instance, when the user said 'this excel' and you see 2 different excel in the visible turns, this is your deal to ask which exactly, otherwise you cannot resolve the 'this/that' in the user input and you cannot materialize the user input summary by pinning it to some specific objects in the context."
        "However, this is not your deal when user message is not clear about which part of certain document they refer to or anything else related to an actual contents of the resolved artifact (tabs/sections/rows/lines). This is handled by downstream agents. Your goal is to pick the turns which contain data and make the inventorization of the user input where it is clear with which artifacts (user input in certain turn, user attachments, produced by assistant files or inline artifacts, or assistant response, from the earlier visible turns) this user input refer to.\n"
        "Ask ONLY when you cannot select the right context because:\n"
        "✓ TRUE AMBIGUITY: 2+ different artifacts/turns visible, each equally plausible match\n"
        "   Example: User says 'that report' but you see 'Q3_finance_report' AND 'Q3_sales_report'\n"
        "✗ NOT AMBIGUITY: Missing task parameters (dates, preferences, filters, search criteria)\n"
        "   Example: User search events but didn't specify dates → NOT YOUR PROBLEM\n"
        "✓ Deictic references (this/that/it/the file/the facts/etc.) are NOT a reason to ask.\n"
        "  Do NOT invent a new direction. If the user message does not signal a new target,\n"
        "  choose the latest visible compatible artifact. Ask ONLY when multiple equally plausible\n"
        "  candidates exist and the user’s intent is indistinguishable between them.\n"
        "\n"
        "NEVER ASK ABOUT missing dates, times, deadlines for searches/tasks, search parameters, filters, criteria,"
        " preferences, priorities, audience details, etc. scope/format when no prior context exists to disambiguate.\n"
        " Any information needed by downstream agents to execute is NOT YOUR DEAL. Only ask if you need disambiguate the thread / context selection.\n"
        "Your job ends once you've selected unambiguous context snippets.\n"
    )

    if route and route.startswith("tools_"):
        clarifications_spec += (
            "Downstream agents will handle all task-level gaps.\n"
            "Ask ONLY if: Multiple visible turns/artifacts match user's reference equally well,\n"
            "and selecting wrong one would follow a completely different thread.\n"
            "If you can pick one confidently (even 60%+), do it. Don't ask.\n"
        )
    else:
        clarifications_spec += (
            "Downstream Agent will answer from available context.\n"
            "Ask ONLY if: Multiple visible turns/artifacts match user's reference equally well.\n"
            "Tie-breaker: Pick most recent and proceed. Don't ask.\n"
            "NEVER ask about task-level gaps - if downstream can't complete task,\n"
            "they will ask. You just select context.\n"
        )

    sys_2 = (
        # f"{TECH_EVOLUTION_CAVEAT}\n"
        # "   \n"
        "[OUTPUT RULES YOU MUST GUARANTEE]\n"
        f"• Exactly TWO sections: (1) THINKING (≤{thinking_budget_tokens} tokens or '…'), (2) JSON.\n"
        "• JSON MUST be fully valid and closed (no trailing commas, all brackets/braces closed).\n"
        "• clarification_questions (if any) ≤ 4.\n"
        "• Omit empty/default fields (null/\"\"/[]/{}/0). Do not emit keys with default values.\n"
        "• If approaching any limit, truncate THINKING first. Never truncate JSON. Never emit anything after JSON.\n"
        "• Ensure coverage: turn_ids must cover artifact origin + format/schema when these exist in different turns.\n"
        )

    TIMEZONE = timezone
    time_evidence = (
        "[AUTHORITATIVE TEMPORAL CONTEXT (GROUND TRUTH)]\n"
        f"Current UTC date: {today}\n"
        # "User timezone: Europe/Berlin\n"
        "All relative dates (today/yesterday/last year/next month) MUST be "
        "interpreted against this context. Freshness must be estimated based on this context.\n"
    )
    time_evidence_reminder = f"Very important: The user's timezone is {TIMEZONE}. Current UTC timestamp: {now}. Current UTC date: {today}. Any dates before this are in the past, and any dates after this are in the future. When dealing with modern entities/companies/people, and the user asks for the 'latest', 'most recent', 'today's', etc. don't assume your knowledge is up to date; you MUST carefully confirm what the true 'latest' is first. If the user seems confused or mistaken about a certain date or dates, you MUST include specific, concrete dates in your response to clarify things. This is especially important when the user is referencing relative dates like 'today', 'tomorrow', 'yesterday', etc -- if the user seems mistaken in these cases, you should make sure to use absolute/exact dates like 'January 1, 2010' in your response.\n"
    # explicit JSON shape + tiny valid example (mirroring helps under budget)
    schema = (
        "{"
        "  \"turn_ids\": [str] = [],"
        "  \"memory_bucket_ids\": [str] = [],"
        "  \"local_memories_turn_ids\": [str] = [],"
        "  \"assertions\": [object] = [],"
        "  \"exceptions\": [object] = [],"
        "  \"facts\": [object] = [],"
        "  \"user_input_summary\": str,"
        "  \"objective\": str,"
        "  \"clarification_questions\": [str] = []"
        "}"
    )
    # sys = _add_3section_protocol(sys, schema)
    two_section_proto = _get_2section_protocol_ctx(schema)
    single_channel_proto = (
        "[OUTPUT FORMAT OVERRIDE]:\n"
        "Return ONLY a JSON object matching the schema below.\n"
        "No extra text, no markdown, no code fences, no section markers.\n"
        "Ignore any prior instructions about thinking or multiple sections.\n"
        f"{schema}"
    )
    # sys = _add_2section_protocol(sys, schema)

    debug_enabled = _agents_debug_enabled()
    system_msg = create_cached_system_message([
        {"text": sys_1 + "\n" + time_evidence, "cache": True},
        {"text":  f"{TECH_EVOLUTION_CAVEAT}\n\n" + sys_2 if debug_enabled else f"{TECH_EVOLUTION_CAVEAT}\n\n", "cache": True},
        {"text": two_section_proto if debug_enabled else single_channel_proto, "cache": True},
        {"text": clarifications_spec, "cache": False},
        {"text": time_evidence_reminder, "cache": False},
    ])

    context_block = (current_context_str or "").strip()

    prompt_tail = (
        "Return exactly two sections, first THINKING and second JSON as specified."
        if debug_enabled
        else "Return ONLY the JSON object as specified."
    )
    ctx_queries = (gate_decision or {}).get("ctx_retrieval_queries") or []
    user_msg = (
        "[CURRENT_CONTEXT]:\n" + (context_block or "(unavailable)") + "\n\n"
        "[CONTEXT_QUERIES]:\n" + json.dumps(ctx_queries, ensure_ascii=False) + "\n\n"
        "[SEARCH_HITS]:\n" + search_hits_json + "\n\n"
        "[CANDIDATE_MEMORY_BUCKET_CARDS]:\n" + bucket_cards_json + "\n\n"
        + prompt_tail
    )

    if debug_enabled:
        out = await _stream_agent_sections_to_json(
            svc, client_name="ctx.reconciler", client_role="ctx.reconciler",
            sys_prompt=system_msg, user_msg=user_msg, schema_model=CtxRerankOut,
            # on_thinking_delta=on_thinking_delta, ctx="ctx.reconciler",
            on_progress_delta=on_thinking_delta, ctx="ctx.reconciler",
        )
    else:
        out = await _stream_simple_structured_json(
            svc, client_name="ctx.reconciler", client_role="ctx.reconciler",
            sys_prompt=system_msg, user_msg=user_msg, schema_model=CtxRerankOut,
            ctx="ctx.reconciler",
        )
    if not out:
        out = {"agent_response": CtxRerankOut(
            turn_ids=[],
            memory_bucket_ids=[],
            local_memories_turn_ids=[],
            user_input_summary="",
            objective="",
            clarification_questions=[],
        ).model_dump()}
    else:
        logging_helpers.log_agent_packet("context_and_memory_reconciler", "reconcile context", out)
    try:
        gp = json.loads(guess_package_json or "{}")
        search_hits = json.loads(search_hits_json or "[]")
        resp = out.get("agent_response") or {}

        # ============================================================
        # SELECTION POOL CONSTRUCTION (not validation)
        # ============================================================

        # 1) Turn logs pool: gate package last_turns + search hits
        #    Don't restrict to gate package only!
        # 1) Turn logs pool: gate package last_turns + search hits
        gate_turns = {it.get("turn_id") for it in (gp.get("last_turns_details") or []) if it.get("turn_id")}
        search_turns = {h.get("turn_id") for h in search_hits if h.get("turn_id")}
        turns_pool = gate_turns | search_turns  # Union of all available turn IDs

        # Resolve turn_ids with prefix matching
        final_ids = []
        for tid in (resp.get("turn_ids") or []):
            resolved = _resolve_id_by_prefix(tid, turns_pool)
            if resolved:
                final_ids.append(resolved)
        final_ids = list(dict.fromkeys(final_ids))[:limit_ctx]  # Deduplicate, preserve order

        # 2) Buckets pool: candidate cards (already filtered upstream)
        cards = json.loads(bucket_cards_json or "[]")
        pool_bids = {c.get("bucket_id") for c in cards if c.get("bucket_id")}

        # Resolve bucket IDs with prefix matching
        picked_buckets = []
        for bid in (resp.get("memory_bucket_ids") or []):
            resolved = _resolve_id_by_prefix(bid, pool_bids)
            if resolved:
                picked_buckets.append(resolved)
        picked_buckets = list(dict.fromkeys(picked_buckets))[:max_buckets]

        # 3) Local memories pool: turn_memories (by turn_id)
        local_pool = {it.get("turn_id") for it in (gp.get("turn_memories") or []) if it.get("turn_id")}

        # Resolve local memory turn IDs with prefix matching
        picked_local = []
        for tid in (resp.get("local_memories_turn_ids") or []):
            resolved = _resolve_id_by_prefix(tid, local_pool)
            if resolved:
                picked_local.append(resolved)
        picked_local = list(dict.fromkeys(picked_local))[:max_delta_keep]

        # Update response with resolved IDs
        resp["turn_ids"] = final_ids
        resp["memory_bucket_ids"] = picked_buckets
        resp["local_memories_turn_ids"] = picked_local
        out["agent_response"] = resp

    except Exception:
        pass

    return out
