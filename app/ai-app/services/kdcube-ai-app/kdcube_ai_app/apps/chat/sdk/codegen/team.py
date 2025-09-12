# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/codegen/team.py

"""
Once logs are JSON, your _history_digest (and any “previous programs” retrieval) can pick the best prior execution by action and targets, not brittle headings. E.g., “find the latest run with action in {"edit","create"} and sections_added contains 'Security'”.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, conlist
from datetime import datetime, timezone
import json

from kdcube_ai_app.apps.chat.sdk.inventory import ModelServiceBase
from kdcube_ai_app.apps.chat.sdk.streaming.streaming import _add_3section_protocol, _stream_agent_sections_to_json

def _today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()

# ---------- Codegen schema ----------
class CodeFile(BaseModel):
    path: str = Field(..., description="Relative path inside the package, e.g., 'main.py'")
    content: str = Field(..., description="UTF-8 source")

class OutputSpec(BaseModel):
    filename: str = Field(..., description="Relative filename the program MUST write into OUTPUT_DIR")
    kind: Literal["json", "text", "binary"] = "json"
    key: Optional[str] = Field(default=None, description="Scratchpad key suggestion for this output")

class SolverCodegenOut(BaseModel):
    entrypoint: str = Field(..., description="Shell command to run the program, e.g., 'python main.py'")
    files: conlist(CodeFile, min_length=1)
    outputs: conlist(OutputSpec, min_length=1)
    notes: str = ""
    # short guidance for the final answer generator about how to read artifacts from this run
    result_interpretation_instruction: str = Field(
        default="",
        description="≤120 words. Explain how to interpret the deliverables of the code you generate (the solution) produced this run (by slot/type), "
                    "that they are system-provided context (not user-authored), how to cite new sources, "
                    "and how to refer to artifacts prduced by code, for example, any files (PDF, PPTX, CSV, etc.), in the final answer presented to a user."
    )

def _adapters_public_view(adapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strip any runtime-only fields and keep exactly what the model should see:
      - id, import, call_template, doc (purpose, args, returns, constraints, examples)
    """
    cleaned: List[Dict[str, Any]] = []
    for a in adapters or []:
        cleaned.append({
            "id": a.get("id"),
            "import": a.get("import"),
            "call_template": a.get("call_template"),
            "is_async": bool(a.get("is_async")),
            "doc": {
                "purpose": (a.get("doc") or {}).get("purpose", ""),
                "args": (a.get("doc") or {}).get("args", {}),             # {"name": "type & rules"} strings
                "returns": (a.get("doc") or {}).get("returns", ""),       # short description/shape
                "constraints": (a.get("doc") or {}).get("constraints", []), # ["k in 1..10", "query non-empty", ...]
                "examples": (a.get("doc") or {}).get("examples", []),     # ["fn(a=..., b=...)", ...]
            }
        })
    return cleaned

async def solver_codegen_stream(
        svc: ModelServiceBase,
        *,
        task: Dict[str, Any],
        adapters: List[Dict[str, Any]],
        solvability: Optional[Dict[str, Any]] = None,
        on_thinking_delta=None,
        ctx: Optional[str] = "solver_codegen",
) -> Dict[str, Any]:
    """
    Generates a self-contained Python 3.11 program that:
      - imports & calls ONLY the adapters we provide (your real libs) WITHIN documented usage
      - reads INPUTS from OUTPUT_DIR/context.json and OUTPUT_DIR/task.json if present
      - writes results to OUTPUT_DIR/<files specified in outputs[]>
      - prints nothing (silent), robust error handling
    """

    today = _today_str()

    # pull optional decision & constraints coming from the planner/ToolManager
    decision = (solvability or {})  # may include tools_to_use, reasoning, output_contract_dyn
    constraints = (task or {}).get("constraints") or {}

    # reasonable defaults
    line_budget = int(constraints.get("line_budget", 80))
    prefer_single_call = bool(constraints.get("prefer_single_call", True) or constraints.get("prefer_direct_tools_exec", False))
    minimize_logic = bool(constraints.get("minimize_logic", True))
    concise = bool(constraints.get("concise", True))

    # ---------- System prompt (authoritative; no ambiguity) ----------

    sys = (
        "# Codegen — single Python program\n"
        "\n"
        "## Authoritative inputs for **this** run\n"
        "- The **dynamic output contract** `output_contract_dyn` is provided **in THIS prompt**. Treat it as the single source of truth.\n"
        "- **Embed** the contract verbatim inside `main.py` (e.g., `CONTRACT = {...}` as a Python dict literal).\n"
        "- Do **not** read the contract from `task.json` or `context.json`.\n"
        "- You may read `OUTPUT_DIR/context.json` for prior history/sources and `OUTPUT_DIR/task.json` for objective/constraints (optional),\n"
        "  but **never** for the contract.\n"
        "\n"
        "## Language & syntax\n"
        "- Python 3.11. Use `True/False/None`. Build JSON with `json.dumps(...)`.\n"
        "\n"
        "## Imports & calls (hard rules)\n"
        "- Paste adapter imports **exactly** as provided; do not alter module paths or aliases.\n"
        "- Call functions exactly per the provided `call_template`.\n"
        "- Import the infra wrapper: `from io_tools import tools as agent_io_tools`.\n"
        "- **Wrap every adapter call** using the wrapper (logging + indexing):\n"
        "  `res = await agent_io_tools.tool_call(\\n"
        "       fn=<alias>.<fn>,\\n"
        "       params_json=json.dumps({<kwargs>}),\\n"
        "       call_reason=\"<5–12 words why this call is needed>\",\\n"
        "       tool_id=\"<qualified id exactly as in ADAPTERS list>\"\\n"
        "  )`\n"
        "- Do **not** call `save_tool_call` directly.\n"
        "\n"
        "## Runtime contract\n"
        "- A global `OUTPUT_DIR` is injected at runtime. Do **not** redefine it.\n"
        "- Write **all** files into `OUTPUT_DIR`, exactly as declared in `outputs[]`.\n"
        "\n"
        "## Persistence (required)\n"
        "- Finish by writing the final result: `await agent_io_tools.save_ret(data=json.dumps(result), filename=\"result.json\")`.\n"
        "\n"
        "## result.json (first outputs[] item)\n"
        "- On success include: `ok=true`, `objective`, `contract=CONTRACT` (echoed), and `out_dyn` (filled **exactly** per CONTRACT).\n"
        "\n"
        "## Contract → out_dyn (strict)\n"
        "- Use `CONTRACT` keys as the **only** keys of `out_dyn`.\n"
        "- **FILE** slots (e.g., `pdf_file`): `{\"file\":\"<OUTPUT_DIR-relative>\", \"mime\":\"<mime>\", \"description\":\"...\"}`.\n"
        "- **TEXT/STRUCT** slots (e.g., `project_canvas`, `summary_md`, `data_json`):\n"
        "  `{\"description\":\"...\", \"value\":\"<stringified>\", \"format\":\"markdown|plain_text|json|yaml|object\"}`.\n"
        "- The `resource_id` for each artifact is the slot name (infra will prefix with `slot:` automatically).\n"
        "- Paths stored in `out_dyn` **must** be `OUTPUT_DIR`-relative (never absolute).\n"
        "\n"
        "## Project canvas (critical)\n"
        "- The editable slot is **`project_canvas`**. Populate it with the final user-facing **Markdown** content only.\n"
        "- Do **not** include solver reasoning, change logs, or TODOs in the canvas.\n"
        "- Preserve existing `[[S:n]]` tokens; add new tokens **only** where NEW/CHANGED factual claims are made.\n"
        "\n"
        "## Project log (critical)\n"
        "- Second editable slot is **`project_log`**. Populate it with the concise list of changes taken on your turn.\n"
        "\n"
        "## History reuse (if applicable)\n"
        "- Read `OUTPUT_DIR/context.json → program_history` `H = program_history` (list of `{ <exec_id>: {...} }`).\n"
        "- Iterate newest→oldest: for each `E in H`, do `exec_id, inner = next(iter(E.items()))`.\n"
        "- Pick the first with a **non-empty** `inner[\"project_canvas\"][\"text\"]`.\n"
        "- `editable = inner[\"project_canvas\"][\"text\"]` (or empty string if none exists).\n"
        "- `prior_sources = inner.get(\"web_links_citations\",{}).get(\"items\",[])`.\n"
        "- If none has non-empty canvas, start a fresh one.\n"
        "\n"
        "## Citations (stable IDs)\n"
        "- Working sources = `prior_sources ∪ new_search_results`; **dedupe by URL** (case-insensitive).\n"
        "- Keep existing SIDs for existing URLs. New URLs keep **adapter-provided** SIDs (runtime seeds after the last used).\n"
        "- Never compress/backfill SIDs; gaps are OK.\n"
        "- Pass the full `sources_json` to the editor LLM and to PDF/PPTX renderers when resolving `[[S:n]]`.\n"
        "- If no new sources were needed, still pass `prior_sources` so existing `[[S:n]]` resolve.\n"
        "\n"
        "## Editor guidance (optional)\n"
        "- You may append a temporary block to the editor input:\n"
        "  `<!--GUIDANCE_START-->\\n<short actionable plan>\\n<!--GUIDANCE_END-->`.\n"
        "- Your edit instruction MUST require applying the guidance and **removing** the entire GUIDANCE block from the output.\n"
        "\n"
        "## Reference editor pipe\n"
        "1) `editable = selected.project_canvas.text` (or \"\").\n"
        "2) prev_project_log = selected.project_log.text (or "").\n"
        "3) guidance_md ... If prev_project_log exists, include only its last ~800 chars (or a one-paragraph summary) in a GUIDANCE block to avoid prompt bloat.\n"
        "4) `sources = merge_and_dedupe(prior.items, new_results)`  # keep old sids; new sids from adapter.\n"
        "5) `edited = agent_llm_tools.edit_text_llm(text=editable_with_guidance, instruction=\"Apply guidance; keep structure; no invented facts; add [[S:n]] only on NEW/CHANGED claims; REMOVE GUIDANCE block.\", keep_formatting=True, sources_json=json.dumps(sources), cite_sources=True, forbid_new_facts_without_sources=True)`.\n"
        "6) `out_dyn[\"project_canvas\"] = {\"description\":\"Updated project canvas (Markdown)\", \"value\": edited, \"format\":\"markdown\"}`.\n"
        "7) `out_dyn[\"project_log\"] = {\"description\":\"Added .. to project log\", \"value\": <your log>, \"format\":\"markdown\"}`.\n"
        "8) If rendering PDF/PPTX, pass the SAME `sources` to `write_pdf`/`write_pptx` with `resolve_citations=True`.\n"
        "\n"
        "## llm_tools.summarize_llm quick rules\n"
        "- `input_mode='text'` for free text; `input_mode='sources'` + `cite_sources=true` for search results.\n"
        "- Source rows: `{sid:int, title:str, url:str, text:str}`.\n"
        "\n"
        "## File/path rules\n"
        "- All files must physically live in `OUTPUT_DIR`.\n"
        "- Store `OUTPUT_DIR`-relative paths in `out_dyn` (e.g., `\"rust_advances.pdf\"`).\n"
        "\n"
        "## Error handling\n"
        "- On any failure: write `ok=false` with keys: `error`, `where`, `details` to the FIRST output file.\n"
        "\n"
        "## Async\n"
        "- If any adapter is async, implement `async def main()` and run with `asyncio.run(main())`.\n"
        "\n"
        "## Style & behavior\n"
        "- Linear and concise. No prints.\n"
        "- USER-FACING STATUS: two short lines (objective; plan). Do **not** name tools/providers/models.\n"
        "\n"
    )
    sys += (
        f"• Keep main.py ≤ {line_budget} lines.\n"
        f"Assume today={today} (UTC).\n"
    )

    # ---------- Strict 3-section protocol ----------
    sys = _add_3section_protocol(
        sys,
        "{"
        "  \"entrypoint\": \"python main.py\","
        "  \"files\": [ {\"path\": \"main.py\", \"content\": \"...\"} ],"
        "  \"outputs\": [ {\"filename\": \"result.json\", \"kind\": \"json\", \"key\": \"worker_output\"} ],"
        "  \"notes\": \"<=40 words\","
        "  \"result_interpretation_instruction\": \"<=120 words, tool-agnostic, concise\""
        "}"
    )

    # ---------- Message: task + adapters with DOCS ----------
    adapters_for_llm = _adapters_public_view(adapters)

    contract_dyn = (decision or {}).get("output_contract_dyn") or {}

    msg = (
        "TASK (objective + constraints for this program):\n"
        f"{json.dumps(task or {}, ensure_ascii=False, indent=2)}\n\n"
        "SOLVABILITY / DECISION (read-only hints):\n"
        f"{json.dumps(decision or {}, ensure_ascii=False, indent=2)}\n\n"
        "DYNAMIC OUTPUT CONTRACT YOU MUST FULFILL - output_contract_dyn (slot → description):\n"
        f"{json.dumps(contract_dyn, ensure_ascii=False, indent=2)}\n\n"
        "ADAPTERS — imports, call templates, is_async:\n"
        f"{json.dumps([{k: v for k,v in a.items() if k in ('id','import','call_template','is_async')} for a in adapters_for_llm], ensure_ascii=False, indent=2)}\n\n"
        "TOOL DOCS (purpose/args/returns/constraints/examples):\n"
        f"{json.dumps([{ 'id': a['id'], 'doc': a.get('doc', {}) } for a in adapters_for_llm], ensure_ascii=False, indent=2)}\n\n"
        "Produce the three sections as instructed."
    )

    # ---------- Stream ----------
    return await _stream_agent_sections_to_json(
        svc,
        client_name="solver_codegen",
        client_role="solver_codegen",
        sys_prompt=sys,
        user_msg=msg,
        schema_model=SolverCodegenOut,
        on_thinking_delta=on_thinking_delta,
        ctx=ctx,
        max_tokens=6000,
    )

# ====================== TOOL ROUTER (topic- & domain-aware) ======================
class ToolCandidate(BaseModel):
    name: str                              # e.g., "vuln_db"
    reason: str = ""
    confidence: float = Field(0.6, ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ToolRouterOut(BaseModel):
    candidates: List[ToolCandidate] = Field(default_factory=list)
    notes: str = ""                        # short commentary

async def tool_router_stream(
        svc: ModelServiceBase,
        user_text: str,
        policy_summary: str = "",
        context_hint: str = "",
        topic_hint: str = "",
        prefs_hint: Dict[str, Any] | None = None,
        *,
        topics: Optional[List[str]] = None,
        tool_catalog: Optional[List[Dict[str, Any]]] = None,
        on_thinking_delta=None,
        max_tokens=None,
) -> Dict[str, Any]:
    today = _today_str()
    sys = (
        "You are a Tool Router. Using the TOOL CATALOG (id, purpose, args), select at most 5 tools that materially help.\n"
        "If none helps, return [].\n"
        f"Assume today={today} (UTC).\n"
        "\nHARD RULES:\n"
        "• Do NOT solve the task. Do NOT run tools. Do NOT invent facts, URLs, dates, or long free text.\n"
        "• Only select tools present in the catalog. For each selection, 'name' MUST equal the catalog 'id'.\n"
        "\nPARAMETER FILL POLICY (Scaffolding only):\n"
        "• Provide MINIMAL, PLAUSIBLE scaffolding for parameters (booleans, enums, small numerics, simple flags).\n"
        "• For any contentful parameter (e.g., text, content_md/markdown, sources_json, url lists, file bodies):\n"
        "    - Do NOT invent content. Use a short placeholder like \"<TBD at runtime>\" or omit the param.\n"
        "• Safe defaults are ok (e.g., n=5, max_tokens=300, style='brief'), but never prewrite summaries or links.\n"
        "\nCLOSED-PLAN COMPOSABILITY CHECK (CRITICAL):\n"
        "• The selected set must form a *closed plan* that can produce the user's requested deliverable end-to-end in a single run,\n"
        "  with NO human-in-the-loop and NO external steps. If a tool requires an input (e.g., content_md for a PDF renderer), ensure that\n"
        "  input is either provided by the user/context OR produced by another selected tool. Otherwise add the missing generator/transformer tool.\n"
        "• If the goal requires *text construction/transformation* (e.g., summary, outline, caption, extraction), include a text/LLM transformer tool from the catalog.\n"
        "• Minimize redundancy: avoid overlapping tools when one suffices; prefer the smallest closed set that keeps the plan feasible.\n"
        "\nSEQUENCING ASSUMPTION:\n"
        "• It is acceptable if outputs must flow between selected tools; the downstream solver will orchestrate (likely via codegen). Your job is to pick a feasible set.\n"
        "CONTEXT-AWARE SEARCH RULE:\n"
        "• If the user intent is to *add / expand / modify / improve*, and there's no strict anti-recommendation for adding a web/search tool, include such tools if relevant.\n"
        "\nOUTPUT FORMAT:\n"
        "• Return up to 5 candidates with reasons and minimal parameters.\n"
        "\nINTERNAL THINKING (STATUS): tiny.\n"
        "USER-FACING (STATUS): 2 short lines (focus; minimal plan). No tool/provider/model names.\n"
    )
    ToolRouterOut.model_json_schema()
    sys += (
        "\nREUSE CONTEXT:\n"
        "• Downstream code will read prior runs from OUTPUT_DIR/context.json → program_history[]. "
        "Prefer selecting tools that can EDIT or UPDATE existing deliverables when appropriate (e.g., LLM editor, file writer), "
        "instead of rebuilding everything from scratch. Only applicable if the new request applies to a past program in the context.\n"
    )
    sys = _add_3section_protocol(
        sys,
        "{ \"candidates\": ["
        "  {\"name\": \"<tool_id>\", \"reason\": \"...\", \"confidence\": 0..1, \"parameters\": {\"a\": \"hello\"}}"
        "], \"notes\": \"(<=25 words)\" }"
    )

    catalog_str = ""
    if tool_catalog:
        preview = [
            {"id": t.get("id"),
             "purpose": (t.get("doc") or {}).get("purpose",""),
             "args": (t.get("doc") or {}).get("args", {})}
            for t in tool_catalog
        ]
        catalog_str = f"TOOL CATALOG:\n{json.dumps(preview, ensure_ascii=False, indent=2)}\n\n"

    msg = (
            catalog_str +
            f"User question:\n{user_text}\n\n"
            f"{'Topics: ' + ', '.join(topics[:6]) if topics else f'Topics (hint): {topic_hint}'}\n"
            f"Policy/context hints:\n{policy_summary[:800]}\n\n"
            f"Preferences hint (assertions/exceptions; treat as constraints when selecting tools):\n{json.dumps((prefs_hint or {}), ensure_ascii=False)[:1200]}\n\n"
            f"Conversation cue:\n{context_hint[:400]}\n\n"
            "Produce the three sections as instructed."
    )

    out = await _stream_agent_sections_to_json(
        svc, client_name="tool_router", client_role="tool_router",
        sys_prompt=sys, user_msg=msg, schema_model=ToolRouterOut,
        on_thinking_delta=on_thinking_delta,
        max_tokens=max_tokens
    )
    out = out or {}
    return out


# ====================== SOLVABILITY (with optional domain/topics) ======================
# ---------- Output contract schema ----------
from typing import Literal

class ContractItem(BaseModel):
    rid: str = Field(..., description="Stable resource id the program MUST use in result.out[].")
    type: Literal["inline","file"] = "inline"
    # For inline
    format: Optional[Literal["markdown","text","json","url"]] = None
    # For file
    mime: Optional[str] = None
    filename_hint: Optional[str] = None

    description: str = ""
    citable: Optional[bool] = None
    source_hint: Optional[str] = Field(default=None, description="Either 'program' or an adapter id to prefer (e.g., 'generic_tools.write_pdf').")
    require_text_fallback: bool = Field(default=False, description="If type='file', require a paired inline description/content.")
    fallback_rid: Optional[str] = Field(default=None, description="Rid to use for the text fallback when required.")
    min_count: int = 1
    max_count: Optional[int] = 1

class SolvabilityOut(BaseModel):
    solvable: bool
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    tools_to_use: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)

    # NOTE: unified names — no 'single_call' anymore
    solver_mode: Literal["direct_tools_exec","codegen","llm_only"] = "llm_only"
    solver_reason: str = ""  # short justification

    # Dynamic contract the program must fulfill iff solver_mode='codegen'
    # map: slot name -> human description of what should be produced
    output_contract_dyn: Optional[Dict[str, str]] = Field(default_factory=dict)

    context_use: bool = True
    project_canvas_slot: str = "project_canvas"                  # the slot codegen MUST populate for text representation of solution
    project_log_slot: str = "project_log"
    history_select: str = "latest"                      # 'latest' | 'by_mention' | 'by_similarity'
    history_select_hint: str = ""                       # short instruction when not 'latest'
    citations_source_path: str = "program_history[].<exec>.web_links_citations.items"
    instructions_for_codegen: str = ""                  # ≤80 words, concrete steps to read context and pick version


async def assess_solvability_stream(
        svc: ModelServiceBase,
        user_text: str,
        candidates: List[Dict[str, Any]],
        policy_summary: str = "",
        prefs_hint: Dict[str, Any] | None = None,
        *,
        is_spec_domain: Optional[bool] = None,
        topics: Optional[List[str]] = None,
        on_thinking_delta=None,
        max_tokens=None,
) -> Dict[str, Any]:

    today = _today_str()
    sys = (
        "# Solvability Checker\n"
        "\n"
        "## Goal\n"
        "- Decide if the request is answerable **now** using ONLY the provided tool candidates.\n"
        "- If solvable, emit a minimal **DYNAMIC OUTPUT CONTRACT** the downstream program MUST produce.\n"
        "\n"
        "## Hard rules\n"
        "- Do **not** solve the task. Do **not** invent tools or content, dates, URLs, or section lists.\n"
        "- **Feasibility gate:** every contract slot must be producible with the selected tools; otherwise change tools or set `solvable=false`.\n"
        "- **Contract minimality:** include **only** what the user objective requires.\n"
        "\n"
        "## Modes\n"
        "- `llm_only` — no tools needed.\n"
        "- `direct_tools_exec` — allowed **only** when exactly one tool is selected.\n"
        "- `codegen` — choose when multiple tools are needed or when outputs must flow between tools.\n"
        "\n"
        "## Closed-plan requirement\n"
        "- Any input needed by a selected tool must come from user/context **or** be produced by another selected tool.\n"
        "\n"
        "## Project canvas (critical)\n"
        "- ALWAYS include a slot named **`project_canvas`** in `output_contract_dyn`. This is the single editable, textual mirror of the project.\n"
        "- The canvas must contain the actual user-facing Markdown content, including any `[[S:n]]` tokens.\n"
        "- Do **not** put solver notes, change logs, or internal reasoning into the canvas.\n"
        "- If the request modifies a prior project, prefer **edit/update** over full regeneration.\n"
        "\n"
        "## Project log (critical)\n"
        "- ALWAYS include a slot named **`project_log`** in `output_contract_dyn`. This is the continuous slot which is filled during project by its internal editors.\n"
        "- The project log must contain the description of objective, actual edits to take and the user current intent or preferences on this turn\n"
        "\n"        
        "## History reuse (if applicable)\n"
        "- The downstream program can read `OUTPUT_DIR/context.json → program_history[]`.\n"
        "- Set `history_select`: `latest` | `by_mention` | `by_similarity`. If not `latest`, add a short `history_select_hint`.\n"
        "- Provide concise `instructions_for_codegen`: how to pick **one** prior version - at \n"
        "  `program_history[i][<exec_id>].project_canvas.text`, prior project log at "
        "  `program_history[i][<exec_id>].project_log.text`and \n"
        "  `prior citations at program_history[i][<exec_id>].web_links_citations.items`.\n"
        "\n"
        "## Citations\n"
        "- The editor inserts `[[S:n]]` for **new/changed** claims. Never ask to renumber existing tokens.\n"
        "\n"
        "## Slot naming\n"
        "- snake_case.\n"
        "- Files: `pdf_file`, `slides_pptx`, `image_png`, `csv_file`, `zip_bundle`.\n"
        "- Text/struct: `project_canvas`, `project_log`, `summary_md`, `outline_md`, `table_md`, `data_json`, `plan_md`.\n"
        "- Only add a separate `sources_md` slot if the user explicitly asks for sources outside the file render so they are needed separately of internal project canvas.\n"
        "\n"
        "## Clarifying questions\n"
        "- Ask ≤2 only if ambiguity **blocks** progress.\n"
        "\n"
    )
    sys += (
        f"Assume today={today} (UTC).\n"
        "\n"
        "INTERNAL THINKING: very concise.\n"
        "USER-FACING STATUS: two short lines (assessment; next action). No tool/provider names.\n"
    )
    sys = _add_3section_protocol(
        sys,
        "{"
        "  \"solvable\": bool,"
        "  \"confidence\": 0..1,"
        "  \"reasoning\": \"(<=25 words)\","
        "  \"tools_to_use\": [\"<tool_id>\"],"
        "  \"clarifying_questions\": [\"...\",\"...\"],"
        "  \"solver_mode\": \"direct_tools_exec\"|\"codegen\"|\"llm_only\","
        "  \"solver_reason\": \"(<=25 words)\","
        "  \"output_contract_dyn\": {\"<slot>\": \"<description>\"}"
        "}"
    )
    topic_line = f"Topics: {', '.join(topics[:6])}" if topics else ""
    domain_line = f"is_spec_domain={is_spec_domain!s}"
    msg = (
        f"{domain_line}\n{topic_line}\n"
        f"User question:\n{user_text}\n"
        f"Policy/context summary:\n{policy_summary[:800]}\n"
        f"Preferences hint (use to constrain what to produce/avoid):\n{json.dumps((prefs_hint or {}), ensure_ascii=False)[:1200]}\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}\n"
        "Produce the three sections as instructed."
    )

    out = await _stream_agent_sections_to_json(
        svc,
        client_name="solvability",
        client_role="solvability",
        sys_prompt=sys,
        user_msg=msg,
        schema_model=SolvabilityOut,
        on_thinking_delta=on_thinking_delta,
        ctx="solvability",
        max_tokens=max_tokens
    )
    out = out or {}
    agent_response = out.setdefault("agent_response", {})

    # constrain to provided candidates
    cand_names = {c.get("name") for c in (candidates or []) if c.get("name")}
    tools = [t for t in (agent_response.get("tools_to_use") or []) if t in cand_names]
    agent_response["tools_to_use"] = tools

    if (agent_response.get("solver_mode") == "direct_tools_exec"
            and len(tools) != 1):
        agent_response["solver_mode"] = "codegen"
    agent_response["solver_reason"] = "Multiple tools selected requires orchestration."

    # --- Robust defaults if model is sparse/quiet ---
    if not cand_names:
        # Can still be solvable without tools
        agent_response.setdefault("solver_mode", "llm_only")
    else:
        # If exactly one candidate selected → prefer direct_tools_exec; else codegen
        if not agent_response.get("solver_mode"):
            agent_response["solver_mode"] = "direct_tools_exec" if len(tools) == 1 else "codegen"

    if not agent_response.get("solvable") and not agent_response.get("clarifying_questions"):
        agent_response["clarifying_questions"] = ["Add one concrete detail so I can tailor the next step."]

    return out

