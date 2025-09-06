# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/codegen/team.py

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
        "You generate ONE Python 3.11 program that uses ONLY the documented adapters provided below.\n"
        "\nLANGUAGE & SYNTAX (HARD RULES):\n"
        "• This is Python, not JSON. Use Python literals: True/False/None (never 'true/false/null').\n"
        "• Build JSON strings with json.dumps(...), never by embedding JSON text in Python code.\n"
        "\nIMPORTS (HARD RULES):\n"
        "• Paste the adapter import **exactly as provided** in the adapters list; do not alter module path or alias.\n"
        "• Call functions exactly as in the provided call_template.\n"
        "\nRUNTIME CONTRACT:\n"
        "• CRITICAL: A global variable OUTPUT_DIR (contains string path) is defined for you at runtime. Do not redefine it!\n"
        "• Use it to read context and task: read OUTPUT_DIR/context.json and OUTPUT_DIR/task.json.\n"
        "• Write ALL outputs only into OUTPUT_DIR, exactly as declared in `outputs`.\n"
        "\nPERSISTENCE (IO ADAPTER — REQUIRED):\n"
        "• After every adapter call, use agent_io_tools.save_tool_output(\n"
        "  tool_id='<adapter id>', data=<raw_return>, params=json.dumps(<your args>), index=<0-based>) and collect returned filenames under result.raw_files[adapter_id].\n"
        "• At the end, write the final JSON via agent_io_tools.save_ret(data=json.dumps(result), filename='result.json').\n"
        "\nRESULT FILE (FIRST outputs[] ITEM) — REQUIRED KEYS:\n"
        "• On success include at least: ok=true, objective, contract (echoed slot→description), out_dyn (filled per contract).\n"
        "• Optional when applicable: queries_used, raw_files.\n"
        "• NEVER write result['out'] directly; save_ret will derive it from out_dyn. Any manual 'out' you add will be ignored.\n"
        "\nCONTRACT→out_dyn MAPPING (STRICT):\n"
        "• You receive a dynamic contract (slot_name→description). You MUST fill out_dyn with EXACTLY those slot names.\n"
        "• For FILE slots (e.g., pdf_file, slides_pptx, image_png, csv_file, zip_bundle):\n"
        "    out_dyn[slot] = {{\"file\": \"<relative path under OUTPUT_DIR>\", \"mime\": \"<mime>\", \"description\"?: \"...\", \"tool_id\"?: \"...\", \"tool_input\"?: {{...}} }}\n"
        "• For TEXT slots (e.g., summary_md, outline_md, caption_md, data_json, table_md, plan_md):\n"
        "    out_dyn[slot] = one of:\n"
        "      {{\"markdown\": \"...\"}} | {{\"text\": \"...\"}} | {{\"json\": <object or string>}}\n"
        "• resource_id for each derived artifact will be the slot name. Do NOT invent extra slots and do NOT omit required slots.\n"
        "• Paths must be RELATIVE to OUTPUT_DIR; if a tool requires an absolute path, construct it with OUTPUT_DIR but store the relative path in out_dyn.\n"
        "\nCITATIONS & SOURCES (NO FABRICATION):\n"
        "• Any external URLs/names/facts/dates you present must come from adapter returns that you saved via save_tool_output.\n"
        "• If you include a “sources” section or enable citations in a summarizer, build them ONLY from the actual search results fetched in THIS run.\n"
        "• Infra will auto-promote citable URLs from saved tool outputs; you must NOT put those URLs into out_dyn unless the contract explicitly requires a text sources slot.\n"
        "FILE PATHS & OUT ITEMS (CRITICAL):\n"
        "• All files MUST physically live inside OUTPUT_DIR.\n"
        "• If adapter returns a path, treat it as relative to OUTPUT_DIR.\n"
        "• If you pass the path to provider, it must be relative to OUTPUT_DIR path.\n"
        "• Do not resolve it to absoulte id store this file in result.out/out_dyn.\n"
        "• For file contract slots, set out_dyn[slot] = {\"file\": \"<OUTPUT_DIR-relative filename>\", \mime\"...\", \"description\": \"...\"}.\n"
        "\nSUMMARY MODES (llm_tools.summarize_llm):\n"
        "• Summarize free text → input_mode='text', text='<content>'.\n"
        "• Summarize search results with citations → build an array of sources like\n"
        "  [{{\"sid\": i, \"title\": r.title, \"url\": (r.url or r.href), \"text\": (r.body or r.snippet)}}] and call with\n"
        "  input_mode='sources', cite_sources=true, sources_json=json.dumps(sources).\n"
        "• Only set cite_sources=true when sources_json is actually provided.\n"
        "\nQUERY HYGIENE (web_search-like adapters):\n"
        "• Disambiguate terms (e.g., 'python programming language'), add sensible recency (e.g., 'past 7 days'), prefer domain qualifiers.\n"
        "• Record exact query strings used in result.queries_used.\n"
        "\nERROR HANDLING:\n"
        "• On any failure, write a JSON object to the FIRST output with ok=false, and keys: error, where, details..\n"
        "\nADAPTER USAGE (CRITICAL):\n"
        "• Import ONLY the provided adapter imports; stdlib allowed. Call adapters exactly per their docs. If a required arg is missing and not recoverable from context/task, fail gracefully.\n"
        "\nASYNC ADAPTERS:\n"
        "• If any adapter is async, implement 'async def main()' and run with 'asyncio.run(main())'.\n"
        "\nSTYLE & BEHAVIOR:\n"
        f"• Keep main.py ≤ {line_budget} lines; linear and concise. {'Be extra concise. ' if concise else ''}No prints.\n"
        "USER-FACING STATUS: two short lines (objective; plan). Do NOT name tools/providers/models.\n"
        f"Assume today={today} (UTC).\n"
    )

    # ---------- Strict 3-section protocol ----------
    sys = _add_3section_protocol(
        sys,
        "{"
        "  \"entrypoint\": \"python main.py\","
        "  \"files\": [ {\"path\": \"main.py\", \"content\": \"...\"} ],"
        "  \"outputs\": [ {\"filename\": \"result.json\", \"kind\": \"json\", \"key\": \"worker_output\"} ],"
        "  \"notes\": \"<=40 words\""
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
        "DYNAMIC OUTPUT CONTRACT YOU MUST FULFILL (slot → description):\n"
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
        "\nPARAMETER FILL POLICY (SCaffolding only):\n"
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
        "\nOUTPUT FORMAT:\n"
        "• Return up to 5 candidates with reasons and minimal parameters.\n"
        "\nINTERNAL THINKING (STATUS): tiny.\n"
        "USER-FACING (STATUS): 2 short lines (focus; minimal plan). No tool/provider/model names.\n"
    )
    ToolRouterOut.model_json_schema()
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

class OutputContract(BaseModel):
    must_produce: conlist(ContractItem, min_length=1) = Field(default_factory=list)
    nice_to_have: List[ContractItem] = Field(default_factory=list)


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


async def assess_solvability_stream(
        svc: ModelServiceBase,
        user_text: str,
        candidates: List[Dict[str, Any]],
        policy_summary: str = "",
        *,
        is_spec_domain: Optional[bool] = None,
        topics: Optional[List[str]] = None,
        on_thinking_delta=None,
        max_tokens=None,
) -> Dict[str, Any]:
    today = _today_str()
    sys = (
    "You are the 'Solvability Checker'.\n"
    "Decide if the request is answerable NOW using ONLY the provided candidates. Choose the mode.\n"
    "\nDYNAMIC OUTPUT CONTRACT — PURPOSE:\n"
    "• Provide output_contract_dyn: dict slot_name → short description (≤18 words).\n"
    "• This is the minimal checklist of deliverables the downstream program MUST emit; the answer generator will ingest only these.\n"
    "\nHARD RULES:\n"
    "• Do NOT solve the task. Consider only the provided candidates; do NOT invent tools.\n"
    "• Feasibility gate: every contract slot must be producible with the selected tools; otherwise adjust tools or set solvable=false.\n"
    "• Non-invention: never invent page counts, sections, URLs, dates, or substantive content.\n"
    "• Contract minimality: include ONLY what the objective requires.\n"
    "\nCLOSED-PLAN / COMPOSABILITY (CRITICAL):\n"
    "• Build a closed plan: any input needed by a selected tool must come from user/context OR from another selected tool.\n"
    "• If a renderer requires content (e.g., content_md/text/sources_json), you MUST also select a text/content producer.\n"    
    "\nLLM-READABLE MIRROR RULE (CRITICAL):\n"
    "• For every FILE deliverable, ALSO add one textual slot that captures its essential content for LLM consumption.\n"
    "  Examples (generic guidance, not a mandate):\n"
    "    pdf_file → summary_md; slides_pptx → outline_md; image_png → caption_md; csv_file → data_json or table_md.\n"
    "• If the objective already demands text (e.g., a summary), reuse that as the mirror slot.\n"
    "\nSLOT NAMING:\n"
    "• snake_case; file slots like: pdf_file, slides_pptx, image_png, csv_file, zip_bundle.\n"
    "• text slots like: summary_md, outline_md, caption_md, table_md, data_json, plan_md.\n"
    "• Include 'sources_md' ONLY if the user explicitly asked for sources/citations.\n"
    "\nMODE SELECTION (STRICT):\n"
    "• llm_only — no tools needed.\n"
    "• direct_tools_exec — allowed ONLY when exactly ONE tool is selected. (Multiple tools imply orchestration.)\n"
    "• codegen — choose whenever outputs of one tool feed another, or when multiple tools are selected for the plan.\n"
    "\nCLARIFYING QUESTIONS: add ≤2 only if ambiguity blocks progress.\n"
    f"Assume today={today} (UTC).\n"
    "INTERNAL THINKING: very concise. USER-FACING: two concise lines (assessment; action). No tool/provider names.\n"
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

