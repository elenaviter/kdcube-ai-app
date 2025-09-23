# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/codegen/contracts.py

from __future__ import annotations
from pydantic import BaseModel, Field
import json
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

class ProgramInputs(BaseModel):
    objective: str = ""
    topics: List[str] = Field(default_factory=list)
    policy_summary: str = ""
    constraints: Dict[str, object] = Field(default_factory=dict)
    tools_selected: List[str] = Field(default_factory=list)

class FileRef(BaseModel):
    filename: str
    key: Optional[str] = None     # conversation-store key (if rehosted)
    mime: Optional[str] = None
    size: Optional[int] = None
    description: str = ""
    slot: Optional[str] = None
    tool_id: Optional[str] = None

class InlineRef(BaseModel):
    mime: Optional[str] = None
    citable: bool = False
    description: str = ""
    value_preview: str = ""       # short preview for indexing/UX
    slot: Optional[str] = None
    tool_id: Optional[str] = None

class Deliverable(BaseModel):
    slot: str
    description: str = ""
    files: List[FileRef] = Field(default_factory=list)
    inlines: List[InlineRef] = Field(default_factory=list)

class ProgramBrief(BaseModel):
    title: str = "Codegen Program"
    language: str = "python"
    codegen_run_id: Optional[str] = None
    inputs: ProgramInputs = Field(default_factory=ProgramInputs)
    deliverables: List[Deliverable] = Field(default_factory=list)
    notes: List[Any] = Field(default_factory=list)

# ---------- Data ----------

@dataclass
class PlannedTool:
    id: str
    purpose: str
    params: Dict[str, Any]
    reason: str
    confidence: float

@dataclass
class SolutionPlan:
    mode: str                 # "llm_only" | "tools"
    tools: List[PlannedTool]  # subset of candidates with concrete params
    confidence: float
    reasoning: str
    clarification_questions: Optional[List[str]] = None
    instructions_for_downstream: Optional[str] = None
    error: Optional[str] = None
    failure_presentation: Optional[dict] = None
    tool_router_notes: Optional[str] = None
    contract_dyn: Optional[Dict[str, str]] = None  # slot -> description
    service: Optional[dict] = None
    solvable: Optional[bool] = None

    @property
    def tool_selector_internal_thinking(self) -> str:
        tr_service = (self.service or {}).get("tool_router") or {}
        return tr_service.get("internal_thinking")

    @property
    def solvability_internal_thinking(self) -> str:
        sv_service = (self.service or {}).get("solvability") or {}
        return sv_service.get("internal_thinking")

    @property
    def tool_selector_raw_data(self) -> str:
        tr_service = (self.service or {}).get("tool_router") or {}
        return tr_service.get("raw_data")

    @property
    def solvability_raw_data(self) -> str:
        sv_service = (self.service or {}).get("solvability") or {}
        return sv_service.get("raw_data")

    @property
    def tool_selector_error(self) -> str:
        tr_service = (self.service or {}).get("tool_router") or {}
        return tr_service.get("error")

    @property
    def solvability_error(self) -> str:
        sv_service = (self.service or {}).get("solvability") or {}
        return sv_service.get("error")

@dataclass
class SolutionExecution:
    error: Optional[str] = None
    failure_presentation: Optional[dict] = None
    out: Optional[dict] = None
    citations: Optional[List[dict]] = None
    calls: Optional[List[dict]] = None  # tool calls made
    deliverables: Optional[Dict[str, Any]] = None  # slot -> {description, value}


@dataclass
class ToolModuleSpec:
    ref: str                 # dotted path or file path (abs/rel)
    use_sk: bool = False     # introspect via Semantic Kernel metadata
    alias: Optional[str] = None  # import alias for 'tools' (unique per module)


@dataclass
class SolveResult:
    raw: Dict[str, Any]
    _round0: Dict[str, Any] = field(default_factory=dict, init=False)

    # ----- core blocks -----
    @property
    def codegen(self) -> Dict[str, Any]:
        return (self.raw or {}).get("codegen") or {}

    @property
    def plan(self) -> SolutionPlan:
        return self.raw.get("plan")

    @property
    def execution(self) -> SolutionExecution:
        return self.raw.get("execution")

    @property
    def failure(self):
        """Error message if the solve failed."""
        return self.plan.error or (not self.execution and self.plan.mode != "llm_only") or (self.execution and self.execution.error)

    @property
    def program_presentation(self):
        return _build_program_presentation_for_answer_agent(
            sr=self,
            citations=self.citations(),
            codegen_run_id=self.run_id(),
        )

    def program_brief(self, rehosted_files: List[dict]) -> Tuple[str, ProgramBrief]:
        return _program_brief_from_contract(self, rehosted_files=rehosted_files)

    @property
    def failure_presentation(self):
        """Structured failure payload (if any) from plan or execution."""
        if self.plan and self.plan.failure_presentation:
            return self.plan.failure_presentation
        exec_ = self.execution
        return exec_.failure_presentation if exec_ and exec_.failure_presentation else None

    def rounds(self) -> List[Dict[str, Any]]:
        return self.codegen.get("rounds") or []

    def result_json(self) -> Optional[Dict[str, Any]]:
        r = self._first_round()
        try:
            items = ((r.get("outputs") or {}).get("items") or [])
            for it in items:
                if it.get("filename") == "result.json" and isinstance(it.get("data"), dict):
                    return it.get("data")
        except Exception:
            pass
        return None

    # ----- out accessors (result.json as source of truth) -----
    def out_items(self) -> List[Dict[str, Any]]:
        """Everything that codegen produced under result.json['out']."""
        rj = self.result_json() or {}
        arr = rj.get("out") or []
        return arr if isinstance(arr, list) else []

    def execution_id(self) -> Optional[str]:
        """Execution/session id emitted by codegen (if present in result.json)."""
        rj = self.result_json() or {}
        eid = rj.get("execution_id")
        if isinstance(eid, str) and eid.strip():
            return eid.strip()
        # fallback to codegen run_id if not present
        return self.run_id()

    # ----- contract/deliverables -----
    def deliverables_map(self) -> Dict[str, Any]:
        """
        Standardized structure returned by CodegenToolManager.solve():
          { slot_name: { "description": str, "value": <artifact dict or None> }, ... }
        NOTE: 'value' is a SINGLE artifact dict for the slot (not a list).
        """
        exec_ = self.execution
        return {} if exec_ is None or exec_.deliverables is None else exec_.deliverables

    def deliverable_slots(self) -> Set[str]:
        """Names of contract slots."""
        return set(self.deliverables_map().keys())

    def deliverables_out(self) -> List[Dict[str, Any]]:
        """Flattened SINGLE artifact per slot (if present)."""
        out: List[Dict[str, Any]] = []
        for _, spec in (self.deliverables_map() or {}).items():
            val = (spec or {}).get("value")
            if isinstance(val, dict):
                out.append(val)
        return out

    # ----- reasoning & hints from the first codegen round -----
    def interpretation_instruction(self) -> str:
        r = self._first_round()
        exec_instruction = r.get("result_interpretation_instruction")
        if not exec_instruction:
            if not self.plan.solvable:
                return f"The objective is considered as not solvable.\nReasoning: {self.plan.reasoning}.\nMessage for answer consolidator: {self.plan.instructions_for_downstream}"
        return ""

    def round_reasoning(self) -> str:
        r = self._first_round()
        return r.get("internal_thinking") or ""

    def round_notes(self) -> str:
        """Return the last non-empty note string from the first round, if any."""
        r = self._first_round()
        notes = r.get("notes")
        if isinstance(notes, list):
            for s in reversed(notes):
                if isinstance(s, str) and s.strip():
                    return s.strip()
        if isinstance(notes, str):
            return notes.strip()
        return ""

    # ----- derived fields -----
    def run_id(self) -> Optional[str]:
        cg = self.codegen
        rid = cg.get("run_id") if cg else None
        if rid:
            return rid
        r = self._first_round()
        return r.get("run_id") if isinstance(r, dict) else None

    def outdir_workdir(self) -> Tuple[Optional[str], Optional[str]]:
        r = self._first_round()
        return (r.get("outdir"), r.get("workdir"))

    # ----- citations: url+title(+text) only -----
    # def citations(self) -> List[Dict[str, Any]]:
    #     """
    #     Extract citable items from result.json['out'] and normalize:
    #       [{url, title, text?}, ...]
    #     """
    #     cites: List[Dict[str, Any]] = []
    #     for row in self.out_items():
    #         if not isinstance(row, dict):
    #             continue
    #         if row.get("type") != "inline" or not bool(row.get("citable")):
    #             continue
    #
    #         data = row.get("output") or row.get("value")
    #         # Unified downstream expects url+title
    #         def _push(d: Dict[str, Any]):
    #             d = d or {}
    #             url = str(d.get("url") or "").strip()
    #             if not url:
    #                 return
    #             d["url"] = url
    #             cites.append({**d,
    #                           "tool_id": row.get("tool_id") or "",
    #                           "resource_id": row.get("resource_id") or ""})
    #
    #         if isinstance(data, list):
    #             for c in data:
    #                 if isinstance(c, dict):
    #                     _push(c)
    #         elif isinstance(data, dict):
    #             _push(data)
    #
    #     # dedupe by URL
    #     seen, uniq = set(), []
    #     for c in cites:
    #         u = c.get("url")
    #         if u and u not in seen:
    #             seen.add(u)
    #             uniq.append(c)
    #     return uniq

    def citations(self) -> List[Dict[str, Any]]:
        """
        Prefer the latest ctx_tools.merge_sources output (one canonical list).
        If none exists in this turn, fall back to flattening all citable inline items.
        Keeps `sid` (int) when present, and annotates each item with tool/resource ids
        so indexing can tag by tool/resource.
        """
        import re

        items = self.out_items() or []

        def _rid_index(row: Dict[str, Any]) -> int:
            """Parse trailing :<n> from resource_id for ordering."""
            try:
                rid = str(row.get("resource_id") or "")
                m = re.search(r":(\d+)$", rid)
                return int(m.group(1)) if m else -1
            except Exception:
                return -1

        def _norm_one(c: Dict[str, Any], *, tool_id: str, resource_id: str) -> Optional[Dict[str, Any]]:
            if not isinstance(c, dict):
                return None
            url = str(c.get("url") or c.get("href") or "").strip()
            if not url:
                return None
            title = c.get("title") or c.get("description") or url
            text  = c.get("text") or c.get("body") or ""
            sid   = c.get("sid")
            try:
                sid = int(sid) if sid is not None and str(sid).strip() != "" else None
            except Exception:
                sid = None

            row = {"url": url, "title": title, "tool_id": tool_id, "resource_id": resource_id}
            if text:
                row["text"] = text
            if sid is not None:
                row["sid"] = sid
            return row

        # 1) Prefer the latest ctx_tools.merge_sources inline+citable
        ms_rows = [
            r for r in items
            if isinstance(r, dict)
               and r.get("type") == "inline"
               and r.get("citable") is True
               and r.get("tool_id") == "ctx_tools.merge_sources"
        ]

        if ms_rows:
            latest = max(ms_rows, key=_rid_index)
            out = latest.get("output")
            candidates = out if isinstance(out, list) else []
            normalized: List[Dict[str, Any]] = []
            seen_urls = set()
            for c in candidates:
                rec = _norm_one(c, tool_id=latest.get("tool_id") or "", resource_id=latest.get("resource_id") or "")
                if not rec:
                    continue
                u = rec["url"]
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                normalized.append(rec)
            return normalized

        # 2) Fallback: collect from all citable inline items (e.g., web_search, kb_search, browsing)
        normalized: List[Dict[str, Any]] = []
        seen_urls = set()
        for row in items:
            if not (isinstance(row, dict) and row.get("type") == "inline" and row.get("citable") is True):
                continue
            out = row.get("output")
            pack: List[Dict[str, Any]] = []
            if isinstance(out, list):
                pack = [c for c in out if isinstance(c, dict)]
            elif isinstance(out, dict):
                pack = [out]
            for c in pack:
                rec = _norm_one(c, tool_id=row.get("tool_id") or "", resource_id=row.get("resource_id") or "")
                if not rec:
                    continue
                u = rec["url"]
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                normalized.append(rec)
        return normalized


    def indexable_tool_ids(self) -> Set[str]:
        """
        Tools we might want to index (tags-only) from result.json style items.
        Keep it conservative and only pick obvious search-like tools.
        """
        def _is_search(tid: str) -> bool:
            tid = (tid or "").lower()
            return tid.endswith(".web_search") or tid.endswith(".kb_search")
        return {it.get("tool_id") for it in self.out_items() if _is_search(it.get("tool_id") or "")} - {None}

    # ----- helpers -----
    def _first_round(self) -> Dict[str, Any]:
        if self._round0:
            return self._round0
        rr = self.rounds()
        self._round0 = (rr[0] if rr else {}) if isinstance(rr, list) else {}
        return self._round0

def _program_brief_from_contract(sr: SolveResult,
                                 rehosted_files: List[dict]) -> Tuple[str, ProgramBrief]:
    """
    Returns:
      (brief_text, ProgramBrief)
    """
    sr = sr or {}
    codegen = sr.codegen
    plan = sr.plan
    execution = sr.execution
    contract = plan.contract_dyn
    deliverables = execution.deliverables

    # ---- derive title / language / inputs ----
    program = (codegen.get("program") or {})
    title = (program.get("title") or "").strip() or "Codegen Program"
    language = (program.get("language") or "python").strip() or "python"

    rounds = (codegen.get("rounds") or [])
    latest_round = next((r for r in reversed(rounds) if isinstance(r, dict)), {}) if rounds else {}
    if not title:
        nt = (latest_round.get("notes") or "")
        if isinstance(nt, str) and nt.strip():
            title = nt.strip()[:120]

    inputs_raw = (latest_round.get("inputs") or {})
    inputs = ProgramInputs(
        objective=inputs_raw.get("objective") or "",
        topics=list(inputs_raw.get("topics") or []),
        policy_summary=inputs_raw.get("policy_summary") or "",
        constraints=dict(inputs_raw.get("constraints") or {}),
        tools_selected=list(inputs_raw.get("tools_selected") or []),
    )

    # ---- normalize deliverables into structured model ----
    # deliverables shape here: {slot: {"description": str, "value": [artifact,...]}}
    struct_delivs: List[Deliverable] = []
    for slot, dv in (deliverables or {}).items():
        desc = (dv.get("description") or "") if isinstance(dv, dict) else ""
        files: List[FileRef] = []
        inlines: List[InlineRef] = []

        v_raw = (dv.get("value") if isinstance(dv, dict) else None)
        vals: List[Any] = (v_raw if isinstance(v_raw, list) else ([v_raw] if isinstance(v_raw, dict) else []))

        for it in vals:
            if not isinstance(it, dict):
                # inline preview from unknown type
                inlines.append(InlineRef(
                    mime="application/json",
                    citable=False,
                    description="result",
                    value_preview=(json.dumps(it, ensure_ascii=False)[:280]),
                    slot=slot, tool_id=None
                ))
                continue
            if it.get("type") == "file":
                files.append(FileRef(
                    filename=(it.get("filename") or it.get("path") or "").split("/")[-1],
                    key=it.get("key"),
                    mime=it.get("mime"),
                    size=it.get("size"),
                    description=it.get("description") or "",
                    slot=slot,
                    tool_id=it.get("tool_id"),
                ))
            else:
                val = it.get("value")
                if not isinstance(val, str):
                    try:
                        val = json.dumps(val, ensure_ascii=False)
                    except Exception:
                        val = str(val)
                inlines.append(InlineRef(
                    mime=it.get("mime"),
                    citable=bool(it.get("citable")),
                    description=it.get("description") or "",
                    value_preview=(val or "")[:280],
                    slot=slot,
                    tool_id=it.get("tool_id"),
                ))
        struct_delivs.append(Deliverable(slot=slot, description=desc, files=files, inlines=inlines))

    # include rehosted file metadata if caller passed it (helps UX/debug)
    # rehosted_files: [{slot, key, filename, mime, size, tool_id, description, owner_id, rn}]
    by_slot_extra: Dict[str, List[FileRef]] = {}
    for rf in (rehosted_files or []):
        by_slot_extra.setdefault(rf.get("slot") or "", []).append(FileRef(
            filename=rf.get("filename") or "",
            key=rf.get("key"),
            mime=rf.get("mime"),
            size=rf.get("size"),
            description=rf.get("description") or "",
            slot=rf.get("slot"),
            tool_id=rf.get("tool_id"),
        ))
    if by_slot_extra:
        for d in struct_delivs:
            extras = by_slot_extra.get(d.slot) or []
            # avoid duplicates by filename+key
            seen = {(f.filename, f.key) for f in d.files}
            for e in extras:
                if (e.filename, e.key) not in seen:
                    d.files.append(e)

    brief_struct = ProgramBrief(
        title=title[:120],
        language=language,
        codegen_run_id=codegen.get("run_id"),
        inputs=inputs,
        deliverables=struct_delivs,
        notes=list(latest_round.get("notes") or [] if isinstance(latest_round.get("notes"), list) else
                   ([latest_round.get("notes")] if latest_round.get("notes") else []))
    )

    # ---- render text (compact, deterministic) ----
    lines: List[str] = []
    lines.append(f"# {brief_struct.title}")
    lines.append(f"- Language: {brief_struct.language}")
    if brief_struct.codegen_run_id:
        lines.append(f"- Run: {brief_struct.codegen_run_id}")

    # Inputs
    lines.append("- Objective:" + (f" {inputs.objective}" if inputs.objective else ""))
    lines.append("- Topics:")
    for t in inputs.topics:
        lines.append(f"  - {t}")
    lines.append("- Policy Summary:" + (f" {inputs.policy_summary}" if inputs.policy_summary else ""))
    lines.append("- Constraints:")
    if inputs.constraints:
        for k in sorted(inputs.constraints):
            v = inputs.constraints[k]
            try:
                vv = json.dumps(v, ensure_ascii=False) if not isinstance(v, (str, int, float, bool)) else v
            except Exception:
                vv = str(v)
            lines.append(f"  - {k}: {vv}")
    lines.append("- Tools Selected:")
    for tool in inputs.tools_selected:
        lines.append(f"  - {tool}")

    lines.append("\n## Notes:")
    for note in brief_struct.notes:
        lines.append(f"  - {note}")
    # Contract + deliverables (files emphasized)
    if contract:
        lines.append("\n## Deliverables")
        for d in struct_delivs:
            lines.append(f"- {d.slot}: {d.description}")
            for f in d.files:
                lines.append(f"  - file: {f.filename}" + (f" ({f.mime})" if f.mime else "") + (f" [key:{f.key}]" if f.key else ""))
            if d.inlines:
                lines.append(f"  - inline: {len(d.inlines)} item(s)")

    brief_text = "\n".join(lines).rstrip()
    return brief_text, brief_struct


def _last_non_empty_note(notes) -> str:
    if isinstance(notes, list):
        for s in reversed(notes):
            if isinstance(s, str) and s.strip():
                return s.strip()
    if isinstance(notes, str):
        return notes.strip()
    return ""

def _kv_preview(d: dict, limit: int = 6) -> str:
    if not isinstance(d, dict): return ""
    items = []
    for i, (k, v) in enumerate(d.items()):
        if i >= limit: break
        try:
            vv = json.dumps(v, ensure_ascii=False) if not isinstance(v, (str, int, float, bool)) else v
        except Exception:
            vv = str(v)
        s = str(vv)
        if len(s) > 120:
            s = s[:119] + "…"
        items.append(f"{k}={s}")
    return ", ".join(items)

def _build_program_presentation_for_answer_agent(
        *,
        sr: SolveResult,
        citations: Optional[List[Dict[str, Any]]] = None,
        codegen_run_id: Optional[str] = None,
        include_reasoning: bool = True,
) -> str:
    """
    Compact execution context for the answer agent:
    - OUT items (name/desc/in/out)
    - Deliverables (by slot)
    - Files (rehosted + types)
    - Result interpretation instruction
    - Notes
    - Citations
    """
    lines: List[str] = []
    lines.append("# Program Presentation")
    if codegen_run_id:
        lines.append(f"_Run ID: `{codegen_run_id}`_")

    # Optional reasoning from this round
    if include_reasoning:
        reasoning = sr.round_reasoning()
        if reasoning:
            lines.append("\n## Solver reasoning (for this turn)")
            lines.append(reasoning)

    # OUT items: name/desc/in/out
    out_items = sr.out_items() or []
    if out_items:
        lines.append("\n## Program OUT")
        for it in out_items:
            rid = it.get("resource_id") or ""
            desc = it.get("description") or ""
            inp = it.get("input") or {}
            outp = it.get("output") or {}
            # preview input/output compactly
            in_prev = _kv_preview(inp, 8) if isinstance(inp, dict) else (str(inp)[:160] + ("…" if len(str(inp)) > 160 else ""))
            out_prev = _kv_preview(outp, 8) if isinstance(outp, dict) else (str(outp)[:160] + ("…" if len(str(outp)) > 160 else ""))
            lines.append(f"- `{rid}` — {desc}")
            if in_prev:
                lines.append(f"  - **in:** {in_prev}")
            if out_prev:
                lines.append(f"  - **out:** {out_prev}")

    # Deliverables (by slot)
    dmap = sr.deliverables_map() or {}
    if dmap:
        lines.append("\n## Deliverables")
        for slot, spec in dmap.items():
            desc = (spec or {}).get("description") or ""

            v_raw = (spec or {}).get("value")
            vals = v_raw if isinstance(v_raw, list) else ([v_raw] if isinstance(v_raw, dict) else [])

            # vals = (spec or {}).get("value") or []
            lines.append(f"- **{slot}** — {desc}")
            files_cnt = 0
            inline_cnt = 0
            for v in vals:
                if not isinstance(v, dict):
                    continue
                if v.get("type") == "file":
                    files_cnt += 1
                else:
                    inline_cnt += 1
            if files_cnt:
                lines.append(f"  - files: {files_cnt}")
            if inline_cnt:
                lines.append(f"  - inline: {inline_cnt}")

    # Files (across deliverables)
    all_files: List[Tuple[str, str, str]] = []  # (filename, slot, mime)
    for slot, spec in (dmap or {}).items():
        v_raw = (spec or {}).get("value")
        vals = v_raw if isinstance(v_raw, list) else ([v_raw] if isinstance(v_raw, dict) else [])
        for v in vals:
            if isinstance(v, dict) and v.get("type") == "file":
                name = (v.get("path") or v.get("filename") or "").split("/")[-1]
                mime = v.get("mime") or ""
                all_files.append((name, slot, mime))
    if all_files:
        lines.append("\n## Files")
        for name, slot, mime in all_files[:24]:
            lines.append(f"- {name}{f' ({mime})' if mime else ''} [slot: {slot}]")
        if len(all_files) > 24:
            lines.append(f"- … and {len(all_files)-24} more")

    # How to interpret the results
    rii = sr.interpretation_instruction()
    if rii:
        lines.append("\n## How to interpret these results")
        lines.append(rii.strip())

    # Last round note (if any)
    r0 = (sr.rounds() or [{}])[0]
    last_note = _last_non_empty_note(r0.get("notes"))
    if last_note:
        lines.append("\n## Solver notes")
        lines.append(last_note)

    # Citations
    if citations:
        urls = dict()
        for c in citations:
            u = (c or {}).get("url") or ""
            if u and u not in urls:
                urls[u] = (c or {}).get("title") or ""
        if urls:
            lines.append("\n## Citations")
            for url, title in itertools.islice(urls.items(), 50):
                lines.append(f"- [{title}]({url})")

    return "\n".join(lines).strip()