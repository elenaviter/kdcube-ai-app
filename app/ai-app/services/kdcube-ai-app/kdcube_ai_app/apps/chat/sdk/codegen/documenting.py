# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/codegen/documenting.py

from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
from pydantic import BaseModel, Field
import json, itertools

from kdcube_ai_app.apps.chat.sdk.codegen.solve_result import SolveResult


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

def _program_brief_from_contract(cg: dict, rehosted_files: List[dict]) -> Tuple[str, ProgramBrief]:
    """
    Returns:
      (brief_text, ProgramBrief)
    """
    codegen = (cg or {}).get("codegen") or {}
    contract = (cg or {}).get("contract_dyn") or {}
    deliverables = (cg or {}).get("deliverables") or {}

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
        vals = (dv.get("value") or []) if isinstance(dv, dict) else []
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
            vals = (spec or {}).get("value") or []
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
        for v in (spec or {}).get("value") or []:
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