# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/io_tools.py

import os, json, pathlib, re, mimetypes, inspect
from typing import Annotated, Optional, Any, Dict, List, Tuple

import semantic_kernel as sk

from kdcube_ai_app.apps.chat.sdk.runtime.workdir_discovery import resolve_output_dir
from kdcube_ai_app.apps.chat.sdk.tools.citations import extract_citation_sids_from_text, dedupe_sources_by_url, \
    CITATION_OPTIONAL_ATTRS, normalize_sources_any

try:
    from semantic_kernel.functions import kernel_function
except Exception:
    from semantic_kernel.utils.function_decorator import kernel_function

_CITABLE_TOOL_IDS = {
    "generic_tools.web_search",
    "generic_tools.browsing",
    "ctx_tools.merge_sources",
}

# ---------- basics ----------

_INDEX_FILE = "tool_calls_index.json"

def _outdir() -> pathlib.Path:
    return resolve_output_dir()

def _sanitize_tool_id(tid: str) -> str:
    # "generic_tools.web_search" -> "generic_tools_web_search"
    return re.sub(r"[^a-zA-Z0-9]+", "_", tid).strip("_")

def _guess_mime(path: str, default: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or default

def _extract_candidate_sources_from_tool_output(tool_id: str, tool_output: Any) -> List[Dict[str, Any]]:
    """
    Extract raw source rows from a tool's return, for citable tools only.
    Tries common shapes without guessing content text; URLs are required.
    Returns a list of loose rows to be normalized later.
    """
    rows: List[Dict[str, Any]] = []
    if not _is_citable_tool(tool_id):
        return rows

    return normalize_sources_any(tool_output)

def _canonical_sources_from_citable_tools_generators(promoted: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Build the canonical source list (with stable sids) **only** from citable tools' outputs
    present in `promoted`. Returns (canonical_sources_list, canonical_by_sid_map).
    """
    collected: List[Dict[str, Any]] = []
    for it in promoted or []:
        tool_id = (it.get("tool_id") or "").strip()
        if not _is_citable_tool(tool_id):
            continue
        tool_output = it.get("output")
        # promoted stores raw decoded output in 'output'
        collected.extend(_extract_candidate_sources_from_tool_output(tool_id, tool_output))

    # Deduplicate by URL and assign/keep SIDs
    unified = dedupe_sources_by_url([], collected)
    canonical_list: List[Dict[str, Any]] = [
        {k: v for k, v in row.items() if k in ("sid", "url", "title", "text") or k in CITATION_OPTIONAL_ATTRS}
        for row in unified if isinstance(row.get("sid"), int) and row.get("url")
    ]
    canonical_by_sid = { int(r["sid"]): r for r in canonical_list }
    return canonical_list, canonical_by_sid

# ---------- formats & normalization ----------

def _detect_format_from_value(val: Any, fallback: str = "plain_text") -> str:
    if isinstance(val, (dict, list)):
        return "json"
    if isinstance(val, str):
        # very light heuristic: treat fenced/headers as markdown
        if "\n#" in val or val.strip().startswith("#") or "```" in val:
            return "markdown"
        return "plain_text"
    return fallback

def _coerce_value_and_format(val: Any, fmt: Optional[str]) -> Tuple[Any, Optional[str]]:
    """
    If format is provided (e.g. 'json') and val is a stringified JSON, parse to typed object.
    Else, keep as-is. Return (value, format or None).
    """
    if not fmt:
        return val, None
    f = fmt.strip().lower()
    if f == "json" and isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s), "json"
            except Exception:
                return val, "json"
        return val, "json"
    return val, f

def _stringify_for_format(v: Any, fmt: Optional[str]) -> str:
    """
    Return a string representation of `v` suitable for the given `fmt`.
    - json  → pretty JSON (UTF-8, unicode kept)
    - yaml  → YAML (falls back to JSON if pyyaml missing or dump fails)
    - markdown/html/text/plain → str(v)
    - default: dict/list → pretty JSON, else str(v)
    """
    # bytes → best-effort decode
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return v.decode("latin-1", errors="replace")

    # strings stay as-is
    if isinstance(v, str):
        return v

    f = (fmt or "").strip().lower()

    if f == "json":
        try:
            return json.dumps(v, ensure_ascii=False, indent=2)
        except Exception:
            # last-ditch
            return str(v)

    if f == "yaml":
        try:
            import yaml as _yaml  # local import to avoid hard dep at module import
            return _yaml.safe_dump(v, allow_unicode=True, sort_keys=False)
        except Exception:
            # fall back to JSON pretty
            try:
                return json.dumps(v, ensure_ascii=False, indent=2)
            except Exception:
                return str(v)

    if f in ("markdown", "html", "text", "plain_text", "plaintext"):
        return str(v)

    # default fallback: objects/arrays → pretty JSON, otherwise str()
    try:
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return str(v)


def _normalize_out_dyn(out_dyn: Dict[str, Any], canonical_by_sid: Optional[Dict[int, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Canonicalize dynamic contract dict {slot: VALUE} → list of artifacts for result['out'].

    TARGET FIELDS PER ARTIFACT (for slots):
      - resource_id: "slot:<slot>"
      - type: "inline" | "file"
      - tool_id: "program"
      - output: <inline string/object> | <relative file path>
      - format: optional (markdown|json|plain_text|url|yaml|xml|object)
      - mime: for files only
      - text: for files, the textual surrogate
      - citable: bool (inline URLs default to True)
      - description: str
      - draft: bool (optional; True = incomplete/partial deliverable)  # ✅ NEW
      - input: {}   # reserved, empty for program slots
      - sources_used: list of citation dicts (optional)
      - sources_used_sids: list of int SIDs (optional)
    """
    artifacts: List[Dict[str, Any]] = []

    def _unify_sources(parsed_sids: List[int], explicit_sources: Any) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Merge SIDs parsed from text with SIDs present in explicit sources_used.
        Then fill any missing rows from canonical_by_sid. Return (filled_sources_list, sorted_sids).
        """
        norm_explicit = normalize_sources_any(explicit_sources)
        by_sid: Dict[int, Dict[str, Any]] = {}

        # Seed explicit rows that carry sids
        for row in norm_explicit:
            sid = row.get("sid")
            if isinstance(sid, int) and sid > 0:
                by_sid[sid] = row

        # Union of sids (parsed + explicit-with-sid)
        unified = set(parsed_sids or [])
        unified.update([sid for sid in by_sid.keys() if isinstance(sid, int)])

        # Fill gaps from canonical space only
        if canonical_by_sid:
            for sid in sorted(unified):
                if sid not in by_sid and sid in canonical_by_sid:
                    by_sid[sid] = canonical_by_sid[sid]

        # Keep any explicit rows that have no sid (they do not affect sids list)
        nosid_rows: List[Dict[str, Any]] = [r for r in norm_explicit if not isinstance(r.get("sid"), int)]
        sid_rows = [by_sid[s] for s in sorted([k for k in by_sid.keys() if isinstance(k, int)])]
        filled_list = sid_rows + nosid_rows
        final_sids = [s for s in sorted([k for k in by_sid.keys() if isinstance(k, int)])]
        return filled_list, final_sids

    def push_inline(slot: str, value: Any, *, fmt: Optional[str], desc: str, citable: bool,
                    sources_used: Any = None, draft: bool = False):
        v, use_fmt = _coerce_value_and_format(value, fmt)
        if not use_fmt:
            use_fmt = _detect_format_from_value(v)

        text_str = _stringify_for_format(v, use_fmt)
        parsed_sids = extract_citation_sids_from_text(text_str)
        filled_sources, final_sids = _unify_sources(parsed_sids, sources_used)

        row = {
            "resource_id": f"slot:{slot}",
            "type": "inline",
            "tool_id": "program",
            "output": {"text": text_str},
            "citable": bool(citable),
            "description": desc or "",
            "input": {},
        }

        # Include draft flag if True
        if draft:
            row["draft"] = True

        if use_fmt:
            row["format"] = use_fmt
        if filled_sources:
            row["sources_used"] = filled_sources
        if final_sids:
            row["sources_used_sids"] = final_sids

        artifacts.append(row)

    def push_file(slot: str, relpath: str, *, mime: Optional[str], desc: str, text: str,
                  citable: bool = False, sources_used: Any = None, draft: bool = False):  # ✅ NEW parameter
        parsed_sids = extract_citation_sids_from_text(text or "")
        filled_sources, final_sids = _unify_sources(parsed_sids, sources_used)

        row = {
            "resource_id": f"slot:{slot}",
            "type": "file",
            "tool_id": "program",
            "output": {"path": relpath, "text": text or ""},
            "mime": (mime or _guess_mime(relpath)),
            "citable": False,
            "description": desc or "",
            "input": {},
        }

        # Include draft flag if True
        if draft:
            row["draft"] = True

        if filled_sources:
            row["sources_used"] = filled_sources
        if final_sids:
            row["sources_used_sids"] = final_sids

        artifacts.append(row)

    for slot, val in (out_dyn or {}).items():

        slot_type = val.get("type")
        desc = val.get("description") or val.get("desc") or ""
        citable = bool(val.get("citable", False))
        fmt = val.get("format")
        draft = bool(val.get("draft", False))
        sources_used = val.get("sources_used")

        if slot_type == "file":
            mime = val.get("mime") or None
            text_surrogate = val.get("text") or "" # may be None; program SHOULD have set this
            if sources_used:
                print(f"File Slot {slot}; {sources_used}")
            filepath = val.get("path")
            push_file(slot, filepath, mime=mime, desc=desc, text=text_surrogate, sources_used=sources_used, draft=draft)
            continue
        if slot_type == "inline":
            if "value" in val:
                if sources_used:
                    print(f"Inline Slot {slot}; {sources_used}")
                push_inline(slot, val["value"], fmt=fmt, desc=desc, citable=citable, sources_used=sources_used, draft=draft)
                continue

    return artifacts


# ---------- promotion of saved tool-calls ----------

def _infer_format_for_tool_output(tool_id: str, out: Any) -> Optional[str]:
    if isinstance(out, (dict, list)):
        return "json"
    if isinstance(out, str) and tool_id.endswith("summarize_llm"):
        return "markdown"
    return _detect_format_from_value(out, fallback=None)

def _is_citable_tool(tool_id: str) -> bool:
    tid = (tool_id or "").lower()
    # allow kb_search variants too
    return tid in _CITABLE_TOOL_IDS or tid.endswith(".kb_search")

def _promote_tool_calls(raw_files: Dict[str, List[str]], outdir: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Promote each saved tool-call JSON as ONE artifact:
      - resource_id: "tool:<tool_id>:<index>"
      - path: <relative filename of the saved call JSON>
      - input: params (decoded)
      - output: ret (decoded, object or string)
      - format: inferred from output
      - type: 'inline' for citable web/browse (keeps path), else 'file'
      - citable: True for web/browsing only
      - mime: only for 'file' type (application/json)
      - description: passthrough if present
    """
    promos: List[Dict[str, Any]] = []
    for tool_id, rels in (raw_files or {}).items():
        for idx, rel in enumerate(rels or []):
            p = (outdir / rel)
            desc = ""
            tool_input: Dict[str, Any] = {}
            tool_output: Any = None
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                desc = payload.get("description") or ""
                tool_input = (payload.get("in") or {}).get("params", {}) or {}
                tool_output = payload.get("ret")
            except Exception:
                tool_output = None

            citable = _is_citable_tool(tool_id)
            fmt = _infer_format_for_tool_output(tool_id, tool_output)

            base = {
                "resource_id": f"tool:{tool_id}:{idx}",
                "tool_id": tool_id,
                "path": rel,
                "citable": citable,
                "description": desc,
                "input": tool_input,
                "output": tool_output
            }
            if fmt:
                base["format"] = fmt
            if citable:
                base["type"] = "inline"
            else:
                base["type"] = "file"
                base["mime"] = "application/json"

            # carry sources_used from llm tool outputs
            try:
                if isinstance(tool_output, dict):
                    if tool_id.endswith("generate_content_llm"):
                        us = tool_output.get("sources_used")
                        if us:
                            base["sources_used"] = us
                            base["sources_used"] = us  # convenience alias
                        # Also add SIDs inferred from content (if any)
                        content = tool_output.get("content")
                        if isinstance(content, str):
                            sids = extract_citation_sids_from_text(content)
                            if sids:
                                base["sources_used_sids"] = sids
            except Exception:
                pass

            promos.append(base)
    return promos


# ---------- index helpers (for wrapper) ----------

def _read_index(od: pathlib.Path) -> Dict[str, List[str]]:
    p = od / _INDEX_FILE
    if not p.exists():
        return {}
    try:
        m = json.loads(p.read_text(encoding="utf-8")) or {}
        # normalize shape
        return {k: list(v or []) for k, v in m.items() if isinstance(v, list)}
    except Exception:
        return {}

def _write_index(od: pathlib.Path, m: Dict[str, List[str]]) -> None:
    (od / _INDEX_FILE).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

def _next_index(m: Dict[str, List[str]], tool_id: str) -> int:
    return len(m.get(tool_id, []))


# ---------- AgentIO tools ----------

class AgentIO:
    """
    Writer helpers and infra tools:
      - tool_call(fn, params_json, call_reason, tool_id, ...) → execute + persist one tool call
      - save_ret(data) → writes result.json with normalized out & auto-promoted tool-call artifacts
      - save_tool_call(...) → internal helper (not exposed to LLM) that writes a single call JSON
    """

    # the wrapper the codegen will use
    @kernel_function(
        name="tool_call",
        description=(
                "Execute a tool function and persist the call payload to OUTPUT_DIR as JSON "
                "(indexing the filename per tool). Returns the tool's raw result."
        )
    )
    async def tool_call(
            self,
            fn: Annotated[Any, "Callable tool to invoke (e.g., generic_tools.web_search)."],
            params_json: Annotated[str, "JSON-encoded dict of keyword arguments forwarded to the tool."] = "{}",
            call_reason: Annotated[Optional[str], "Short human reason for the call (5–12 words)."] = None,
            tool_id: Annotated[Optional[str], "Qualified id like 'generic_tools.web_search'. If omitted, derived from fn.__module__+'.'+fn.__name__."] = None,
            filename: Annotated[Optional[str], "Override filename for the saved JSON (relative in OUTPUT_DIR)."] = None,
    ) -> Annotated[Any, "Raw return from the tool"]:
        # parse params
        try:
            params = json.loads(params_json) if isinstance(params_json, str) else dict(params_json or {})
        except Exception:
            params = {"_raw": params_json}

        # execute
        ret = fn(**params) if params else fn()
        if inspect.isawaitable(ret):
            ret = await ret

        # persist
        od = _outdir()
        idx = _read_index(od)

        # derive tool id if not provided
        tid = (tool_id or f"{(getattr(fn, '__module__', 'tool').split('.')[-1])}.{getattr(fn, '__name__', 'call')}").strip()
        next_i = _next_index(idx, tid)
        rel = filename or f"{_sanitize_tool_id(tid)}-{next_i}.json"

        # save the call JSON using the internal helper
        await self.save_tool_call(
            tool_id=tid,
            description=str(call_reason or ""),
            data=ret,                                  # pass RAW output
            params=json.dumps(params, ensure_ascii=False, default=str),
            index=next_i,
            filename=rel,
        )

        # update index
        idx.setdefault(tid, []).append(rel)
        _write_index(od, idx)

        return ret

    # INTERNAL helper (kept public on the instance, but not advertised to LLM)
    async def save_tool_call(
            self,
            tool_id: str,
            description: Optional[str],
            data: Any,
            params: str = "{}",
            index: int = 0,
            filename: Optional[str] = None,
    ) -> str:
        od = _outdir()
        rel = filename or f"{_sanitize_tool_id(tool_id)}-{index}.json"
        path = od / rel

        # decode params
        try:
            p = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception:
            p = {"_raw": params}

        # keep RAW unless it's JSON-looking string
        ret: Any = data
        if isinstance(data, str):
            s = data.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    ret = json.loads(s)
                except Exception:
                    ret = s

        payload = {"description": description or "", "in": {"tool_id": tool_id, "params": p}, "ret": ret}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return rel

    @kernel_function(
        name="save_ret",
        description=(
                "Write the program's result to OUTPUT_DIR (default 'result.json').\n"
                "RESULT SHAPE (authoritative):\n"
                "  - ok: bool (required)\n"
                "  - objective: str (recommended)\n"
                "  - contract: dict(slot-> {description, type})\n"
                "  - out_dyn:  dict(slot->VALUE)\n"
                "  - error: dict with keys error, where, details, managed: optional keys for failures\n"
        )
    )
    async def save_ret(
            self,
            data: Annotated[str, "JSON-encoded object to write."],
            filename: Annotated[str, "Relative filename (defaults to 'result.json')."] = "result.json",
    ) -> Annotated[str, "Saved relative filename"]:
        od = _outdir()
        rel = filename or "result.json"
        path = od / rel

        obj = json.loads(data) if isinstance(data, str) else data

        # 1) Merge tool-call index FIRST so we see all persisted calls
        raw_files = obj.get("raw_files") or {}
        idx_map = _read_index(od)
        if idx_map:
            merged: Dict[str, List[str]] = {k: list(v) for k, v in (raw_files or {}).items()}
            for k, arr in idx_map.items():
                merged.setdefault(k, [])
                for relpath in arr:
                    if relpath not in merged[k]:
                        merged[k].append(relpath)
            raw_files = merged
            obj["raw_files"] = merged  # keep for traceability

        # 2) Promote saved tool calls
        promoted = _promote_tool_calls(raw_files, od)

        # 3) Build canonical source space **only** from citable tools' outputs
        canonical_list, canonical_by_sid = _canonical_sources_from_citable_tools_generators(promoted)
        if canonical_list:
            # persist for downstream turns (helps reconciliation)
            obj["canonical_sources"] = canonical_list

        # 4) Normalize dynamic contract deliverables with access to canonical_by_sid
        out_dyn = obj.get("out_dyn") or {}
        normalized_out = _normalize_out_dyn(out_dyn, canonical_by_sid=canonical_by_sid) if isinstance(out_dyn, dict) else []

        # 5) Merge normalized slots with promoted artifacts (de-dup)
        def _key(a: Dict[str, Any]):
            rid = a.get("resource_id")
            if rid:
                return ("rid", rid)
            return ("fallback", a.get("type"), (a.get("output") or {}).get("text") or a.get("path"))

        seen = set()
        merged_out: List[Dict[str, Any]] = []
        for row in normalized_out + promoted:
            k = _key(row)
            if k in seen:
                continue
            seen.add(k)
            merged_out.append(row)

        obj["out"] = merged_out
        if out_dyn:
            obj["_out_dyn_raw"] = out_dyn

        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return rel


# module-level exports
kernel = sk.Kernel()
tools = AgentIO()
kernel.add_plugin(tools, "agent_io_tools")
