# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/io_tools.py

import os, json, pathlib, re, mimetypes
from typing import Annotated, Optional, Any, Dict, List, Tuple, Union

import semantic_kernel as sk

from kdcube_ai_app.apps.chat.sdk.runtime.workdir_discovery import resolve_output_dir

try:
    from semantic_kernel.functions import kernel_function
except Exception:
    from semantic_kernel.utils.function_decorator import kernel_function

def _outdir() -> pathlib.Path:
    return resolve_output_dir()

def _sanitize_tool_id(tid: str) -> str:
    # "generic_tools.web_search" -> "generic_tools_web_search"
    return re.sub(r"[^a-zA-Z0-9]+", "_", tid).strip("_")

def _guess_mime(path: str, default: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or default


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


def _normalize_out_dyn(out_dyn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Canonicalize dynamic contract dict {slot: VALUE} → list of artifacts for result['out'].

    TARGET FIELDS PER ARTIFACT (for slots):
      - resource_id: "slot:<slot>"
      - type: "inline" | "file"
      - tool_id: "program"
      - output: <inline string/object> | <relative file path>
      - format: optional (markdown|json|plain_text|url|yaml|xml|object)
      - mime: for files only
      - citable: bool (inline URLs default to True)
      - description: str
      - input: {}   # reserved, empty for program slots
    """
    artifacts: List[Dict[str, Any]] = []

    def push_inline(slot: str, value: Any, *, fmt: Optional[str], desc: str, citable: bool):
        v, use_fmt = _coerce_value_and_format(value, fmt)
        if not use_fmt:
            use_fmt = _detect_format_from_value(v)
        row = {
            "resource_id": f"slot:{slot}",
            "type": "inline",
            "tool_id": "program",
            "output": v,
            "citable": bool(citable),
            "description": desc or "",
            "input": {}
        }
        # include format only when known
        if use_fmt:
            row["format"] = use_fmt
        artifacts.append(row)

    def push_file(slot: str, relpath: str, *, mime: Optional[str], desc: str):
        row = {
            "resource_id": f"slot:{slot}",
            "type": "file",
            "tool_id": "program",
            "output": relpath,                     # <-- normalized key
            "mime": (mime or _guess_mime(relpath)),
            "citable": False,
            "description": desc or "",
            "input": {}
        }
        artifacts.append(row)

    for slot, val in (out_dyn or {}).items():
        desc = ""
        fmt: Optional[str] = None
        citable = False
        mime = None

        # Normalize several accepted shapes
        if isinstance(val, str):
            # treat as inline plain text
            push_inline(slot, val, fmt=None, desc="", citable=False)
            continue

        if isinstance(val, dict):
            # Common metadata pass-through
            desc = val.get("description") or val.get("desc") or ""
            citable = bool(val.get("citable", False))
            fmt = val.get("format")

            # Preferred new schema: value + format
            if "value" in val:
                push_inline(slot, val["value"], fmt=fmt, desc=desc, citable=citable)
                continue

            # Legacy/compat keys for inline:
            if "inline" in val:
                push_inline(slot, val["inline"], fmt=fmt or "plain_text", desc=desc, citable=citable)
                continue
            if "text" in val:
                push_inline(slot, val["text"], fmt=fmt or "plain_text", desc=desc, citable=citable)
                continue
            if "markdown" in val:
                push_inline(slot, val["markdown"], fmt=fmt or "markdown", desc=desc, citable=citable)
                continue
            if "json" in val:
                push_inline(slot, val["json"], fmt=fmt or "json", desc=desc, citable=citable)
                continue
            if "yaml" in val:
                push_inline(slot, val["yaml"], fmt=fmt or "yaml", desc=desc, citable=citable)
                continue
            if "xml" in val:
                push_inline(slot, val["xml"], fmt=fmt or "xml", desc=desc, citable=citable)
                continue
            if "url" in val and isinstance(val["url"], str):
                # represent as inline URL, citable
                push_inline(slot, val["url"], fmt=fmt or "url", desc=desc, citable=True)
                continue

            # File shape (either 'file' or 'path')
            file_key = "file" if "file" in val else ("path" if "path" in val else None)
            if file_key and isinstance(val[file_key], str):
                mime = val.get("mime") or None
                push_file(slot, val[file_key], mime=mime, desc=desc)
                continue

        # Fallback: stringify into inline
        try:
            as_str = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
        except Exception:
            as_str = str(val)
        push_inline(slot, as_str, fmt=None, desc=desc, citable=False)

    return artifacts


def _infer_format_for_tool_output(tool_id: str, out: Any) -> Optional[str]:
    if isinstance(out, (dict, list)):
        return "json"
    if isinstance(out, str) and tool_id.endswith("summarize_llm"):
        return "markdown"
    return _detect_format_from_value(out, fallback=None)


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
                # keep as None if unreadable
                tool_output = None

            citable = tool_id in {"generic_tools.web_search", "generic_tools.browsing"}
            fmt = _infer_format_for_tool_output(tool_id, tool_output)

            base = {
                "resource_id": f"tool:{tool_id}:{idx}",
                "tool_id": tool_id,
                "path": rel,                     # JSON file we wrote with save_tool_call
                "citable": citable,
                "description": desc,
                "input": tool_input,
                "output": tool_output
            }
            if fmt:
                base["format"] = fmt

            if citable:
                base["type"] = "inline"
                # no mime for inline
            else:
                base["type"] = "file"
                base["mime"] = "application/json"

            promos.append(base)
    return promos

class AgentIO:
    """
    Writer helpers for generated programs:
      - save_tool_call(tool_id, data, params, ...) → writes {in:{...}, ret:...}.json
      - save_ret(data) → writes result.json with normalized out & auto-promoted tool-call artifacts
    """

    @kernel_function(
        name="save_tool_call",
        description="Persist a tool call payload to OUTPUT_DIR as JSON: {in:{tool_id,params}, ret:...}."
    )
    async def save_tool_call(
        self,
        tool_id: Annotated[str, "Qualified id, e.g. 'generic_tools.web_search'."],
        description: Annotated[Optional[str], "Short description of why this call needed and what it produces."],
        data: Annotated[str, "Raw return; Pass as is"],
        params: Annotated[str, "JSON-encoded dict of parameters used for the call."] = "{}",
        index: Annotated[int, "Monotonic index per tool, starting at 0."] = 0,
        filename: Annotated[Optional[str], "Override filename (relative in OUTPUT_DIR)."] = None,
    ) -> Annotated[str, "Saved relative filename"]:
        od = _outdir()
        rel = filename or f"{_sanitize_tool_id(tool_id)}-{index}.json"
        path = od / rel

        # decode params
        try:
            p = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception:
            p = {"_raw": params}

        # decode data if JSON-looking; else keep string
        ret: Any = data
        if isinstance(data, str):
            s = data.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    ret = json.loads(s)
                except Exception:
                    ret = s

        payload = {"description": description, "in": {"tool_id": tool_id, "params": p}, "ret": ret}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return rel

    @kernel_function(
        name="save_ret",
        description=(
                "Write the program's result to OUTPUT_DIR (default 'result.json').\n"
                "RESULT SHAPE (authoritative):\n"
                "  - ok: bool (required)\n"
                "  - objective: str (recommended)\n"
                "  - contract: dict(slot->description)  # echo of the dynamic contract you received\n"
                "  - out_dyn:  dict(slot->VALUE)        # YOU fill this per slot (value+format for inline, file+mime for files, description, )\n"
                "  - queries_used?: [str]\n"
                "  - raw_files?: { adapter_id: [saved_json_filename, ...] }\n"
                "Do NOT set result['out']; this method derives it from 'out_dyn' and promotes tool call files as single artifacts."
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

        # 1) normalize contract outputs (slots)
        out_dyn = obj.get("out_dyn") or {}
        normalized_out = _normalize_out_dyn(out_dyn) if isinstance(out_dyn, dict) else []

        # 2) auto-promote saved tool calls from raw_files
        raw_files = obj.get("raw_files") or {}
        promoted = _promote_tool_calls(raw_files, od)

        # 3) merge with simple de-duplication
        def _key(a: Dict[str, Any]):
            # Prefer stable identity; fall back to (type, output/path)
            rid = a.get("resource_id")
            if rid:
                return ("rid", rid)
            # the slot/file case uses "output" (path) and tool-call has "path"
            return ("fallback", a.get("type"), a.get("output") or a.get("path"))

        seen = set()
        merged: List[Dict[str, Any]] = []
        for row in normalized_out + promoted:
            k = _key(row)
            if k in seen: continue
            seen.add(k)
            merged.append(row)

        obj["out"] = merged  # canonical list used by downstream
        # keep original for traceability
        if out_dyn: obj["_out_dyn_raw"] = out_dyn

        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return rel

kernel = sk.Kernel()
tools = AgentIO()
kernel.add_plugin(tools, "agent_io_tools")