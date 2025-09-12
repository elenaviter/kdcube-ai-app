# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/io_tools.py

import os, json, pathlib, re, mimetypes, inspect
from typing import Annotated, Optional, Any, Dict, List, Tuple

import semantic_kernel as sk

from kdcube_ai_app.apps.chat.sdk.runtime.workdir_discovery import resolve_output_dir

try:
    from semantic_kernel.functions import kernel_function
except Exception:
    from semantic_kernel.utils.function_decorator import kernel_function


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
        if use_fmt:
            row["format"] = use_fmt
        artifacts.append(row)

    def push_file(slot: str, relpath: str, *, mime: Optional[str], desc: str):
        row = {
            "resource_id": f"slot:{slot}",
            "type": "file",
            "tool_id": "program",
            "output": relpath,
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
            push_inline(slot, val, fmt=None, desc="", citable=False)
            continue

        if isinstance(val, dict):
            desc = val.get("description") or val.get("desc") or ""
            citable = bool(val.get("citable", False))
            fmt = val.get("format")

            if "value" in val:
                push_inline(slot, val["value"], fmt=fmt, desc=desc, citable=citable)
                continue

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
                push_inline(slot, val["url"], fmt=fmt or "url", desc=desc, citable=True)
                continue

            file_key = "file" if "file" in val else ("path" if "path" in val else None)
            if file_key and isinstance(val[file_key], str):
                mime = val.get("mime") or None
                push_file(slot, val[file_key], mime=mime, desc=desc)
                continue

        # Fallback
        try:
            as_str = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
        except Exception:
            as_str = str(val)
        push_inline(slot, as_str, fmt=None, desc=desc, citable=False)

    return artifacts


# ---------- promotion of saved tool-calls ----------

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
                tool_output = None

            citable = tool_id in {"generic_tools.web_search", "generic_tools.browsing"}
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
                "  - contract: dict(slot->description)  # echo of the dynamic contract you received\n"
                "  - out_dyn:  dict(slot->VALUE)        # YOU fill this per slot (value+format for inline, file+mime for files, description)"
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

        # 2) auto-discover saved tool calls
        raw_files = obj.get("raw_files") or {}
        # merge in the shared index (written by tool_call)
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

        promoted = _promote_tool_calls(raw_files, od)

        # 3) merge with simple de-duplication
        def _key(a: Dict[str, Any]):
            rid = a.get("resource_id")
            if rid:
                return ("rid", rid)
            return ("fallback", a.get("type"), a.get("output") or a.get("path"))

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
