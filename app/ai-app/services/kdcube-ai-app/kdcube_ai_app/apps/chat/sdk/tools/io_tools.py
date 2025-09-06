# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/io_tools.py

import os, json, pathlib, re, mimetypes
from typing import Annotated, Optional, Any, Dict, List

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

def _normalize_out_dyn(out_dyn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a dict of {slot_name: value} into normalized artifact rows:
      {resource_id, type, tool_id, path|value, mime?, citable?, description?, tool_input?}
    Accepted VALUE shapes:
      - string → inline (we don't force markdown vs text; keep as 'value' string)
      - {"text"| "markdown"| "json"| "url": <str>} → inline; url uses mime 'text/x-url'
      - {"file" | "path": <str>, "mime"?: str} → file; mime guessed if absent
      - common optional keys: description, citable, tool_id, tool_input
    """
    artifacts: List[Dict[str, Any]] = []
    for slot, val in (out_dyn or {}).items():
        base_id = str(slot)
        def push_inline(v: str, as_url: bool = False, meta: Dict[str, Any] = None):
            meta = meta or {}
            artifacts.append({
                "resource_id": base_id,
                "type": "inline",
                "tool_id": meta.get("tool_id", "program"),
                "value": v,
                "mime": "text/x-url" if as_url else meta.get("mime") or None,
                "citable": bool(meta.get("citable", as_url)),
                "description": meta.get("description") or "",
                "tool_input": meta.get("tool_input") or {},
            })

        def push_file(p: str, meta: Dict[str, Any] = None):
            meta = meta or {}
            artifacts.append({
                "resource_id": base_id,
                "type": "file",
                "tool_id": meta.get("tool_id", "program"),
                "path": p,
                "mime": meta.get("mime") or _guess_mime(p),
                "citable": bool(meta.get("citable", False)),
                "description": meta.get("description") or "",
                "tool_input": meta.get("tool_input") or {},
            })

        if isinstance(val, str):
            push_inline(val)
            continue

        if isinstance(val, dict):
            meta = {k: v for k, v in val.items() if k not in ("text","markdown","json","url","file","path")}
            if "url" in val and isinstance(val["url"], str):
                push_inline(val["url"], as_url=True, meta=meta); continue
            if "text" in val and isinstance(val["text"], str):
                push_inline(val["text"], as_url=False, meta=meta); continue
            if "markdown" in val and isinstance(val["markdown"], str):
                push_inline(val["markdown"], as_url=False, meta=meta); continue
            if "json" in val:
                payload = val["json"]
                push_inline(json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload, as_url=False, meta=meta); continue
            file_key = "file" if "file" in val else ("path" if "path" in val else None)
            if file_key and isinstance(val[file_key], str):
                if "mime" in val: meta["mime"] = val["mime"]
                push_file(val[file_key], meta=meta); continue

        # Fallback: best-effort stringification
        push_inline(json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val)
    return artifacts

def _auto_promote_citations(raw_files: Dict[str, List[str]], outdir: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Heuristics: if a saved tool payload looks like web search results ({ret: [ {title,url,body}, ...]}),
    emit inline URL artifacts (citable=true).
    """
    promoted: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for tool_id, rels in (raw_files or {}).items():
        for i, rel in enumerate(rels or []):
            p = (outdir / rel)
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            ret = payload.get("ret")
            if isinstance(ret, list):
                for j, row in enumerate(ret):
                    if not isinstance(row, dict): continue
                    url = row.get("url") or row.get("href")
                    title = row.get("title", "")
                    if not url or url in seen_urls: continue
                    seen_urls.add(url)
                    promoted.append({
                        "resource_id": f"src:{len(seen_urls)}",
                        "type": "inline",
                        "tool_id": tool_id,
                        "value": url,
                        "mime": "text/x-url",
                        "citable": True,
                        "description": title,
                        "tool_input": payload.get("in", {}).get("params", {}),
                    })
    return promoted

class AgentIO:
    """
    File I/O helpers for generated programs:
      - save_tool_output: persist {"in": {tool_id, params}, "ret": <parsed>} as JSON
      - save_ret: write any JSON-serializable object to a filename (default: result.json)
    """

    @kernel_function(
        name="save_tool_output",
        description="Persist a tool call payload to disk as JSON with shape {in:{tool_id, params}, ret:...}."
    )
    async def save_tool_output(
        self,
        tool_id: Annotated[str, "Qualified id, e.g. 'generic_tools.web_search'."],
        data: Annotated[str, "Raw return; pass string or JSON-encoded string."],
        params: Annotated[str, "JSON-encoded dict of parameters used for the call."] = "{}",
        index: Annotated[int, "Monotonic index per tool, starting at 0."] = 0,
        filename: Annotated[Optional[str], "Override filename (relative in OUTPUT_DIR)."] = None,
    ) -> Annotated[str, "Saved relative filename"]:
        od = _outdir()
        # choose filename
        rel = filename or f"{_sanitize_tool_id(tool_id)}-{index}.json"
        path = od / rel

        # parse params if possible
        try:
            p = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception:
            p = {"_raw": params}

        # decode data if JSON-looking; else keep as string
        ret: Any = data
        if isinstance(data, str):
            s = data.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    ret = json.loads(s)
                except Exception:
                    ret = s

        payload = {"in": {"tool_id": tool_id, "params": p}, "ret": ret}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return rel

    @kernel_function(
        name="save_ret",
        description=(
                "Write the program's result JSON to OUTPUT_DIR (default 'result.json').\n"
                "RESULT SHAPE (authoritative):\n"
                "  - ok: bool (required)\n"
                "  - objective: str (recommended)\n"
                "  - contract: dict(slot->description)  # echo of the dynamic contract you received\n"
                "  - out_dyn:  dict(slot->VALUE)        # YOU fill this per slot; infra derives result.out from it\n"
                "  - queries_used?: [str]\n"
                "  - raw_files?: { adapter_id: [saved_json_filename, ...] }\n"
                "VALUE forms:\n"
                "  * text/markdown/json: {'text'|'markdown'|'json': ...}\n"
                "  * file/path: {'file'|'path': 'relative/path.ext', 'mime'?: '...'}\n"
                "Optional keys in VALUE: description, citable, tool_id, tool_input, mime.\n"
                "IMPORTANT: Do NOT write result['out'] yourself. This method will normalize out_dyn into result['out']\n"
                "and auto-promote citable web URLs discovered in saved tool outputs."
        )
    )
    async def save_ret(
        self,
        data: Annotated[str, "JSON-encoded object to write."],
        filename: Annotated[str, "Relative filename to write (defaults to 'result.json')."] = "result.json",
    ) -> Annotated[str, "Saved relative filename"]:
        od = _outdir()
        rel = filename or "result.json"
        path = od / rel

        obj = json.loads(data) if isinstance(data, str) else data
        # normalize dynamic output if present
        out_dyn = obj.get("out_dyn") or {}
        normalized_out = _normalize_out_dyn(out_dyn) if isinstance(out_dyn, dict) else []

        # auto-promote citable items from raw_files
        raw_files = obj.get("raw_files") or {}
        promoted = _auto_promote_citations(raw_files, od)

        # merge (contract outputs first, then promotions), dedupe by (type,value/path)
        def _key(a: Dict[str, Any]):
            if a.get("type") == "file": return ("file", a.get("path"))
            return ("inline", a.get("value"))
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