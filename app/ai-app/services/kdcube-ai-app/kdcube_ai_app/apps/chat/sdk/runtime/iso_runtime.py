# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter
import contextvars
# chat/sdk/runtime/iso_runtime.py
import io
import json
import os, sys, re
import asyncio
import pathlib
import runpy
import tokenize
import traceback
from typing import Dict, Any, List, Tuple
from textwrap import dedent

from kdcube_ai_app.infra.service_hub.inventory import AgentLogger

def _run_in_executor_with_ctx(loop, fn, *args, **kwargs):
    ctx = contextvars.copy_context()
    return loop.run_in_executor(None, lambda: ctx.run(fn, *args, **kwargs))

def build_current_tool_imports(alias_map: dict[str, str]) -> str:
    """
    alias_map: {"io_tools":"io_tools","ctx_tools":"ctx_tools","generic_tools":"generic_tools","llm_tools":"llm_tools", ...}
    Returns a Markdown code block with python import lines.
    """
    # Always show the wrapper import first
    lines = ["from io_tools import tools as agent_io_tools  # wrapper for tool_call/save_ret"]
    seen = set()

    for alias, module in (alias_map or {}).items():
        if not alias or not module:
            continue
        # Avoid duplicating the wrapper line
        line = f"from {module} import tools as {alias}"
        if line not in seen:
            lines.append(line)
            seen.add(line)

    return "```python\n" + "\n".join(lines) + "\n```"

def build_packages_installed_block() -> str:
    return (
        "## Available Packages\n"
        "- Data: pandas, numpy\n"
        "- Files: python-docx, python-pptx, pymupdf, pypdf, reportlab, Pillow\n"
        "- Web: requests, aiohttp, playwright, beautifulsoup4, lxml\n"
        "- Viz: matplotlib, seaborn, plotly, networkx, graphviz, diagrams\n"
        "- Text: markdown-it-py, pygments, jinja2\n"
        "- Utils: pydantic, orjson, python-dotenv, PyJWT\n"
    )

def _merge_timeout_result(path: pathlib.Path, *, objective: str, seconds: int):
    reason = f"Solver runtime exceeded {seconds}s and was terminated."
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        # preserve existing fields, only enforce error shape and ok=False
        data["ok"] = False
        err = (data.get("error") or {})
        err.update({
            "where": "runtime",
            "error": "timeout",
            "description": reason,
            "managed": True,
        })
        data["error"] = err
        if not data.get("objective"):
            data["objective"] = objective
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # no prior file — write minimal
    payload = {
        "ok": False,
        "objective": objective,
        "out": [],
        "error": {
            "where": "runtime",
            "error": "timeout",
            "description": reason,
            "managed": True
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _inject_header_after_future(src: str, header: str) -> str:
    lines = src.splitlines(True)
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("from __future__ import"):
        i += 1
    # idempotent
    if header.strip() in src:
        return src
    return "".join(lines[:i] + [header] + lines[i:])

def _fix_json_bools(src: str) -> str:
    """
    Replace NAME tokens 'true'/'false'/'null' with Python's True/False/None,
    preserving all original spacing and positions, and skipping strings/comments.
    Includes a fast path that returns src unchanged if none of the literals occur.
    """
    # Fast path: avoid tokenization if none of the JSON literals appear
    if ("true" not in src) and ("false" not in src) and ("null" not in src):
        return src

    mapping = {"true": "True", "false": "False", "null": "None"}
    out = []

    # Use TokenInfo objects to preserve exact spacing via start/end positions
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in mapping:
            tok = tok._replace(string=mapping[tok.string])
        out.append(tok)

    return tokenize.untokenize(out)

def _max_sid_from_context(outdir: pathlib.Path) -> int:
    try:
        p = outdir / "context.json"
        if not p.exists():
            return 0
        import json as _json
        data = _json.loads(p.read_text(encoding="utf-8"))
        ph = data.get("program_history") or []
        mx = 0
        for entry in ph:
            if not isinstance(entry, dict):
                continue
            rec = next(iter(entry.values()), {})  # {"<exec_id>": {...}}
            items = (((rec or {}).get("web_links_citations") or {}).get("items") or [])
            for it in items:
                try:
                    mx = max(mx, int(it.get("sid") or 0))
                except Exception:
                    pass
        return mx
    except Exception:
        return 0

def _module_parent_dirs(tool_modules: List[Tuple[str, object]]) -> List[str]:
    paths = []
    for name, mod in tool_modules or []:
        try:
            p = pathlib.Path(getattr(mod, "__file__", "")).resolve()
            if p.exists():
                paths.append(str(p.parent))
        except Exception:
            pass
    # de-dup while preserving order
    seen = set(); uniq = []
    for d in paths:
        if d not in seen:
            uniq.append(d); seen.add(d)
    return uniq

async def _run_subprocess(entry_path: pathlib.Path, *, cwd: pathlib.Path, env: dict, timeout_s: int, outdir: pathlib.Path):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(entry_path),
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        return {"error": "timeout", "seconds": timeout_s}
    finally:
        # save captured logs
        try:
            out, err = await proc.communicate()
        except Exception:
            out, err = (b"", b"")
        try:
            (outdir / "runtime.out.log").write_bytes(out or b"")
            (outdir / "runtime.err.log").write_bytes(err or b"")
        except Exception:
            pass

    return {"ok": proc.returncode == 0, "returncode": proc.returncode}

class _InProcessRuntime:
    def __init__(self, logger: AgentLogger):
        self.log = logger or AgentLogger("tool_runtime")

    def _ensure_modules_on_sys_modules(self, modules: List[Tuple[str, object]]):
        """Make sure codegen can 'from <name> import tools as <alias>' for each module."""
        for name, mod in modules or []:
            if name and name not in sys.modules:
                sys.modules[name] = mod

    async def run_snippet(
        self,
        *,
        code: str,
        output_dir: pathlib.Path,
        tool_modules: List[Tuple[str, object]],
        globals: Dict[str, Any] = None,
        timeout_s: int = 90,
    ) -> Dict[str, Any]:
        import tempfile, json

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect tool module names for child bootstrap + shutdown
        tool_module_names = [name for name, _ in (tool_modules or []) if name]
        # Always consider KB client module as a shutdown candidate
        shutdown_candidates = list(tool_module_names) + [
            "kdcube_ai_app.apps.chat.sdk.retrieval.kb_client"
        ]

        # Prepare GLOBALS block
        globals_src = ""
        if globals:
            for k, v in globals.items():
                if k and (k != "__name__"):
                    globals_src += f"\n{k} = {repr(v)}\n"

        injected_header = dedent('''\
# === AGENT-RUNTIME HEADER (auto-injected, do not edit) ===
from pathlib import Path
import json as _json
import os, importlib, asyncio, atexit, signal
from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV
from io_tools import tools as agent_io_tools

# --- Directories / CV fallbacks ---
OUTPUT_DIR = OUTDIR_CV.get() or os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR missing in run context")
OUTPUT = Path(OUTPUT_DIR)

# --- Portable spec handoff (context vars + model service + registry + communicator) ---
# Accept either ENV ("PORTABLE_SPEC") or a global literal ("PORTABLE_SPEC_JSON") the parent injected.
_PORTABLE_SPEC = os.environ.get("PORTABLE_SPEC")
if not _PORTABLE_SPEC:
    try:
        _PORTABLE_SPEC = globals().get("PORTABLE_SPEC_JSON") or None
    except Exception:
        _PORTABLE_SPEC = None

_TOOL_MODULES = _json.loads(os.environ.get("RUNTIME_TOOL_MODULES") or "[]")
_SHUTDOWN_MODULES = _json.loads(os.environ.get("RUNTIME_SHUTDOWN_MODULES") or "[]")

def _bootstrap_child():
    if not _PORTABLE_SPEC:
        return
    try:
        from kdcube_ai_app.apps.chat.sdk.runtime.bootstrap import bootstrap_from_spec
        for mod_name in (_TOOL_MODULES or []):
            try:
                m = importlib.import_module(mod_name)
            except Exception:
                continue
            # Bind ModelServiceBase + registry + integrations into each tool module
            bootstrap_from_spec(_PORTABLE_SPEC, tool_module=m)
    except Exception:
        # non-fatal
        pass

def _expose_tool_aliases():
    """
    Make 'ctx_tools', 'io_tools', 'llm_tools', etc. visible even if the program forgot to import them.
    We map the last dotted segment of each module to its .tools object (if present).
    """
    for mod_name in (_TOOL_MODULES or []):
        try:
            m = importlib.import_module(mod_name)
            alias = mod_name.rsplit(".", 1)[-1]
            if alias and hasattr(m, "tools") and alias not in globals():
                globals()[alias] = getattr(m, "tools")
        except Exception:
            # best effort only
            pass

_bootstrap_child()
_expose_tool_aliases()

<GLOBALS_SRC>

# -------- Live progress cache (safe, in-process) --------
_PROGRESS = {
    "objective": "",
    "status": "In progress",
    "story": [],          # list[str]
    "out_dyn": {},        # slot_name -> slot dict (inline/file)
}

_FINALIZED = False  # prevents late checkpoints from overwriting the final result


def set_progress(*, objective=None, status=None, story_append=None, out_dyn_patch=None, flush=False):
    if objective is not None:
        _PROGRESS["objective"] = str(objective)
    if status is not None:
        _PROGRESS["status"] = str(status)
    if story_append:
        if isinstance(story_append, (list, tuple)):
            _PROGRESS["story"].extend([str(s) for s in story_append])
        else:
            _PROGRESS["story"].append(str(story_append))
    if out_dyn_patch:
        od = _PROGRESS["out_dyn"]
        for k, v in (out_dyn_patch or {}).items():
            od[k] = v
    if flush:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_write_checkpoint(reason="progress", managed=True))
            else:
                loop.run_until_complete(_write_checkpoint(reason="progress", managed=True))
        except Exception:
            pass

def _build_project_log_md(objective: str, status: str) -> str:
    slots_md = ""
    for name, data in (_PROGRESS.get("out_dyn") or {}).items():
        if name == "project_log":
            continue
        t = (data.get("type") or "inline")
        desc = data.get("description", "")
        slots_md += f"\n### {name} ({t})\n{desc}\n"
        if t == "file":
            slots_md += f"**Mime:** {data.get('mime','')}\n**Filename:** {data.get('path','')}\n"
        else:
            fmt = data.get("format","")
            if fmt:
                slots_md += f"**Format:** {fmt}\n"
    return (
        f"# Project Log\n\n"
        f"## Objective\n{objective}\n\n"
        f"## Status\n{status}\n\n"
        f"## Story\n{' '.join(_PROGRESS.get('story') or [])}\n\n"
        f"## Produced Slots\n{slots_md}"
    ).strip()

async def _write_checkpoint(reason: str = "checkpoint", managed: bool = True):
    try:
        g = globals()
        contract = g.get("CONTRACT", {}) or {}
        objective = (_PROGRESS.get("objective") or g.get("objective") or g.get("OBJECTIVE") or "")
        status = _PROGRESS.get("status") or "In progress"
        project_log_md = _build_project_log_md(objective, status)
        out_dyn = dict(_PROGRESS.get("out_dyn") or {})
        out_dyn["project_log"] = {"type": "inline", "format": "markdown", "description": "Live run log", "value": project_log_md}
        payload = {
            "ok": False,
            "objective": str(objective),
            "contract": contract,
            "out_dyn": out_dyn,
            "error": {"where": "runtime", "details": "", "error": reason, "description": reason, "managed": bool(managed)}
        }
        await agent_io_tools.save_ret(data=_json.dumps(payload), filename="result.json")
    except Exception:
        pass

async def done():
    g = globals()
    contract = g.get("CONTRACT", {}) or {}
    objective = (_PROGRESS.get("objective") or g.get("objective") or g.get("OBJECTIVE") or "")
    status = _PROGRESS.get("status") or "Completed"
    if status.lower().startswith("in progress"): status = "Completed"
    project_log_md = _build_project_log_md(objective, status)
    out_dyn = dict(_PROGRESS.get("out_dyn") or {})
    out_dyn["project_log"] = {"type": "inline", "format": "markdown", "description": "Live run log", "value": project_log_md}
    payload = {"ok": True, "objective": str(objective), "contract": contract, "out_dyn": out_dyn}
    return await agent_io_tools.save_ret(data=_json.dumps(payload), filename="result.json")

def _on_term(signum, frame):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_write_checkpoint(reason=f"signal:{signum}", managed=True))
        else:
            loop.run_until_complete(_write_checkpoint(reason=f"signal:{signum}", managed=True))
    except Exception:
        pass

# --- Module shutdown on exit (KB, tool modules etc.) ---
async def _async_shutdown_mod(mod):
    try:
        if hasattr(mod, "shutdown") and callable(mod.shutdown):
            maybe = mod.shutdown()
            if asyncio.iscoroutine(maybe):
                await maybe
        elif hasattr(mod, "close") and callable(mod.close):
            maybe = mod.close()
            if asyncio.iscoroutine(maybe):
                await maybe
    except Exception:
        pass

def _sync_shutdown_all():
    try:
        import importlib
        mods = []
        for name in set(_SHUTDOWN_MODULES or []):
            try:
                mods.append(importlib.import_module(name))
            except Exception:
                pass
        async def _run():
            for m in mods:
                await _async_shutdown_mod(m)
        asyncio.run(_run())
    except Exception:
        pass

try:
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
except Exception:
    pass
atexit.register(lambda: asyncio.run(_write_checkpoint(reason="atexit", managed=True)))
atexit.register(_sync_shutdown_all)

async def fail(description: str,
               where: str = "",
               error: str = "",
               details: str = "",
               managed: bool = True,
               out_dyn: dict | None = None):
    """
    Managed failure helper. Always writes result.json with a normalized envelope.
    Also **injects project_log** so downstream always sees the live log on errors.
    """
    try:
        g = globals()
        contract = g.get("CONTRACT", {}) or {}
        objective = (g.get("objective") or g.get("OBJECTIVE")
                     or _PROGRESS.get("objective") or "")
        status = _PROGRESS.get("status") or "Failed"
        project_log_md = _build_project_log_md(objective, status)
    except Exception:
        contract, objective = {}, ""
        project_log_md = ""
    out = dict(out_dyn or _PROGRESS.get("out_dyn") or {})
    out["project_log"] = {"type": "inline", "format": "markdown", "description": "Live run log", "value": project_log_md}
    payload = {
        "ok": False,
        "objective": str(objective or description),
        "contract": contract,
        "out_dyn": out,
        "error": {
            "where": (where or "runtime"),
            "details": str(details or ""),
            "error": str(error or ""),
            "description": description,
            "managed": bool(managed),
        }
    }
    return await agent_io_tools.save_ret(data=_json.dumps(payload), filename="result.json")    
# === END HEADER ===
# === CHAT COMMUNICATOR RECONSTRUCTION ===
from kdcube_ai_app.apps.chat.sdk.protocol import (
    ChatEnvelope, ServiceCtx, ConversationCtx
)
from kdcube_ai_app.apps.chat.emitters import (ChatRelayCommunicator, ChatCommunicator, _RelayEmitterAdapter)
from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import set_comm

def _rebuild_communicator_from_spec(spec: dict) -> ChatCommunicator:
    redis_url = (spec or {}).get("redis_url") or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    channel   = (spec or {}).get("channel")   or "chat.events"
    relay = ChatRelayCommunicator(redis_url=redis_url, channel=channel)
    emitter = _RelayEmitterAdapter(relay)
    svc = ServiceCtx(**(spec.get("service") or {}))
    conv = ConversationCtx(**(spec.get("conversation") or {}))
    comm = ChatCommunicator(
        emitter=emitter,
        service=svc.model_dump() if hasattr(svc, "model_dump") else dict(svc),
        conversation=conv.model_dump() if hasattr(conv, "model_dump") else dict(conv),
        room=spec.get("room"),
        target_sid=spec.get("target_sid"),
    )
    return comm

try:
    _COMM_SPEC = globals().get("COMM_SPEC") or {}
    if _COMM_SPEC:
        _comm_obj = _rebuild_communicator_from_spec(_COMM_SPEC)
        set_comm(_comm_obj)
except Exception as _comm_err:
    pass
# === END COMMUNICATOR SETUP ===
''').replace('<GLOBALS_SRC>', globals_src)

        # Prepare snippet source
        src = _fix_json_bools(code)
        src = _inject_header_after_future(src, injected_header)

        # Temp workdir for snippet
        import tempfile
        workdir = pathlib.Path(tempfile.mkdtemp(prefix="cg_snip_"))
        (workdir / "main.py").write_text(src, encoding="utf-8")

        # Child env
        child_env = os.environ.copy()
        child_env["OUTPUT_DIR"] = str(output_dir)
        # passthrough portable spec if provided in globals
        ps = (globals or {}).get("PORTABLE_SPEC_JSON") or (globals or {}).get("PORTABLE_SPEC")
        if ps:
            child_env["PORTABLE_SPEC"] = ps if isinstance(ps, str) else json.dumps(ps, ensure_ascii=False)
        import json as _json
        child_env["RUNTIME_TOOL_MODULES"] = _json.dumps(tool_module_names)
        child_env["RUNTIME_SHUTDOWN_MODULES"] = _json.dumps(shutdown_candidates)

        # Augment PYTHONPATH with parents of tool modules
        parents = _module_parent_dirs(tool_modules)
        if parents:
            child_env["PYTHONPATH"] = os.pathsep.join(parents + [child_env.get("PYTHONPATH", "")])

        # Run as subprocess (+ logs captured)
        return await _run_subprocess(
            entry_path=workdir / "main.py",
            cwd=workdir,
            env=child_env,
            timeout_s=timeout_s,
            outdir=output_dir,
        )

    async def run_main_py(
        self,
        *,
        workdir: pathlib.Path,
        output_dir: pathlib.Path,
        tool_modules: List[Tuple[str, object]],
        globals: Dict[str, Any] = None,
        timeout_s: int = 90,
    ) -> Dict[str, Any]:
        workdir.mkdir(parents=True, exist_ok=True)

        from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV, WORKDIR_CV, SOURCE_ID_CV
        import json as _json
        def _runner():
            old_env = dict(os.environ)
            old_path = list(sys.path)

            t_out = OUTDIR_CV.set(str(output_dir))
            t_wrk = WORKDIR_CV.set(str(workdir))

            last_sid = _max_sid_from_context(output_dir)
            t_sid = SOURCE_ID_CV.set({"next": int(last_sid) + 1})

            try:
                # Ensure the workdir is importable and tool modules resolvable
                sys.path.insert(0, str(workdir))
                self._ensure_modules_on_sys_modules(tool_modules)

                # --- Prepare child bootstrap data in ENV ---
                # 1) PortableSpec (if caller provided as GLOBAL, prefer that; else allow env to be already set)
                if globals and "PORTABLE_SPEC" in globals and isinstance(globals["PORTABLE_SPEC"], str):
                    os.environ["PORTABLE_SPEC"] = globals["PORTABLE_SPEC"]
                    del globals["PORTABLE_SPEC"]

                # 2) Tool module names (for service binding in child)
                tool_module_names = [name for name, _ in (tool_modules or []) if name]
                os.environ["RUNTIME_TOOL_MODULES"] = _json.dumps(tool_module_names)

                # 3) Modules that should receive shutdown() / close() in child atexit
                shutdown_module_names = list(tool_module_names)
                # include KB client if present; it's safe if import fails in child
                shutdown_module_names.append("kdcube_ai_app.apps.chat.sdk.retrieval.kb_client")
                os.environ["RUNTIME_SHUTDOWN_MODULES"] = _json.dumps(shutdown_module_names)

                # 4) OUTPUT_DIR redundantly (header also reads via OUTDIR_CV)
                os.environ["OUTPUT_DIR"] = str(output_dir)

                # --- Read and transform main.py ---
                src = (workdir / "main.py").read_text(encoding="utf-8")

                # 1) Fix JSON booleans/null FIRST (no exceptions here)
                src = _fix_json_bools(src)

                # 2) Build globals prelude (idempotent, simple assignments)
                globals_src = ""
                if globals:
                    for k, v in globals.items():
                        if k and (k != "__name__"):
                            globals_src += f"\n{k} = {repr(v)}\n"

                imports_src = ""
                alias_map = (globals or {}).get("TOOL_ALIAS_MAP") or {}
                for alias, mod_name in (alias_map or {}).items():
                    # io_tools already has a special import as agent_io_tools; we can still expose alias too
                    imports_src += f"\nfrom {mod_name} import tools as {alias}\n"

                # 3) Inject our runtime header after any __future__
                injected_header = (dedent('''\
# === AGENT-RUNTIME HEADER (auto-injected, do not edit) ===
from pathlib import Path
import json as _json
import os, importlib, asyncio, atexit, signal
from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV
from io_tools import tools as agent_io_tools

# --- Directories / CV fallbacks ---
OUTPUT_DIR = OUTDIR_CV.get() or os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR missing in run context")
OUTPUT = Path(OUTPUT_DIR)

# --- Portable spec handoff (context vars + model service + registry + communicator) ---
_PORTABLE_SPEC = os.environ.get("PORTABLE_SPEC")
_TOOL_MODULES = _json.loads(os.environ.get("RUNTIME_TOOL_MODULES") or "[]")
_SHUTDOWN_MODULES = _json.loads(os.environ.get("RUNTIME_SHUTDOWN_MODULES") or "[]")

def _bootstrap_child():
    if not _PORTABLE_SPEC:
        return
    try:
        from kdcube_ai_app.apps.chat.sdk.runtime.bootstrap import bootstrap_from_spec
        for mod_name in (_TOOL_MODULES or []):
            try:
                m = importlib.import_module(mod_name)
            except Exception:
                continue
            # bootstrap binds ModelServiceBase + registry + integrations into each tool module
            bootstrap_from_spec(_PORTABLE_SPEC, tool_module=m)
    except Exception:
        # non-fatal
        pass

_bootstrap_child()

# --- Tool alias imports (auto-injected) ---
<TOOL_IMPORTS_SRC>

<GLOBALS_SRC>

# -------- Live progress cache (safe, in-process) --------
_PROGRESS = {
    "objective": "",
    "status": "In progress",
    "story": [],          # list[str]
    "out_dyn": {},        # slot_name -> slot dict (inline/file)
}
_FINALIZED = False  # prevents late checkpoints from overwriting the final result

def _build_project_log_md() -> str:
    lines = []
    lines.append("# Project Log")
    lines.append("")
    lines.append("## Objective")
    lines.append(str(_PROGRESS.get("objective", "")))
    lines.append("")
    lines.append("## Status")
    lines.append(str(_PROGRESS.get("status", "")))
    lines.append("")
    lines.append("## Story")
    story = " ".join(_PROGRESS.get("story") or [])
    lines.append(story)
    lines.append("")
    lines.append("## Produced Slots")

    for name, data in (_PROGRESS.get("out_dyn") or {}).items():
        if name == "project_log":
            continue
        t = (data.get("type") or "inline")
        desc = data.get("description", "")
        lines.append(f"### {name} ({t})")
        if desc:
            lines.append(desc)
        if t == "file":
            mime = data.get("mime", "")
            path = data.get("path", "")
            if mime:
                lines.append(f"**Mime:** {mime}")
            if path:
                lines.append(f"**Filename:** {path}")
        else:
            fmt = data.get("format", "")
            if fmt:
                lines.append(f"**Format:** {fmt}")

    return "\\n".join(lines).strip()

def _refresh_project_log_slot():
    """Keep 'project_log' as a first-class slot in _PROGRESS['out_dyn']."""
    od = _PROGRESS["out_dyn"]
    md = _build_project_log_md()
    od["project_log"] = {
        "type": "inline",
        "format": "markdown",
        "description": "Live run log",
        "value": md,
    }

async def set_progress(*, objective=None, status=None, story_append=None, out_dyn_patch=None, flush=False):
    if objective is not None:
        _PROGRESS["objective"] = str(objective)
    if status is not None:
        _PROGRESS["status"] = str(status)
    if story_append:
        if isinstance(story_append, (list, tuple)):
            _PROGRESS["story"].extend([str(s) for s in story_append])
        else:
            _PROGRESS["story"].append(str(story_append))
    if out_dyn_patch:
        od = _PROGRESS["out_dyn"]
        for k, v in (out_dyn_patch or {}).items():
            od[k] = v

    _refresh_project_log_slot()

    if flush and not globals().get("_FINALIZED", False):
        await _write_checkpoint(reason="progress", managed=True)
        
# initialize with an empty log so it's present even before first set_progress()
_refresh_project_log_slot()

async def _write_checkpoint(reason: str = "checkpoint", managed: bool = True):
    if globals().get("_FINALIZED", False):
        return
    try:
        # ensure log is up-to-date at checkpoint time
        _refresh_project_log_slot()
        g = globals()
        payload = {
            "ok": False,
            "objective": str(_PROGRESS.get("objective") or g.get("objective") or g.get("OBJECTIVE") or ""),
            "contract": (g.get("CONTRACT") or {}),
            "out_dyn": dict(_PROGRESS.get("out_dyn") or {}),
            "error": {"where": "runtime", "details": "", "error": reason, "description": reason, "managed": bool(managed)}
        }
        await agent_io_tools.save_ret(data=_json.dumps(payload), filename="result.json")
    except Exception:
        pass

async def done():
    # ensure latest log
    _refresh_project_log_slot()
    g = globals()
    # normalize status
    status = (_PROGRESS.get("status") or "Completed")
    if status.lower().startswith("in progress"):
        _PROGRESS["status"] = "Completed"
        _refresh_project_log_slot()
    payload = {
        "ok": True,
        "objective": str(_PROGRESS.get("objective") or g.get("objective") or g.get("OBJECTIVE") or ""),
        "contract": (g.get("CONTRACT") or {}),
        "out_dyn": dict(_PROGRESS.get("out_dyn") or {})
    }
    globals()["_FINALIZED"] = True  # ← set BEFORE writing final file
    return await agent_io_tools.save_ret(data=_json.dumps(payload), filename="result.json")

async def fail(description: str,
               where: str = "",
               error: str = "",
               details: str = "",
               managed: bool = True,
               out_dyn: dict | None = None):
    """
    Managed failure helper. Always writes result.json with a normalized envelope.
    Uses the canonical _PROGRESS['out_dyn'] which already contains 'project_log'.
    """
    # update status for the log and refresh slot
    _PROGRESS["status"] = "Failed"
    _refresh_project_log_slot()

    g = globals()
    payload = {
        "ok": False,
        "objective": str(_PROGRESS.get("objective") or description),
        "contract": (g.get("CONTRACT") or {}),
        "out_dyn": dict(_PROGRESS.get("out_dyn") or {}),
        "error": {
            "where": (where or "runtime"),
            "details": str(details or ""),
            "error": str(error or ""),
            "description": description,
            "managed": bool(managed),
        }
    }
    globals()["_FINALIZED"] = True  # ← set BEFORE writing final file
    return await agent_io_tools.save_ret(data=_json.dumps(payload), filename="result.json")

def _on_term(signum, frame):
    try:
        if globals().get("_FINALIZED", False):
            return
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_write_checkpoint(reason=f"signal:{signum}", managed=True))
        else:
            loop.run_until_complete(_write_checkpoint(reason=f"signal:{signum}", managed=True))
    except Exception:
        pass

# --- Module shutdown on exit (KB, tool modules etc.) ---
async def _async_shutdown_mod(mod):
    try:
        if hasattr(mod, "shutdown") and callable(mod.shutdown):
            maybe = mod.shutdown()
            if asyncio.iscoroutine(maybe):
                await maybe
        elif hasattr(mod, "close") and callable(mod.close):
            maybe = mod.close()
            if asyncio.iscoroutine(maybe):
                await maybe
    except Exception:
        pass

def _sync_shutdown_all():
    try:
        import importlib
        mods = []
        for name in set(_SHUTDOWN_MODULES or []):
            try:
                mods.append(importlib.import_module(name))
            except Exception:
                pass
        async def _run():
            for m in mods:
                await _async_shutdown_mod(m)
        asyncio.run(_run())
    except Exception:
        pass

def _on_atexit():
    try:
        if globals().get("_FINALIZED", False):
            return
        asyncio.run(_write_checkpoint(reason="atexit", managed=True))
    except Exception:
        pass

try:
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
except Exception:
    pass
# atexit.register(lambda: asyncio.run(_write_checkpoint(reason="atexit", managed=True)))
atexit.register(_on_atexit)
atexit.register(_sync_shutdown_all)

# === END HEADER ===
# === CHAT COMMUNICATOR RECONSTRUCTION ===
from kdcube_ai_app.apps.chat.sdk.protocol import (
    ChatEnvelope, ServiceCtx, ConversationCtx
)
from kdcube_ai_app.apps.chat.emitters import (ChatRelayCommunicator, ChatCommunicator, _RelayEmitterAdapter)
from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import set_comm

def _rebuild_communicator_from_spec(spec: dict) -> ChatCommunicator:
    redis_url = (spec or {}).get("redis_url") or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    channel   = (spec or {}).get("channel")   or "chat.events"
    relay = ChatRelayCommunicator(redis_url=redis_url, channel=channel)
    emitter = _RelayEmitterAdapter(relay)
    svc = ServiceCtx(**(spec.get("service") or {}))
    conv = ConversationCtx(**(spec.get("conversation") or {}))
    comm = ChatCommunicator(
        emitter=emitter,
        service=svc.model_dump() if hasattr(svc, "model_dump") else dict(svc),
        conversation=conv.model_dump() if hasattr(conv, "model_dump") else dict(conv),
        room=spec.get("room"),
        target_sid=spec.get("target_sid"),
    )
    return comm

try:
    _COMM_SPEC = globals().get("COMM_SPEC") or {}
    if _COMM_SPEC:
        _comm_obj = _rebuild_communicator_from_spec(_COMM_SPEC)
        set_comm(_comm_obj)
except Exception as _comm_err:
    pass
# === END COMMUNICATOR SETUP ===
''')
                                   .replace('<GLOBALS_SRC>', globals_src)
                                   .replace('<TOOL_IMPORTS_SRC>', imports_src))
                src = _inject_header_after_future(src, injected_header)

                # 3b) Persist the rewritten file (optional: keep original alongside)
                (workdir / "main.py").write_text(src, encoding="utf-8")

                # --- Execute as a script (isolated __main__) ---
                runpy.run_path(str(workdir / "main.py"), run_name="__main__")

            finally:
                try:
                    OUTDIR_CV.reset(t_out)
                    WORKDIR_CV.reset(t_wrk)
                    SOURCE_ID_CV.reset(t_sid)
                except Exception:
                    pass
                try:
                    # best-effort parent-side shutdown for tools with .shutdown()
                    for _, mod in tool_modules or []:
                        try:
                            if hasattr(mod, "shutdown"):
                                mod.shutdown()
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    sys.path[:] = old_path
                    os.environ.clear()
                    os.environ.update(old_env)
                except Exception:
                    print(traceback.format_exc())

        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(_run_in_executor_with_ctx(loop, _runner), timeout=timeout_s)
            return {"ok": True}
        except asyncio.TimeoutError:
            return {"error": "timeout", "seconds": timeout_s}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


