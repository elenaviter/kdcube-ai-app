# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/runtime/simple_runtime.py
import io
import os, sys, re
import asyncio
import pathlib
import runpy
import tokenize
from typing import Dict, Any, List, Tuple

from kdcube_ai_app.apps.chat.sdk.inventory import AgentLogger

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
    """Replace NAME tokens true/false/null with True/False/None (not inside strings)."""
    out = []
    tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    mapping = {"true": "True", "false": "False", "null": "None"}
    for toknum, tokval, start, end, line in tokens:
        if toknum == tokenize.NAME and tokval in mapping:
            tokval = mapping[tokval]
        out.append((toknum, tokval))
    return tokenize.untokenize(out)

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
        tool_modules: List[Tuple[str, object]],   # <-- CHANGED: list of (module_name, module_obj)
        timeout_s: int = 90,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        def _runner():
            old_env = dict(os.environ)
            try:
                os.environ["OUTPUT_DIR"] = str(output_dir)
                self._ensure_modules_on_sys_modules(tool_modules)
                glb = {"__name__": "__main__"}
                exec(compile(code, "<solver_snippet>", "exec"), glb, glb)
            finally:
                os.environ.clear(); os.environ.update(old_env)

        try:
            await asyncio.wait_for(asyncio.to_thread(_runner), timeout=timeout_s)
            return {"ok": True}
        except asyncio.TimeoutError:
            return {"error": "timeout", "seconds": timeout_s}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    async def run_main_py(
        self,
        *,
        workdir: pathlib.Path,
        output_dir: pathlib.Path,
        tool_modules: List[Tuple[str, object]],   # <-- CHANGED
        timeout_s: int = 90,
    ) -> Dict[str, Any]:
        workdir.mkdir(parents=True, exist_ok=True)

        from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV, WORKDIR_CV
        def _runner():
            old_env = dict(os.environ)
            old_path = list(sys.path)
            t_out = OUTDIR_CV.set(str(output_dir))
            t_wrk = WORKDIR_CV.set(str(workdir))
            try:
                sys.path.insert(0, str(workdir))
                self._ensure_modules_on_sys_modules(tool_modules)

                src = (workdir / "main.py").read_text(encoding="utf-8")

                # 1) Fix JSON booleans/null FIRST (no exceptions here)
                src = _fix_json_bools(src)

                # 2) Inject the OUTPUT_DIR header (after any __future__)
                injected_header = """
# === AGENT-RUNTIME HEADER (auto-injected, do not edit) ===
from pathlib import Path
from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV
OUTPUT_DIR = OUTDIR_CV.get()
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR missing in run context")
OUTPUT = Path(OUTPUT_DIR)
# === END HEADER ===
"""
                src = _inject_header_after_future(src, injected_header)

                # 3) Persist the rewritten file and run it
                (workdir / "main.py").write_text(src, encoding="utf-8")
                runpy.run_path(str(workdir / "main.py"), run_name="__main__")
            finally:
                OUTDIR_CV.reset(t_out); WORKDIR_CV.reset(t_wrk)
                sys.path[:] = old_path
                os.environ.clear(); os.environ.update(old_env)

        try:
            await asyncio.wait_for(asyncio.to_thread(_runner), timeout=timeout_s)
            return {"ok": True}
        except asyncio.TimeoutError:
            return {"error": "timeout", "seconds": timeout_s}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

