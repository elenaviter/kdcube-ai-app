# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Boris Varer

# tools/execute_code.py

from __future__ import annotations

import ast
import io
import json
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pathlib import Path

import numpy as np  # exposed to the executed code as np
import pandas as pd  # exposed to the executed code as pd

from kdcube_ai_app.tools.processing import ProcessingResult, record_timing

PLOTS_DIR = Path(os.environ.get("PLOTS_DIR", "")).resolve()
if PLOTS_DIR != "":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class ExecuteCodeResult(ProcessingResult):
    """
    Structured result of executing a code snippet in the sandbox.
    Inherits timestamps/duration and fits your unified result model.
    """
    code_executed: str
    output: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    success: bool = True
    saved_files_urls: List[str] = []
    shared_memory_keys: List[str] = []


class ExecuteCodeTool:

    name = "execute_code"
    description = "Execute Python code with shared memory and matplotlib plotting."

    def __init__(self, save_file_callable):
        """
        Parameters
        ----------
        save_file_callable : callable
            A function like `save_file(bytes_or_str, filename)` -> "/files/..."
            Your runtime already exposes it; we accept it via DI so this class
            is easy to test.
        """
        self.save_file = save_file_callable

    @record_timing 
    def run(self, context, code: str) -> ExecuteCodeResult:
        saved_files_urls: List[str] = []

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shared = context.shared_memory
            if shared is None:
                shared = {}

            captured_output = io.StringIO()

            def new_print(*args, **kwargs):
                print(*args, file=captured_output, **kwargs)

            original_savefig = plt.savefig

            def enhanced_savefig(filename, *args, **kwargs):
                nonlocal saved_files_urls

                file_path = PLOTS_DIR / filename
                original_name = filename
                if file_path.exists():
                    name, ext = os.path.splitext(filename)
                    unique_filename = f"{name}-{uuid4().hex[:8]}{ext}"
                    original_name = name
                    file_path = PLOTS_DIR / unique_filename
                    filename = unique_filename

                original_savefig(file_path, *args, **kwargs)

                url = f"/files/{filename}"
                new_print(f"Plot saved: {url}")
                saved_files_urls.append(url)

                display_name = original_name
                try:
                    fig = plt.gcf()
                    if getattr(fig, "_suptitle", None) and fig._suptitle.get_text().strip():
                        display_name = fig._suptitle.get_text().strip()
                    elif fig.axes and fig.axes[0].get_title().strip():
                        display_name = fig.axes[0].get_title().strip()
                    elif fig.axes:
                        ax = fig.axes[0]
                        xlabel = ax.get_xlabel().strip()
                        ylabel = ax.get_ylabel().strip()
                        if xlabel and ylabel:
                            display_name = f"{ylabel} vs {xlabel}"
                except Exception:
                    pass

                shared[str(filename)] = {
                    "type": "image",
                    "filename": str(original_name),
                    "display_name": display_name,
                    "url": url,
                }
                return url

            exec_globals: Dict[str, Any] = {
                "pd": pd,
                "np": np,
                "plt": plt,
                "print": new_print,
                "save_file": self.save_file,
                "shared": shared,
            }

            try:
                plt.savefig = enhanced_savefig

                result_obj = _eval_last_expr(code, globals_=exec_globals)
                output = captured_output.getvalue()

                if hasattr(context.context.shared_memory, "_persist"):
                    context.context.shared_memory._persist()

                return ExecuteCodeResult(
                    code_executed=code,
                    output=output,
                    result=str(result_obj) if result_obj is not None else None,
                    success=True,
                    saved_files_urls=saved_files_urls,
                    shared_memory_keys=list(shared.keys()),
                )

            except Exception as e:
                return ExecuteCodeResult(
                    code_executed=code,
                    output=captured_output.getvalue(),
                    error=str(e),
                    traceback=traceback.format_exc(),
                    success=False,
                    saved_files_urls=saved_files_urls,
                    shared_memory_keys=list(shared.keys()),
                )
            finally:
                try:
                    plt.savefig = original_savefig
                except Exception:
                    pass

        except Exception as e:
            return ExecuteCodeResult(
                code_executed=code,
                error=f"Failed to execute code: {e}",
                success=False,
                saved_files_urls=saved_files_urls,
                shared_memory_keys=[],
            )


def _eval_last_expr(code_str: str, globals_: Optional[Dict[str, Any]] = None, locals_: Optional[Dict[str, Any]] = None):
    """
    Execute a code block, and if the last statement is an expression, return its value.
    This mirrors your previous `eval_last_expr` helper.
    """
    globals_ = globals_ or {}
    if locals_ is None:
        locals_ = globals_
    globals_.setdefault("__builtins__", __builtins__)

    tree = ast.parse(code_str)
    if not tree.body:
        return None

    *body, last = tree.body
    exec(compile(ast.Module(body=body, type_ignores=[]), filename="<exec>", mode="exec"), globals_, locals_)
    if isinstance(last, ast.Expr):
        return eval(compile(ast.Expression(body=last.value), filename="<eval>", mode="eval"), globals_, locals_)
    else:
        exec(compile(ast.Module(body=[last], type_ignores=[]), filename="<exec>", mode="exec"), globals_, locals_)
        return None
