# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/runtime/workdir_discovery.py
import pathlib
from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV

def resolve_output_dir() -> pathlib.Path:
    v = OUTDIR_CV.get()
    if not v:
        raise RuntimeError("OUTPUT_DIR not set in run context")
    p = pathlib.Path(v).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p