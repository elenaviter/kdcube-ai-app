# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/runtime/run_ctx.py
from contextvars import ContextVar

OUTDIR_CV = ContextVar("OUTDIR_CV", default="")
WORKDIR_CV = ContextVar("WORKDIR_CV", default="")
