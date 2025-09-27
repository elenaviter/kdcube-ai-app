# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/runtime/run_ctx.py
# Context variables to hold runtime context info
from contextvars import ContextVar

OUTDIR_CV = ContextVar("OUTDIR_CV", default="")
WORKDIR_CV = ContextVar("WORKDIR_CV", default="")

# Holds {'next': int} for continuous sources numbering
SOURCE_ID_CV = ContextVar("SOURCE_ID_CV", default=None)