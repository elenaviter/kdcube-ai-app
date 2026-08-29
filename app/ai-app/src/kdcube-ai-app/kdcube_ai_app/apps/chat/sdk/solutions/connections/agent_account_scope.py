# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Compatibility alias; implementation lives in prokura.agent_account_scope."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("prokura.agent_account_scope")
