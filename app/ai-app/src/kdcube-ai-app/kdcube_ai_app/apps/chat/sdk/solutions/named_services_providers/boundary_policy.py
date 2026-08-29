# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Compatibility alias; authority boundary policy lives in Prokura."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("prokura.named_service_boundary")
