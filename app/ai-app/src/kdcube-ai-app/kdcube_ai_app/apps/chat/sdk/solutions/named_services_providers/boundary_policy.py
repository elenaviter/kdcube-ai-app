# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Compatibility alias; authority boundary policy lives in Connection Hub."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("connection_hub.named_service_boundary")
