# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/context/prefs/value_eq.py
from __future__ import annotations
from typing import Any, Callable
from math import isfinite

# get_policy: Callable[[str], KeyPolicy]

def _num_eq(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    if not (isfinite(a) and isfinite(b)):
        return False
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom <= tol

def values_equivalent(key: str, a: Any, b: Any, *, get_policy: Callable):
    pol = get_policy(key)
    tol = pol.numeric_tolerance
    if type(a) != type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return _num_eq(float(a), float(b), tol)
        return False
    if isinstance(a, (str, bool)) or a is None:
        return a == b
    if isinstance(a, (int, float)):
        return _num_eq(float(a), float(b), tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(values_equivalent(key, x, y, get_policy=get_policy) for x, y in zip(a, b))
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a:
            if not values_equivalent(key, a[k], b[k], get_policy=get_policy):
                return False
        return True
    return a == b

def canonicalize_value(key: str, v: Any, *, get_policy: Callable):
    pol = get_policy(key)
    if pol.canonicalizer:
        try:
            return pol.canonicalizer(v)
        except Exception:
            pass
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, list):
        return [canonicalize_value(key, x, get_policy=get_policy) for x in v]
    if isinstance(v, dict):
        return {k: canonicalize_value(key, v[k], get_policy=get_policy) for k in sorted(v.keys())}
    return v
