# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# utils/text.py

import re

_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

def strip_surrogates(s: str) -> str:
    if not s:
        return s
    return _SURROGATE_RE.sub('', s)

def strip_lone_surrogates(s: str) -> str:
    # Replace any code points in the surrogate range with U+FFFD
    return ''.join('\uFFFD' if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in s)

# def has_surrogates(s: str) -> bool:
#     return any(0xD800 <= ord(ch) <= 0xDFFF for ch in s)

def has_surrogates(obj) -> bool:
    if isinstance(obj, str):
        return bool(_SURROGATE_RE.search(obj))
    if isinstance(obj, list):
        return any(has_surrogates(v) for v in obj)
    if isinstance(obj, dict):
        return any(has_surrogates(v) for v in obj.values())
    return False

def assert_no_surrogates(obj,
                         log,
                         path="root"):
    if isinstance(obj, str):
        if has_surrogates(obj):
            log.error(f"Surrogates found at {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_surrogates(k, f"{path}.<key:{k!r}>")
            assert_no_surrogates(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_surrogates(v, f"{path}[{i}]")

def deep_strip_surrogates(obj):
    if isinstance(obj, str):
        return strip_lone_surrogates(obj)
    if isinstance(obj, list):
        return [deep_strip_surrogates(x) for x in obj]
    if isinstance(obj, dict):
        return {deep_strip_surrogates(k): deep_strip_surrogates(v) for k, v in obj.items()}
    return obj

