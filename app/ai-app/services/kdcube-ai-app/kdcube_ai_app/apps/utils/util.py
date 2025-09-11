# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/util.py

from collections.abc import Mapping, Set as _Set

def _deep_merge(a, b):
    """
    Recursively merge b into a without mutating inputs.
    Rules:
      - dict vs dict  -> recursive merge (right side wins on conflicts)
      - list/tuple    -> concatenate
      - set/frozenset -> union
      - other         -> b overwrites a
    """
    # fast path: exact same object
    if a is b:
        return a

    # dict-like
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = dict(a)
        for k, bv in b.items():
            av = out.get(k, None)
            if k in out:
                out[k] = _deep_merge(av, bv)
            else:
                out[k] = bv
        return out

    # sequences (keep type of left operand where possible)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        cat = list(a) + list(b)
        return type(a)(cat) if isinstance(a, tuple) else cat

    # sets
    if isinstance(a, _Set) and isinstance(b, _Set):
        return type(a)(set(a) | set(b))

    # fallback: overwrite
    return b
