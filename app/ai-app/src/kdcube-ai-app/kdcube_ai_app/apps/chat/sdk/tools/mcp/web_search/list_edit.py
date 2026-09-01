# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Textual editor for the config.yaml egress lists.

Serves the operator-enabled ``allowlist_edit`` tool: adds/removes
entries under ``filter.allowlist`` / ``filter.blocklist`` by editing
LINES, so everything else in the file — comments included — stays
byte-identical. Only block-style lists are supported (the template's
shape); anything else is refused with the reason, never guessed at.

The editor touches nothing but list entries: keys, the ssrf_guard knob,
secrets, and models are out of its reach by construction.
"""

from __future__ import annotations

import pathlib
import re
from typing import List, Optional, Tuple

_ENTRY_RE = re.compile(r"^\*?\.?[a-z0-9]([a-z0-9.-]*[a-z0-9])?\.[a-z0-9-]{2,}$")


def normalize_entry(entry: str) -> Optional[str]:
    """Lowercased, dot-trimmed domain entry, or None when it is not a
    plausible domain (we refuse rather than write garbage into an
    egress control)."""
    e = str(entry or "").strip().lower().rstrip(".")
    if e.startswith("*.") and _ENTRY_RE.match(e[2:] and "x." + e[2:] or ""):
        pass  # validated below on the suffix
    candidate = e[2:] if e.startswith("*.") else e
    if not candidate or not _ENTRY_RE.match(candidate):
        return None
    return e


def _line_entry(line: str) -> Optional[str]:
    m = re.match(r"^(\s+)-\s+(.+?)\s*(#.*)?$", line)
    if not m:
        return None
    value = m.group(2).strip().strip("'\"")
    return value.lower().rstrip(".")


def edit_lists(
    config_path: str | pathlib.Path,
    *,
    list_name: str,
    add: Optional[List[str]] = None,
    remove: Optional[List[str]] = None,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Apply adds/removes to one list in the YAML config, textually.

    Returns (entries_after, None) on success or (None, reason) on
    refusal. Refusals: unknown list name, invalid entries, a list that
    comes from a separate *_file (edit that file instead), or a config
    shape the line editor cannot handle safely.
    """
    if list_name not in ("allowlist", "blocklist"):
        return None, f"unknown list '{list_name}'; use 'allowlist' or 'blocklist'"

    adds: List[str] = []
    for raw in add or []:
        norm = normalize_entry(raw)
        if norm is None:
            return None, f"'{raw}' does not look like a domain entry; refused"
        adds.append(norm)
    removes = {normalize_entry(r) or str(r).strip().lower() for r in (remove or [])}
    if not adds and not removes:
        return None, "nothing to do: pass add and/or remove entries"

    path = pathlib.Path(config_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as e:
        return None, f"cannot read config: {e}"

    # locate the top-level filter: block
    filter_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^filter:\s*(#.*)?$", line):
            filter_idx = i
            break
    if filter_idx is None:
        if f"{list_name}_file" in "".join(lines):
            return None, (
                f"the {list_name} comes from a separate file "
                f"({list_name}_file); edit that file instead"
            )
        # no filter block at all: append one
        block = [f"filter:\n", f"  {list_name}:\n"] + [f"    - {e}\n" for e in adds]
        lines = lines + (["\n"] if lines and not lines[-1].endswith("\n") else []) + block
        path.write_text("".join(lines), encoding="utf-8")
        return adds, None

    # bounds of the filter block: until the next non-indented, non-blank line
    end_idx = len(lines)
    for i in range(filter_idx + 1, len(lines)):
        stripped = lines[i].rstrip("\n")
        if stripped and not stripped.startswith((" ", "\t", "#")):
            end_idx = i
            break

    block_text = "".join(lines[filter_idx:end_idx])
    if f"{list_name}_file" in block_text:
        return None, (
            f"the {list_name} comes from a separate file ({list_name}_file); "
            "edit that file instead"
        )

    key_idx = None
    for i in range(filter_idx + 1, end_idx):
        m = re.match(rf"^(\s+){list_name}:\s*(\S.*)?$", lines[i])
        if m:
            if m.group(2) and not m.group(2).startswith("#"):
                return None, (
                    f"'{list_name}' uses an inline/flow value the line editor "
                    "does not handle; edit the file manually"
                )
            key_idx = i
            break

    if key_idx is None:
        if removes and not adds:
            return [], None  # nothing listed, nothing to remove
        insert = [f"  {list_name}:\n"] + [f"    - {e}\n" for e in adds]
        lines[end_idx:end_idx] = insert
        path.write_text("".join(lines), encoding="utf-8")
        return adds, None

    # collect existing entry lines directly under the key
    entry_indices: List[int] = []
    for i in range(key_idx + 1, end_idx):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _line_entry(lines[i]) is not None:
            entry_indices.append(i)
        else:
            break  # a sibling key ends the list

    existing = {(_line_entry(lines[i]) or ""): i for i in entry_indices}

    drop = {i for e, i in existing.items() if e in removes}
    new_adds = [e for e in adds if e not in existing]

    out: List[str] = []
    kept_indices = [i for i in entry_indices if i not in drop]
    anchor_idx = kept_indices[-1] if kept_indices else key_idx
    indent = "    "
    if entry_indices:
        m = re.match(r"^(\s+)-", lines[entry_indices[0]])
        if m:
            indent = m.group(1)
    for i, line in enumerate(lines):
        if i in drop:
            continue
        out.append(line)
        if i == anchor_idx and new_adds:
            out.extend(f"{indent}- {e}\n" for e in new_adds)

    path.write_text("".join(out), encoding="utf-8")

    after = [e for e, i in sorted(existing.items(), key=lambda kv: kv[1]) if i not in drop]
    after.extend(new_adds)
    return after, None
