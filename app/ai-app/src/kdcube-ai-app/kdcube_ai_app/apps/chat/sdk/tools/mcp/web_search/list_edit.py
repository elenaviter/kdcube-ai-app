# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Textual editor for the config.yaml egress lists.

Serves the operator-enabled ``site_filter_edit`` tool: adds/removes
entries under ``filter.allowlist`` / ``filter.blocklist`` by editing LINES,
including when that mapping is nested inside an embedding host's selected
configuration section or one exact ``agent.tools[].settings`` mapping.
Everything else in the file — comments included — stays byte-identical. Only
block-style lists are supported (the template's shape); anything else is
refused with the reason, never guessed at.

The editor touches nothing but list entries: keys, the ssrf_guard knob,
secrets, and models are out of its reach by construction.
"""

from __future__ import annotations

import pathlib
import re
from typing import List, Optional, Tuple

_ENTRY_RE = re.compile(r"^\*?\.?[a-z0-9]([a-z0-9.-]*[a-z0-9])?\.[a-z0-9-]{2,}$")


def _has_active_key(text: str, key: str) -> bool:
    """True when ``key:`` appears as a real (uncommented) mapping key."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(rf"^{re.escape(key)}\s*:", stripped):
            return True
    return False


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


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_end(lines: List[str], start: int, *, indent: int, limit: int) -> int:
    for index in range(start + 1, limit):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _indent(lines[index]) <= indent:
            return index
    return limit


def _mapping_scope(
    lines: List[str],
    section: Optional[str],
) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    """Return (content_start, content_end, child_indent) for a YAML mapping."""
    start = 0
    end = len(lines)
    parent_indent = -2
    for key in (section or "").split("."):
        if not key:
            continue
        child_indent = parent_indent + 2
        pattern = re.compile(
            rf"^{' ' * child_indent}{re.escape(key)}:\s*(#.*)?$"
        )
        key_index = next(
            (index for index in range(start, end) if pattern.match(lines[index].rstrip("\n"))),
            None,
        )
        if key_index is None:
            return None, f"config section '{section}' is missing or is not a block mapping"
        end = _block_end(lines, key_index, indent=child_indent, limit=end)
        start = key_index + 1
        parent_indent = child_indent
    return (start, end, parent_indent + 2), None


def _tool_settings_scope(
    lines: List[str],
    tool_id: str,
) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    """Locate one exact block-style ``agent.tools`` item's settings mapping."""
    agent_scope, error = _mapping_scope(lines, "agent")
    if error:
        return None, error
    assert agent_scope is not None
    agent_start, agent_end, agent_child_indent = agent_scope
    tools_pattern = re.compile(rf"^{' ' * agent_child_indent}tools:\s*(#.*)?$")
    tools_index = next(
        (
            index
            for index in range(agent_start, agent_end)
            if tools_pattern.match(lines[index].rstrip("\n"))
        ),
        None,
    )
    if tools_index is None:
        return None, "config section 'agent.tools' is missing or is not a block list"
    tools_end = _block_end(
        lines,
        tools_index,
        indent=agent_child_indent,
        limit=agent_end,
    )
    item_indent = agent_child_indent + 2
    id_pattern = re.compile(rf"^{' ' * item_indent}-\s+id:\s*([^#]+?)(?:\s+#.*)?$")
    matches: List[int] = []
    for index in range(tools_index + 1, tools_end):
        match = id_pattern.match(lines[index].rstrip("\n"))
        if not match:
            continue
        value = match.group(1).strip().strip("'\"")
        if value == tool_id:
            matches.append(index)
    if not matches:
        return None, f"config has no block-style agent tool {tool_id!r}"
    if len(matches) > 1:
        return None, f"config has duplicate agent tool {tool_id!r}"

    item_index = matches[0]
    item_end = _block_end(lines, item_index, indent=item_indent, limit=tools_end)
    settings_indent = item_indent + 2
    settings_key = re.compile(rf"^{' ' * settings_indent}settings:\s*(.*)$")
    settings_matches = [
        (index, settings_key.match(lines[index].rstrip("\n")))
        for index in range(item_index + 1, item_end)
        if settings_key.match(lines[index].rstrip("\n"))
    ]
    if len(settings_matches) > 1:
        return None, f"agent tool {tool_id!r} has duplicate settings mappings"
    if settings_matches:
        settings_index, match = settings_matches[0]
        assert match is not None
        trailing = match.group(1).strip()
        if trailing and not trailing.startswith("#"):
            return None, (
                f"agent tool {tool_id!r} settings uses an inline/flow value; "
                "edit the file manually"
            )
        settings_end = _block_end(
            lines,
            settings_index,
            indent=settings_indent,
            limit=item_end,
        )
        return (settings_index + 1, settings_end, settings_indent + 2), None

    if item_end and not lines[item_end - 1].endswith("\n"):
        lines[item_end - 1] += "\n"
    lines[item_end:item_end] = [f"{' ' * settings_indent}settings:\n"]
    return (item_end + 1, item_end + 1, settings_indent + 2), None


def edit_lists(
    config_path: str | pathlib.Path,
    *,
    list_name: str,
    add: Optional[List[str]] = None,
    remove: Optional[List[str]] = None,
    config_section: Optional[str] = None,
    config_tool_id: Optional[str] = None,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Apply adds/removes to one list in the YAML config, textually.

    Returns (entries_after, None) on success or (None, reason) on
    refusal. Refusals: unknown list name, invalid entries, a list that
    comes from a separate *_file (edit that file instead), or a config
    shape the line editor cannot handle safely.
    """
    if config_section and config_tool_id:
        return None, "config section and tool id are mutually exclusive"
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

    if config_tool_id:
        scope, scope_error = _tool_settings_scope(lines, config_tool_id)
    else:
        scope, scope_error = _mapping_scope(lines, config_section)
    if scope_error:
        return None, scope_error
    assert scope is not None
    scope_start, scope_end, filter_indent = scope

    # Locate the filter block inside the selected configuration mapping.
    filter_idx = None
    filter_pattern = re.compile(
        rf"^{' ' * filter_indent}filter:\s*(#.*)?$"
    )
    for i in range(scope_start, scope_end):
        if filter_pattern.match(lines[i].rstrip("\n")):
            filter_idx = i
            break
    if filter_idx is None:
        if _has_active_key("".join(lines[scope_start:scope_end]), f"{list_name}_file"):
            return None, (
                f"the {list_name} comes from a separate file "
                f"({list_name}_file); edit that file instead"
            )
        indent = " " * filter_indent
        block = [f"{indent}filter:\n", f"{indent}  {list_name}:\n"] + [
            f"{indent}    - {entry}\n" for entry in adds
        ]
        if scope_end and not lines[scope_end - 1].endswith("\n"):
            lines[scope_end - 1] += "\n"
        lines[scope_end:scope_end] = block
        path.write_text("".join(lines), encoding="utf-8")
        return adds, None

    end_idx = _block_end(lines, filter_idx, indent=filter_indent, limit=scope_end)

    block_text = "".join(lines[filter_idx:end_idx])
    if _has_active_key(block_text, f"{list_name}_file"):
        return None, (
            f"the {list_name} comes from a separate file ({list_name}_file); "
            "edit that file instead"
        )

    key_idx = None
    key_indent = filter_indent + 2
    for i in range(filter_idx + 1, end_idx):
        m = re.match(
            rf"^({' ' * key_indent}){list_name}:\s*(\S.*)?$",
            lines[i],
        )
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
        indent = " " * key_indent
        insert = [f"{indent}{list_name}:\n"] + [
            f"{indent}  - {entry}\n" for entry in adds
        ]
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
    indent = " " * (key_indent + 2)
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
