# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Domain allowlist for the web tools.

Entry format (one domain per entry, operator-owned):

  example.org        matches example.org and every subdomain of it
  www.example.org    matches that exact host and its subdomains
  *.example.org      matches subdomains only, never the bare domain

Matching is case-insensitive on the URL hostname; ports are not part of
the match. When no allowlist is configured, every host is allowed (the
tools behave as before). When an allowlist is configured, only listed
hosts pass — a configured but empty list denies everything.

Sources, in order of precedence:

  WEB_ALLOWLIST_YAML  path to a YAML file whose ``allowlist:`` key holds
                      the entries (the server's config.yaml). An embedding
                      host may select a containing mapping or one exact
                      ``agent.tools[].settings`` mapping. Re-read whenever its
                      mtime changes, so edits apply to the next call.
  WEB_ALLOWLIST_FILE  path to a text file, one entry per line,
                      blank lines and ``#`` comments ignored. Also
                      re-read on mtime change.
  WEB_ALLOWLIST       comma-separated entries, fixed for the process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

ALLOWLIST_YAML_ENV = "WEB_ALLOWLIST_YAML"
ALLOWLIST_FILE_ENV = "WEB_ALLOWLIST_FILE"
ALLOWLIST_ENV = "WEB_ALLOWLIST"

BLOCKLIST_YAML_ENV = "WEB_BLOCKLIST_YAML"
BLOCKLIST_FILE_ENV = "WEB_BLOCKLIST_FILE"
BLOCKLIST_ENV = "WEB_BLOCKLIST"

FILTER_YAML_SECTION_ENV = "WEB_FILTER_YAML_SECTION"
FILTER_YAML_TOOL_ID_ENV = "WEB_FILTER_YAML_TOOL_ID"


def select_yaml_mapping(
    document: Any,
    *,
    path: str,
    section: Optional[str] = None,
    tool_id: Optional[str] = None,
) -> Mapping[str, Any]:
    """Select root, dotted-section, or exact agent-tool settings."""
    if section and tool_id:
        raise ValueError("config section and tool id are mutually exclusive")
    if not isinstance(document, Mapping):
        raise ValueError(f"config file {path} must hold a mapping")
    selected: Any = document
    if section:
        for key in section.split("."):
            selected = selected.get(key) if isinstance(selected, Mapping) else None
        if not isinstance(selected, Mapping):
            raise ValueError(f"config file {path} has no mapping section {section!r}")
    elif tool_id:
        agent = document.get("agent")
        tools = agent.get("tools") if isinstance(agent, Mapping) else None
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise ValueError(f"config file {path} has no agent.tools list")
        matches = [
            item
            for item in tools
            if isinstance(item, Mapping) and str(item.get("id") or "").strip() == tool_id
        ]
        if not matches:
            raise ValueError(f"config file {path} has no agent tool {tool_id!r}")
        if len(matches) > 1:
            raise ValueError(f"config file {path} has duplicate agent tool {tool_id!r}")
        selected = matches[0].get("settings")
        if not isinstance(selected, Mapping):
            raise ValueError(
                f"config file {path} agent tool {tool_id!r} has no settings mapping"
            )
    return selected


def parse_entries(raw_lines: List[str]) -> List[str]:
    entries: List[str] = []
    for line in raw_lines:
        entry = line.split("#", 1)[0].strip().lower().rstrip(".")
        if entry:
            entries.append(entry)
    return entries


def _entry_matches(entry: str, host: str) -> bool:
    if entry.startswith("*."):
        return host.endswith(entry[1:])  # ".example.org" suffix, subdomains only
    return host == entry or host.endswith("." + entry)


def hostname_allowed(entries: List[str], host: Optional[str]) -> bool:
    """True when ``host`` matches one of the allowlist ``entries``."""
    if not host:
        return False
    host = host.strip().lower().rstrip(".")
    return any(_entry_matches(entry, host) for entry in entries)


@dataclass
class Allowlist:
    """A domain list with its source; file sources are re-read on mtime
    change. ``yaml_section`` selects an embedding host's containing mapping;
    ``yaml_tool_id`` selects one exact ``agent.tools[].settings`` mapping;
    ``yaml_key`` selects the list inside its ``filter:`` scope, so the same
    class serves the allowlist and the blocklist."""

    yaml_path: Optional[str] = None
    file_path: Optional[str] = None
    env_value: Optional[str] = None
    yaml_key: str = "allowlist"
    env_label: str = ALLOWLIST_ENV
    yaml_section: Optional[str] = None
    yaml_tool_id: Optional[str] = None
    _entries: List[str] = field(default_factory=list)
    _mtime: Optional[float] = None
    # For the YAML source only: whether the key was present at last read.
    # A watched config whose key is absent counts as NOT configured, so
    # adding the key later (or removing it entirely) applies live.
    _yaml_key_present: bool = False

    @classmethod
    def from_env(cls) -> "Allowlist":
        allowlist = cls(
            yaml_path=os.environ.get(ALLOWLIST_YAML_ENV) or None,
            file_path=os.environ.get(ALLOWLIST_FILE_ENV) or None,
            env_value=os.environ.get(ALLOWLIST_ENV) or None,
            yaml_section=os.environ.get(FILTER_YAML_SECTION_ENV) or None,
            yaml_tool_id=os.environ.get(FILTER_YAML_TOOL_ID_ENV) or None,
        )
        allowlist.refresh()
        return allowlist

    @classmethod
    def blocklist_from_env(cls) -> "Allowlist":
        blocklist = cls(
            yaml_path=os.environ.get(BLOCKLIST_YAML_ENV) or None,
            file_path=os.environ.get(BLOCKLIST_FILE_ENV) or None,
            env_value=os.environ.get(BLOCKLIST_ENV) or None,
            yaml_key="blocklist",
            env_label=BLOCKLIST_ENV,
            yaml_section=os.environ.get(FILTER_YAML_SECTION_ENV) or None,
            yaml_tool_id=os.environ.get(FILTER_YAML_TOOL_ID_ENV) or None,
        )
        blocklist.refresh()
        return blocklist

    @property
    def configured(self) -> bool:
        if self.yaml_path:
            self.refresh()
            return self._yaml_key_present
        return bool(self.file_path) or self.env_value is not None

    def _read_source_file(self, path: str, *, as_yaml: bool) -> None:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self._entries = []
            self._mtime = None
            self._yaml_key_present = False
            return
        if mtime == self._mtime:
            return
        if as_yaml:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data = select_yaml_mapping(
                data,
                path=path,
                section=self.yaml_section,
                tool_id=self.yaml_tool_id,
            )
            raw = None
            if isinstance(data, Mapping):
                scope = data.get("filter")
                if isinstance(scope, Mapping):
                    raw = scope.get(self.yaml_key)
                if raw is None:
                    raw = data.get(self.yaml_key)
            self._yaml_key_present = isinstance(raw, (list, tuple))
            raw_lines = [str(v) for v in raw] if isinstance(raw, (list, tuple)) else []
            self._entries = parse_entries(raw_lines)
        else:
            with open(path, "r", encoding="utf-8") as f:
                self._entries = parse_entries(f.readlines())
        self._mtime = mtime

    def refresh(self) -> None:
        if self.yaml_path:
            self._read_source_file(self.yaml_path, as_yaml=True)
        elif self.file_path:
            self._read_source_file(self.file_path, as_yaml=False)
        elif self.env_value is not None:
            self._entries = parse_entries(self.env_value.split(","))
        else:
            self._entries = []

    @property
    def entries(self) -> List[str]:
        self.refresh()
        return list(self._entries)

    def check(self, host: Optional[str]) -> bool:
        if not self.configured:
            return True
        return hostname_allowed(self.entries, host)

    def matches(self, host: Optional[str]) -> bool:
        """True when a configured list contains ``host`` (blocklist use)."""
        return self.configured and hostname_allowed(self.entries, host)

    def describe(self) -> Tuple[str, List[str]]:
        """(source description, entries) — the same truth for operator and model."""
        entries = self.entries
        if self.yaml_path:
            selector = ""
            if self.yaml_tool_id:
                selector = f"#agent.tools[id={self.yaml_tool_id}].settings"
            elif self.yaml_section:
                selector = f"#{self.yaml_section}"
            return f"config: {self.yaml_path}{selector}", entries
        if self.file_path:
            return f"file: {self.file_path}", entries
        if self.env_value is not None:
            return f"env: {self.env_label}", entries
        if self.yaml_key == "blocklist":
            return "not configured (no host is blocked)", entries
        return "not configured (every host is allowed)", entries


@dataclass
class EgressFilter:
    """The operator's egress policy: blocklist (deny wins) over allowlist.

    ``check(host)`` is the one gate: a blocklisted host is denied even
    when the allowlist would admit it; with no allowlist configured,
    everything except the blocklist passes.
    """

    allowlist: Allowlist
    blocklist: Allowlist

    @classmethod
    def from_env(cls) -> "EgressFilter":
        return cls(
            allowlist=Allowlist.from_env(),
            blocklist=Allowlist.blocklist_from_env(),
        )

    @property
    def configured(self) -> bool:
        return self.allowlist.configured or self.blocklist.configured

    def check(self, host: Optional[str]) -> bool:
        if self.blocklist.matches(host):
            return False
        return self.allowlist.check(host)

    def deny_reason(self, host: Optional[str]) -> str:
        """Why ``check`` said no — for in-band denial results."""
        if self.blocklist.matches(host):
            source, _entries = self.blocklist.describe()
            return f"host '{host}' is on the blocklist ({source})"
        source, _entries = self.allowlist.describe()
        return f"host '{host}' is outside the allowlist ({source})"
