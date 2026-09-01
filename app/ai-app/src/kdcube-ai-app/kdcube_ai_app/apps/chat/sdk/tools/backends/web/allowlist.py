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
                      the entries (the server's config.yaml). Re-read
                      whenever its mtime changes, so edits apply to the
                      next call without a restart.
  WEB_ALLOWLIST_FILE  path to a text file, one entry per line,
                      blank lines and ``#`` comments ignored. Also
                      re-read on mtime change.
  WEB_ALLOWLIST       comma-separated entries, fixed for the process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ALLOWLIST_YAML_ENV = "WEB_ALLOWLIST_YAML"
ALLOWLIST_FILE_ENV = "WEB_ALLOWLIST_FILE"
ALLOWLIST_ENV = "WEB_ALLOWLIST"

BLOCKLIST_YAML_ENV = "WEB_BLOCKLIST_YAML"
BLOCKLIST_FILE_ENV = "WEB_BLOCKLIST_FILE"
BLOCKLIST_ENV = "WEB_BLOCKLIST"


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
    change. ``yaml_key`` selects which key of the YAML's ``filter:``
    scope holds the entries, so the same class serves the allowlist and
    the blocklist."""

    yaml_path: Optional[str] = None
    file_path: Optional[str] = None
    env_value: Optional[str] = None
    yaml_key: str = "allowlist"
    env_label: str = ALLOWLIST_ENV
    _entries: List[str] = field(default_factory=list)
    _mtime: Optional[float] = None

    @classmethod
    def from_env(cls) -> "Allowlist":
        allowlist = cls(
            yaml_path=os.environ.get(ALLOWLIST_YAML_ENV) or None,
            file_path=os.environ.get(ALLOWLIST_FILE_ENV) or None,
            env_value=os.environ.get(ALLOWLIST_ENV) or None,
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
        )
        blocklist.refresh()
        return blocklist

    @property
    def configured(self) -> bool:
        return bool(self.yaml_path) or bool(self.file_path) or self.env_value is not None

    def _read_source_file(self, path: str, *, as_yaml: bool) -> None:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self._entries = []
            self._mtime = None
            return
        if mtime == self._mtime:
            return
        if as_yaml:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            raw = None
            if isinstance(data, dict):
                scope = data.get("filter")
                if isinstance(scope, dict):
                    raw = scope.get(self.yaml_key)
                if raw is None:
                    raw = data.get(self.yaml_key)
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
            return f"config: {self.yaml_path}", entries
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
