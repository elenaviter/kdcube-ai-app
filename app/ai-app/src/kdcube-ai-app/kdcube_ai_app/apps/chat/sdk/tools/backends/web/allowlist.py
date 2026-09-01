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

  WEB_ALLOWLIST_FILE  path to a text file, one entry per line,
                      blank lines and ``#`` comments ignored. The file is
                      re-read whenever its mtime changes, so edits apply
                      to the next call without a restart.
  WEB_ALLOWLIST       comma-separated entries, fixed for the process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ALLOWLIST_FILE_ENV = "WEB_ALLOWLIST_FILE"
ALLOWLIST_ENV = "WEB_ALLOWLIST"


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
    """Allowlist with its source; a file source is re-read on mtime change."""

    file_path: Optional[str] = None
    env_value: Optional[str] = None
    _entries: List[str] = field(default_factory=list)
    _mtime: Optional[float] = None

    @classmethod
    def from_env(cls) -> "Allowlist":
        allowlist = cls(
            file_path=os.environ.get(ALLOWLIST_FILE_ENV) or None,
            env_value=os.environ.get(ALLOWLIST_ENV) or None,
        )
        allowlist.refresh()
        return allowlist

    @property
    def configured(self) -> bool:
        return bool(self.file_path) or self.env_value is not None

    def refresh(self) -> None:
        if self.file_path:
            try:
                mtime = os.path.getmtime(self.file_path)
            except OSError:
                self._entries = []
                self._mtime = None
                return
            if mtime == self._mtime:
                return
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._entries = parse_entries(f.readlines())
            self._mtime = mtime
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

    def describe(self) -> Tuple[str, List[str]]:
        """(source description, entries) — the same truth for operator and model."""
        entries = self.entries
        if self.file_path:
            return f"file: {self.file_path}", entries
        if self.env_value is not None:
            return f"env: {ALLOWLIST_ENV}", entries
        return "not configured (every host is allowed)", entries
