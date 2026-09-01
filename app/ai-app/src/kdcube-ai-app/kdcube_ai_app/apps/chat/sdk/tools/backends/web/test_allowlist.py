# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

import os

from kdcube_ai_app.apps.chat.sdk.tools.backends.web.allowlist import (
    Allowlist,
    hostname_allowed,
    parse_entries,
)


def test_entry_matching():
    entries = parse_entries(["example.org", "*.only-subs.net", "WWW.Mixed.Com"])

    assert hostname_allowed(entries, "example.org")
    assert hostname_allowed(entries, "data.example.org")
    assert hostname_allowed(entries, "EXAMPLE.ORG")
    assert not hostname_allowed(entries, "notexample.org")
    assert not hostname_allowed(entries, "example.org.evil.com")

    assert hostname_allowed(entries, "a.only-subs.net")
    assert not hostname_allowed(entries, "only-subs.net")

    assert hostname_allowed(entries, "www.mixed.com")
    assert hostname_allowed(entries, "cdn.www.mixed.com")
    assert not hostname_allowed(entries, "mixed.com")


def test_parse_entries_skips_comments_and_blanks():
    entries = parse_entries(["# science sources", "", "usgs.gov  # geology", "  "])
    assert entries == ["usgs.gov"]


def test_unconfigured_allows_configured_empty_denies(monkeypatch):
    monkeypatch.delenv("WEB_ALLOWLIST_FILE", raising=False)
    monkeypatch.delenv("WEB_ALLOWLIST", raising=False)
    allowlist = Allowlist.from_env()
    assert not allowlist.configured
    assert allowlist.check("anything.example")

    monkeypatch.setenv("WEB_ALLOWLIST", "")
    allowlist = Allowlist.from_env()
    assert not allowlist.configured  # empty env var reads as unset
    assert allowlist.check("anything.example")

    monkeypatch.setenv("WEB_ALLOWLIST", " , ")
    allowlist = Allowlist.from_env()
    assert allowlist.configured
    assert not allowlist.check("anything.example")


def test_file_source_reloads_on_mtime_change(tmp_path, monkeypatch):
    path = tmp_path / "allowlist.txt"
    path.write_text("# science sources\nexample.org\n")
    monkeypatch.setenv("WEB_ALLOWLIST_FILE", str(path))
    allowlist = Allowlist.from_env()
    assert allowlist.configured
    assert allowlist.check("example.org")
    assert not allowlist.check("usgs.gov")

    path.write_text("example.org\nusgs.gov\n")
    os.utime(path, (os.path.getmtime(path) + 10, os.path.getmtime(path) + 10))
    assert allowlist.check("usgs.gov")


def test_missing_file_denies(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_ALLOWLIST_FILE", str(tmp_path / "absent.txt"))
    allowlist = Allowlist.from_env()
    assert allowlist.configured
    assert not allowlist.check("example.org")
    source, entries = allowlist.describe()
    assert entries == [] and "file:" in source
