# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

import pathlib

from kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.list_edit import (
    edit_lists,
    normalize_entry,
)

TEMPLATE = """# operator config
filter:
  allowlist:            # the live source
    - example.org       # keep
    - old.example
  # a comment inside the block
services:
  secrets:
    brave:
      api_key: "k"
"""


def _write(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_normalize_entry():
    assert normalize_entry(" Example.ORG. ") == "example.org"
    assert normalize_entry("*.example.net") == "*.example.net"
    assert normalize_entry("not a domain") is None
    assert normalize_entry("localhost") is None
    assert normalize_entry("") is None


def test_add_and_remove_preserve_everything_else(tmp_path):
    p = _write(tmp_path, TEMPLATE)
    entries, err = edit_lists(p, list_name="allowlist", add=["noaa.gov"], remove=["old.example"])
    assert err is None
    assert entries == ["example.org", "noaa.gov"]
    text = p.read_text()
    assert "- noaa.gov" in text and "old.example" not in text
    # comments and unrelated sections byte-identical
    assert "# operator config" in text
    assert "# the live source" in text
    assert "# a comment inside the block" in text
    assert 'api_key: "k"' in text


def test_first_blocklist_add_creates_key(tmp_path):
    p = _write(tmp_path, TEMPLATE)
    entries, err = edit_lists(p, list_name="blocklist", add=["tracker.example"])
    assert err is None and entries == ["tracker.example"]
    text = p.read_text()
    assert "blocklist:" in text and "- tracker.example" in text
    # allowlist untouched
    assert "- example.org" in text


def test_invalid_entry_refused(tmp_path):
    p = _write(tmp_path, TEMPLATE)
    before = p.read_text()
    entries, err = edit_lists(p, list_name="allowlist", add=["rm -rf /"])
    assert entries is None and "does not look like a domain" in err
    assert p.read_text() == before  # untouched on refusal


def test_file_sourced_list_refused(tmp_path):
    p = _write(tmp_path, "filter:\n  allowlist_file: /etc/x.txt\n")
    entries, err = edit_lists(p, list_name="allowlist", add=["example.org"])
    assert entries is None and "allowlist_file" in err


def test_flow_style_refused(tmp_path):
    p = _write(tmp_path, "filter:\n  allowlist: [example.org]\n")
    entries, err = edit_lists(p, list_name="allowlist", add=["noaa.gov"])
    assert entries is None and "inline/flow" in err


def test_no_filter_block_appends_one(tmp_path):
    p = _write(tmp_path, "services:\n  secrets: {}\n")
    entries, err = edit_lists(p, list_name="allowlist", add=["example.org"])
    assert err is None and entries == ["example.org"]
    assert "filter:" in p.read_text()
