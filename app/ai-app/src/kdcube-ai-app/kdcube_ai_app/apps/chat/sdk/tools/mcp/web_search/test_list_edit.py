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


def test_commented_file_source_does_not_refuse(tmp_path):
    """Regression: the template ships a commented-out allowlist_file
    line, and a substring check mistook it for a real file source."""
    p = _write(
        tmp_path,
        "filter:\n"
        "  allowlist:\n    - example.org\n"
        "  # allowlist_file: /etc/claude/web-allowlist.txt\n",
    )
    entries, err = edit_lists(p, list_name="allowlist", add=["noaa.gov"])
    assert err is None
    assert entries == ["example.org", "noaa.gov"]
    assert "# allowlist_file" in p.read_text()  # comment preserved


def test_flow_style_refused(tmp_path):
    p = _write(tmp_path, "filter:\n  allowlist: [example.org]\n")
    entries, err = edit_lists(p, list_name="allowlist", add=["noaa.gov"])
    assert entries is None and "inline/flow" in err


def test_no_filter_block_appends_one(tmp_path):
    p = _write(tmp_path, "services:\n  secrets: {}\n")
    entries, err = edit_lists(p, list_name="allowlist", add=["example.org"])
    assert err is None and entries == ["example.org"]
    assert "filter:" in p.read_text()


def test_nested_config_section_is_edited_without_touching_agent_config(tmp_path):
    p = _write(
        tmp_path,
        "agent:\n"
        "  topic: keep-this\n"
        "web_search:\n"
        "  filter:\n"
        "    allowlist:\n"
        "      - python.org\n"
        "output:\n"
        "  directory: ./output\n",
    )

    entries, err = edit_lists(
        p,
        list_name="allowlist",
        add=["docs.python.org"],
        config_section="web_search",
    )

    assert err is None
    assert entries == ["python.org", "docs.python.org"]
    text = p.read_text()
    assert "  topic: keep-this" in text
    assert "      - docs.python.org" in text
    assert "  directory: ./output" in text


def test_exact_agent_tool_settings_are_edited_without_touching_other_tools(tmp_path):
    p = _write(
        tmp_path,
        "agent:\n"
        "  topic: keep-this\n"
        "  tools:\n"
        "    - id: other.search\n"
        "      settings:\n"
        "        filter:\n"
        "          allowlist:\n"
        "            - wrong.example\n"
        "    - id: demo.web_search\n"
        "      enabled: true\n"
        "      settings:\n"
        "        filter:\n"
        "          allowlist:\n"
        "            - python.org\n"
        "          ssrf_guard: true\n"
        "  run_directory: ./output\n",
    )

    entries, err = edit_lists(
        p,
        list_name="allowlist",
        add=["docs.python.org"],
        config_tool_id="demo.web_search",
    )

    assert err is None
    assert entries == ["python.org", "docs.python.org"]
    text = p.read_text()
    assert text.count("- wrong.example") == 1
    assert text.count("- docs.python.org") == 1
    assert "  topic: keep-this" in text
    assert "  run_directory: ./output" in text


def test_exact_agent_tool_editor_can_create_settings_and_filter(tmp_path):
    p = _write(
        tmp_path,
        "agent:\n"
        "  tools:\n"
        "    - id: demo.web_search\n"
        "      enabled: true\n"
        "    - id: other.tool\n"
        "      enabled: true\n",
    )

    entries, err = edit_lists(
        p,
        list_name="allowlist",
        add=["python.org"],
        config_tool_id="demo.web_search",
    )

    assert err is None
    assert entries == ["python.org"]
    assert (
        "    - id: demo.web_search\n"
        "      enabled: true\n"
        "      settings:\n"
        "        filter:\n"
        "          allowlist:\n"
        "            - python.org\n"
        "    - id: other.tool\n"
    ) in p.read_text()


def test_exact_agent_tool_editor_rejects_missing_duplicate_and_mixed_selector(tmp_path):
    p = _write(
        tmp_path,
        "agent:\n"
        "  tools:\n"
        "    - id: duplicate.search\n"
        "      settings: {}\n"
        "    - id: duplicate.search\n"
        "      settings: {}\n",
    )

    entries, err = edit_lists(
        p,
        list_name="allowlist",
        add=["python.org"],
        config_tool_id="missing.search",
    )
    assert entries is None and "no block-style agent tool" in err

    entries, err = edit_lists(
        p,
        list_name="allowlist",
        add=["python.org"],
        config_tool_id="duplicate.search",
    )
    assert entries is None and "duplicate agent tool" in err

    entries, err = edit_lists(
        p,
        list_name="allowlist",
        add=["python.org"],
        config_section="agent",
        config_tool_id="duplicate.search",
    )
    assert entries is None and "mutually exclusive" in err
