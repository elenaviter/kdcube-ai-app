# SPDX-License-Identifier: MIT

import pytest

from kdcube_ai_app.apps.chat.sdk.tools.citations import (
    CitationStreamState,
    replace_citation_tokens_streaming_stateful,
    split_safe_citation_prefix,
    split_safe_stream_prefix,
)


@pytest.mark.parametrize(
    "chunk",
    [
        "[",  # lone open bracket at end
        "[[",
        "[[S",
        "[[ S",
        "[[S:",
        "[[ S :",
        "[[S:1, 2-",
        "[[S:1, 2-3]",
        "\u200b[[",  # ZWSP + token start
        " \u200b[[S:",
        "\u200b[S",  # ZWSP between brackets can still split into malformed start
    ],
)
def test_split_safe_citation_prefix_holds_back_partial_tokens(chunk):
    safe, dangling = split_safe_citation_prefix(chunk)
    assert safe == ""
    assert dangling == len(chunk)


@pytest.mark.parametrize(
    "chunk, expected_safe",
    [
        ("Hello [[S:1]] world", "Hello [[S:1]] world"),
        ("[[S:1]]", "[[S:1]]"),
        ("No citations here", "No citations here"),
    ],
)
def test_split_safe_citation_prefix_allows_complete_chunks(chunk, expected_safe):
    safe, dangling = split_safe_citation_prefix(chunk)
    assert safe == expected_safe
    assert dangling == 0


@pytest.mark.parametrize(
    "chunk, expected_safe",
    [
        ("prefix [", "prefix "),
        ("prefix [[S", "prefix "),
        ("prefix [[S:", "prefix "),
        ("prefix [[S:1,", "prefix "),
        ("prefix [[S:1,2-", "prefix "),
    ],
)
def test_split_safe_stream_prefix_holds_back_suffix_tokens(chunk, expected_safe):
    safe, dangling = split_safe_stream_prefix(chunk)
    assert safe == expected_safe
    assert dangling == (len(chunk) - len(expected_safe))


def test_streaming_replacement_handles_all_token_splits():
    token = "[[S:1]]"
    prefix = "Start "
    suffix = " End"
    citation_map = {1: {"url": "https://example.com/only", "title": "Only Source"}}

    def _run_chunks(chunks: list[str]) -> str:
        state = CitationStreamState()
        out = []
        for ch in chunks:
            out.append(replace_citation_tokens_streaming_stateful(ch, citation_map, state))
        out.append(replace_citation_tokens_streaming_stateful("", citation_map, state, flush=True))
        return "".join(out)

    # All 2-piece splits
    for i in range(1, len(token)):
        chunks = [prefix + token[:i], token[i:] + suffix]
        rendered = _run_chunks(chunks)
        assert "[[S:" not in rendered
        assert "https://example.com/only" in rendered

    # All 3-piece splits
    for i in range(1, len(token) - 1):
        for j in range(i + 1, len(token)):
            chunks = [prefix + token[:i], token[i:j], token[j:] + suffix]
            rendered = _run_chunks(chunks)
            assert "[[S:" not in rendered
            assert "https://example.com/only" in rendered
