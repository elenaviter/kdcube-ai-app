# SPDX-License-Identifier: MIT

import pytest

from kdcube_ai_app.apps.chat.sdk.tools.citations import (
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
