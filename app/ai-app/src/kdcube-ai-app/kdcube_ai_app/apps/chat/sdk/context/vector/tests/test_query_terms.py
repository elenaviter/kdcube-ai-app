# SPDX-License-Identifier: MIT

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.context.vector.query_terms import (
    lexical_or_fallback_query,
    trigram_query_tokens,
)


# --- trigram tokens ---------------------------------------------------------

def test_trigram_tokens_drop_recall_framing_and_stop_words():
    tokens = trigram_query_tokens("last time i worked with excel on the forecast")
    assert tokens == ["excel", "forecast"]


def test_trigram_tokens_keep_filenames_and_identifiers():
    tokens = trigram_query_tokens('"Forecast-Q2-2026.xlsx" openpyxl IndexError')
    assert tokens == ["Forecast", "2026", "xlsx", "openpyxl", "IndexError"]


def test_trigram_tokens_fall_back_to_raw_when_everything_is_filler():
    # A query made only of stop-listed words still searches something rather
    # than returning nothing for the trigram arm to average over.
    assert trigram_query_tokens("what did we discuss before") == ["what", "did", "discuss", "before"]


def test_trigram_tokens_drop_or_operator():
    assert trigram_query_tokens("spreadsheet OR xlsx") == ["spreadsheet", "xlsx"]


def test_trigram_tokens_empty():
    assert trigram_query_tokens("") == []
    assert trigram_query_tokens("a an") == []


# --- lexical OR fallback ------------------------------------------------------

def test_or_fallback_rewrites_conversational_query():
    assert lexical_or_fallback_query("last time i worked with excel on the forecast") == "excel OR forecast"


def test_or_fallback_keeps_quoted_phrases_as_units():
    out = lexical_or_fallback_query('"openpyxl IndexError" forecast spreadsheet')
    assert out == '"openpyxl IndexError" OR forecast OR spreadsheet'


def test_or_fallback_dedupes_case_insensitively():
    assert lexical_or_fallback_query("Excel excel forecast") == "Excel OR forecast"


def test_or_fallback_needs_two_units():
    assert lexical_or_fallback_query("excel") == ""
    assert lexical_or_fallback_query("last time excel") == ""
    assert lexical_or_fallback_query("") == ""


def test_or_fallback_skips_queries_that_already_use_or():
    assert lexical_or_fallback_query("spreadsheet OR xlsx") == ""


def test_or_fallback_drops_exclusions():
    # A negation binds to its neighbour in websearch syntax; an OR chain would
    # misplace it, so the fallback carries only the positive units.
    assert lexical_or_fallback_query("forecast spreadsheet -draft") == "forecast OR spreadsheet"


def test_or_fallback_strips_punctuation_but_keeps_inner_dots_and_dashes():
    assert lexical_or_fallback_query("Forecast-Q2-2026.xlsx, openpyxl.") == "Forecast-Q2-2026.xlsx OR openpyxl"
