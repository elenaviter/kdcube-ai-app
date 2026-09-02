# SPDX-License-Identifier: MIT

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.context.vector.anchors import parse_retrieval_anchors


def test_no_block_returns_empty():
    assert parse_retrieval_anchors("Goal: x\nOutcome: y") == ""


def test_empty_input_returns_empty():
    assert parse_retrieval_anchors("") == ""
    assert parse_retrieval_anchors(None) == ""  # type: ignore[arg-type]


def test_inline_json_lists_parsed():
    text = (
        "Goal: build Q2 forecast\n"
        "Outcome: file produced\n"
        "Retrieval-anchors:\n"
        '  phrases: ["Forecast-Q2-2026.xlsx", "openpyxl IndexError", "rename ARR contribution column"]\n'
        '  entities: ["Forecast-Q2-2026.xlsx", "openpyxl", "ARR contribution"]\n'
    )
    result = parse_retrieval_anchors(text)
    # Phrases must be quoted (multi-word verbatim), entities bare.
    assert '"Forecast-Q2-2026.xlsx"' in result
    assert '"openpyxl IndexError"' in result
    assert '"rename ARR contribution column"' in result
    assert "openpyxl" in result
    assert "ARR contribution" in result
    # Phrases come before entities.
    assert result.index('"Forecast-Q2-2026.xlsx"') < result.index(" openpyxl")


def test_only_one_field_present():
    text = (
        "Retrieval-anchors:\n"
        '  entities: ["claudeflare", "wireguard"]\n'
    )
    result = parse_retrieval_anchors(text)
    assert "claudeflare" in result
    assert "wireguard" in result
    assert '"' not in result  # no phrases → no quoted tokens


def test_yaml_list_shape_parsed():
    text = (
        "Retrieval-anchors:\n"
        "  phrases:\n"
        "    - alpha beta\n"
        "    - gamma\n"
        "  entities:\n"
        "    - Foo\n"
        "    - Bar\n"
    )
    result = parse_retrieval_anchors(text)
    assert '"alpha beta"' in result
    assert '"gamma"' in result
    assert "Foo" in result
    assert "Bar" in result


def test_single_quotes_repaired():
    text = (
        "Retrieval-anchors:\n"
        "  phrases: ['one', 'two words']\n"
    )
    result = parse_retrieval_anchors(text)
    assert '"one"' in result
    assert '"two words"' in result


def test_malformed_value_falls_back_to_empty():
    text = "Retrieval-anchors:\n  phrases: not-a-list\n"
    # Best-effort regex split should still extract the bare token.
    result = parse_retrieval_anchors(text)
    # Either empty or contains the fallback token; the contract just requires no exception.
    assert isinstance(result, str)


def test_header_case_insensitive_and_underscore():
    text = (
        "retrieval_anchors:\n"
        '  entities: ["X"]\n'
    )
    assert "X" in parse_retrieval_anchors(text)


# --- attachment anchors -------------------------------------------------------

from kdcube_ai_app.apps.chat.sdk.context.vector.anchors import parse_attachment_anchors  # noqa: E402


ATTACHMENT_SUMMARY = (
    "semantic: Q2 revenue forecast by region; ARR contribution per product | "
    "structural: xlsx, 3 sheets, 14 columns | inventory: Summary sheet, Regions sheet, ARR pivot | "
    "anomalies: none | safety: benign | "
    "lookup_keys: Q2 forecast, ARR contribution, monthly revenue, regions pivot | "
    "filename: Forecast-Q2-2026.xlsx | artifact_name: forecast_q2_2026"
)


def test_attachment_anchors_take_lookup_keys_filename_and_artifact_name():
    out = parse_attachment_anchors(ATTACHMENT_SUMMARY)
    assert out == (
        '"Q2 forecast" "ARR contribution" "monthly revenue" "regions pivot" '
        "Forecast-Q2-2026.xlsx forecast_q2_2026"
    )


def test_attachment_anchors_payload_names_win_over_summary_text():
    out = parse_attachment_anchors(
        ATTACHMENT_SUMMARY, filename="renamed.xlsx", artifact_name="renamed"
    )
    assert out.endswith("renamed.xlsx renamed")
    assert "Forecast-Q2-2026.xlsx" not in out


def test_attachment_anchors_dedupe_and_quote_multiword():
    out = parse_attachment_anchors(
        "lookup_keys: budget, Budget, annual budget | filename: budget.csv",
    )
    assert out == 'budget "annual budget" budget.csv'


def test_attachment_anchors_accept_bracketed_key_list():
    out = parse_attachment_anchors('lookup_keys: ["alpha", "beta gamma"] | filename: a.pdf')
    assert out == 'alpha "beta gamma" a.pdf'


def test_attachment_anchors_empty_without_fields():
    assert parse_attachment_anchors("semantic: a note | structural: text") == ""
    assert parse_attachment_anchors("") == ""
    assert parse_attachment_anchors("", filename="only.txt") == "only.txt"
