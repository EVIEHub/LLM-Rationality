"""Unit tests for the GSM8K verifier.

Covers the four documented extraction formats (``####``, ``\\boxed{}``,
"the answer is X", standalone last number), priority ordering between
them, numeric normalisation (commas, currency, percent, decimals, signs),
and the float-typed return contract from AGENT.md §3.3.
"""

from __future__ import annotations

import pytest

from src.verification.gsm8k import extract_answer, verify


# ---------------------------------------------------------------------------
# Positive cases: verify() returns 1.0
# ---------------------------------------------------------------------------

POSITIVE_CASES = [
    pytest.param(
        "Janet's ducks lay 16 eggs per day...\n#### 18",
        "Janet's ducks lay 16 eggs per day...\n#### 18",
        id="hash_with_explanation_both_sides",
    ),
    pytest.param("So we get \\boxed{42}.", "42", id="boxed"),
    pytest.param("After computation, the answer is 100.", "100", id="answer_is"),
    pytest.param("After computing, we find 7.", "7", id="bare_last_number"),
    pytest.param("#### 1,234", "1234", id="commas_in_generation"),
    pytest.param("Working: \\boxed{1,234.56}", "1234.56", id="boxed_with_commas_decimal"),
    pytest.param("She paid $50.", "50", id="currency_prefix"),
    pytest.param("#### 0.5", "0.5", id="decimal"),
    pytest.param("Final: -10.", "-10", id="negative"),
    pytest.param("First #### 5\nthen #### 10", "10", id="last_hash_wins"),
    pytest.param("#### 72", "#### 72", id="ground_truth_with_hash"),
    pytest.param("#### 72.0", "72", id="decimal_int_match"),
    pytest.param("####    72   ", "72", id="whitespace_around_hash"),
    pytest.param("\\boxed{1} then \\boxed{2}", "2", id="last_boxed_wins"),
]


@pytest.mark.parametrize("generation,ground_truth", POSITIVE_CASES)
def test_verify_positive(generation: str, ground_truth: str) -> None:
    assert verify(generation, ground_truth) == 1.0


# ---------------------------------------------------------------------------
# Negative cases: verify() returns 0.0
# ---------------------------------------------------------------------------

NEGATIVE_CASES = [
    pytest.param("#### 72", "73", id="wrong_number"),
    pytest.param("#### 0.5", "0.6", id="wrong_decimal"),
    pytest.param("I have no idea.", "42", id="no_number_in_generation"),
    pytest.param("#### 42", "unknown", id="no_number_in_ground_truth"),
    pytest.param("", "42", id="empty_generation"),
    pytest.param("#### 42", "", id="empty_ground_truth"),
    pytest.param("#### -10", "10", id="wrong_sign"),
    pytest.param("#### 100", "1000", id="off_by_magnitude"),
    pytest.param("The answer is forty-two.", "42", id="word_answer_not_extracted"),
    pytest.param("Reasoning ending in 5.", "10", id="bare_number_wrong"),
    pytest.param("\\boxed{1/2}", "0.5", id="fraction_not_supported_here"),
]


@pytest.mark.parametrize("generation,ground_truth", NEGATIVE_CASES)
def test_verify_negative(generation: str, ground_truth: str) -> None:
    assert verify(generation, ground_truth) == 0.0


# ---------------------------------------------------------------------------
# Priority ordering between the four extractor patterns
# ---------------------------------------------------------------------------


def test_priority_hash_beats_boxed() -> None:
    # When both #### and \boxed{} appear, #### takes priority.
    assert extract_answer("\\boxed{72}\nbut also #### 100") == 100.0


def test_priority_boxed_beats_answer_is() -> None:
    assert extract_answer("the answer is 5 but \\boxed{42}") == 42.0


def test_priority_answer_is_beats_bare() -> None:
    assert extract_answer("the answer is 42, then we noted 99") == 42.0


def test_priority_bare_used_only_when_others_absent() -> None:
    assert extract_answer("After working through, we get 7") == 7.0


# ---------------------------------------------------------------------------
# extract_answer edge cases
# ---------------------------------------------------------------------------


def test_extract_returns_none_on_no_number() -> None:
    assert extract_answer("no numbers anywhere here") is None


def test_extract_returns_none_on_empty_string() -> None:
    assert extract_answer("") is None


def test_extract_negative_decimal() -> None:
    assert extract_answer("Final: -3.14") == -3.14


def test_extract_currency_with_commas() -> None:
    assert extract_answer("Total: $12,345.67") == 12345.67


def test_extract_large_integer_no_commas() -> None:
    # Numbers without comma grouping must still parse.
    assert extract_answer("#### 1234567") == 1234567.0


# ---------------------------------------------------------------------------
# Return-type contract from AGENT.md §3.3
# ---------------------------------------------------------------------------


def test_extract_answer_returns_float_type() -> None:
    result = extract_answer("#### 42")
    assert isinstance(result, float)
    assert result == 42.0


def test_verify_one_is_float() -> None:
    result = verify("#### 72", "72")
    assert isinstance(result, float)
    assert result == 1.0


def test_verify_zero_is_float() -> None:
    result = verify("#### 72", "73")
    assert isinstance(result, float)
    assert result == 0.0
