"""Unit tests for the MATH verifier.

Covers:
- ``\\boxed{}`` extraction with balanced-brace handling and last-match-wins
- Symbolic equivalence delegated to ``math-verify`` (LaTeX/decimal/radical
  equivalence, algebraic rearrangement, intervals)
- Negative cases: wrong values, missing ``\\boxed{}``, empty inputs
- Timeout / exception robustness (AGENT.md §3.3: SymPy failures must
  surface as ``0.0``, never propagate as exceptions)
- Float-typed return contract
"""

from __future__ import annotations

import pytest

from src.verification.math import _extract_last_boxed, verify


# ---------------------------------------------------------------------------
# Positive cases: verify() returns 1.0
# ---------------------------------------------------------------------------

POSITIVE_CASES = [
    pytest.param("Reasoning... \\boxed{42}", "42", id="basic_integer"),
    pytest.param("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", id="exact_fraction"),
    pytest.param("\\boxed{0.5}", "\\frac{1}{2}", id="decimal_to_fraction"),
    pytest.param("\\boxed{1/2}", "\\frac{1}{2}", id="ascii_fraction_to_latex"),
    pytest.param("\\boxed{\\sqrt{12}}", "2\\sqrt{3}", id="radical_simplification"),
    pytest.param("\\boxed{2\\sqrt{3}}", "\\sqrt{12}", id="radical_simplification_reversed"),
    pytest.param("\\boxed{1.5}", "\\frac{3}{2}", id="decimal_mixed_number"),
    pytest.param("\\boxed{(2,3)}", "(2,3)", id="coordinate_pair"),
    pytest.param("\\boxed{x^2+1}", "x^2 + 1", id="algebraic_whitespace_invariance"),
    pytest.param("\\boxed{x^2+1}", "1+x^2", id="algebraic_commutativity"),
    pytest.param("\\boxed{\\pi}", "\\pi", id="constant_pi"),
    pytest.param("\\boxed{42}", "\\boxed{42}", id="ground_truth_already_boxed"),
    pytest.param("First \\boxed{1}, then \\boxed{2}", "2", id="last_boxed_wins"),
    pytest.param("\\boxed{\\frac{a+b}{c}}", "\\frac{a+b}{c}", id="nested_braces_extraction"),
]


@pytest.mark.parametrize("generation,ground_truth", POSITIVE_CASES)
def test_verify_positive(generation: str, ground_truth: str) -> None:
    assert verify(generation, ground_truth) == 1.0


# ---------------------------------------------------------------------------
# Negative cases: verify() returns 0.0
# ---------------------------------------------------------------------------

NEGATIVE_CASES = [
    pytest.param("\\boxed{42}", "43", id="wrong_integer"),
    pytest.param("\\boxed{\\frac{1}{2}}", "\\frac{1}{3}", id="wrong_fraction"),
    pytest.param("\\boxed{\\sqrt{12}}", "2\\sqrt{2}", id="wrong_radical"),
    pytest.param("No boxed answer here", "42", id="missing_boxed_in_generation"),
    pytest.param("", "42", id="empty_generation"),
    pytest.param("\\boxed{42}", "", id="empty_ground_truth"),
    pytest.param("\\boxed{x+1}", "x-1", id="wrong_sign_in_expression"),
    pytest.param("\\boxed{(2,3)}", "(3,2)", id="swapped_coordinates"),
    pytest.param("\\boxed{0.5}", "0.6", id="wrong_decimal"),
    pytest.param("\\boxed{x^2}", "x^3", id="wrong_exponent"),
    pytest.param("\\boxed{}", "42", id="empty_boxed"),
]


@pytest.mark.parametrize("generation,ground_truth", NEGATIVE_CASES)
def test_verify_negative(generation: str, ground_truth: str) -> None:
    assert verify(generation, ground_truth) == 0.0


# ---------------------------------------------------------------------------
# _extract_last_boxed direct tests (balanced-brace correctness)
# ---------------------------------------------------------------------------


def test_extract_simple() -> None:
    assert _extract_last_boxed("\\boxed{42}") == "42"


def test_extract_nested_braces() -> None:
    assert _extract_last_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_deeply_nested() -> None:
    assert (
        _extract_last_boxed("\\boxed{\\frac{\\sqrt{x+1}}{y^2}}")
        == "\\frac{\\sqrt{x+1}}{y^2}"
    )


def test_extract_last_of_multiple() -> None:
    assert _extract_last_boxed("first \\boxed{1} then \\boxed{2}") == "2"


def test_extract_returns_none_when_absent() -> None:
    assert _extract_last_boxed("no boxed here") is None


def test_extract_returns_none_on_empty_string() -> None:
    assert _extract_last_boxed("") is None


def test_extract_unbalanced_falls_back() -> None:
    # Latest \boxed{ has no closing brace; falls back to earlier balanced one.
    assert _extract_last_boxed("\\boxed{42} then \\boxed{unclosed") == "42"


def test_extract_all_unbalanced_returns_none() -> None:
    assert _extract_last_boxed("\\boxed{unclosed") is None


def test_extract_escaped_brace_does_not_close() -> None:
    # The \} should not be counted as a closing brace.
    assert _extract_last_boxed("\\boxed{\\{x\\}}") == "\\{x\\}"


def test_extract_whitespace_between_command_and_brace() -> None:
    # \boxed  {42} should still be recognised.
    assert _extract_last_boxed("\\boxed  {42}") == "42"


# ---------------------------------------------------------------------------
# Timeout / exception robustness (AGENT.md §3.3)
# ---------------------------------------------------------------------------


def test_verify_returns_zero_on_math_verify_internal_exception(monkeypatch) -> None:
    """If math_verify.verify raises despite raise_on_error=False, we catch
    it and return 0.0 — a single hard prompt must not abort the run."""
    import math_verify

    def boom(*args, **kwargs):
        raise RuntimeError("simulated SymPy hang or library bug")

    monkeypatch.setattr(math_verify, "verify", boom)
    assert verify("\\boxed{42}", "42") == 0.0


def test_verify_returns_zero_on_parse_exception(monkeypatch) -> None:
    """Same defence at the parse layer."""
    import math_verify

    def boom(*args, **kwargs):
        raise RuntimeError("simulated parser failure")

    monkeypatch.setattr(math_verify, "parse", boom)
    assert verify("\\boxed{42}", "42") == 0.0


def test_verify_returns_zero_on_unparseable_garbage() -> None:
    """math-verify returns an empty parse list on garbage; we map that to 0.0."""
    assert verify("\\boxed{!!! @@@ ###}", "42") == 0.0


# ---------------------------------------------------------------------------
# Return-type contract from AGENT.md §3.3
# ---------------------------------------------------------------------------


def test_verify_one_is_float() -> None:
    result = verify("\\boxed{42}", "42")
    assert isinstance(result, float)
    assert result == 1.0


def test_verify_zero_is_float() -> None:
    result = verify("\\boxed{42}", "43")
    assert isinstance(result, float)
    assert result == 0.0
