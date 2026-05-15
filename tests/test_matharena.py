"""Tests for src.verification.matharena."""

from __future__ import annotations

import pytest

from src.verification.matharena import _extract_last_boxed, verify


class TestExtractLastBoxed:
    def test_simple_integer(self) -> None:
        assert _extract_last_boxed("reasoning... \\boxed{70}") == "70"

    def test_latex_expression(self) -> None:
        assert _extract_last_boxed("\\boxed{2^{99}}") == "2^{99}"

    def test_nested_braces(self) -> None:
        assert _extract_last_boxed(
            "answer: \\boxed{\\frac{1}{2}}"
        ) == "\\frac{1}{2}"

    def test_last_one_wins(self) -> None:
        assert _extract_last_boxed(
            "first \\boxed{1} then \\boxed{42}"
        ) == "42"

    def test_no_boxed(self) -> None:
        assert _extract_last_boxed("just plain text") is None


class TestVerify:
    def test_integer_match(self) -> None:
        assert verify("Therefore \\boxed{70}", "70") == 1.0

    def test_integer_mismatch(self) -> None:
        assert verify("Therefore \\boxed{42}", "70") == 0.0

    def test_symbolic_equivalence_latex(self) -> None:
        # Both sides in LaTeX form — same expression, different spacing.
        assert verify("\\boxed{2^{99}}", "2^{99}") == 1.0
        # Different formatting of the same algebraic value.
        assert verify("\\boxed{2x + 3}", "3 + 2x") == 1.0

    def test_fraction_decimal_equivalence(self) -> None:
        assert verify("\\boxed{1/2}", "0.5") == 1.0

    def test_no_boxed_returns_zero(self) -> None:
        assert verify("the answer is 70", "70") == 0.0

    def test_empty_ground_truth(self) -> None:
        assert verify("\\boxed{70}", "") == 0.0

    def test_unparseable_falls_back_to_string_match(self) -> None:
        # Both sides are exact strings -> verify wins; or via string fallback.
        assert verify("\\boxed{foo}", "foo") == 1.0
        assert verify("\\boxed{foo}", "bar") == 0.0
