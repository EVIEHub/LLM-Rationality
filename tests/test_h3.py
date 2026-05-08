"""Tests for src.metrics.h3.

H3 cells reuse :func:`src.metrics.rational_gap.compute_rational_gap` for
the actual rational-gap computation; that path is exercised by
``test_rational_gap.py``. The H3 module itself adds dataset-specific
answer-key extractors and the bootstrap-resampled SC utility matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.h3 import (
    _extract_gsm8k_key,
    _extract_math_key,
    extract_answer_key,
    self_consistency_utility_matrix,
    supports_answer_key_extraction,
)


# ---------------------------------------------------------------------------
# GSM8K answer-key extractor
# ---------------------------------------------------------------------------

class TestGSM8KKeyExtractor:
    def test_hashed_marker(self) -> None:
        assert _extract_gsm8k_key("Long reasoning... #### 42") == 42.0

    def test_boxed(self) -> None:
        assert _extract_gsm8k_key("Therefore \\boxed{18}.") == 18.0

    def test_natural_language(self) -> None:
        assert _extract_gsm8k_key("the answer is 7") == 7.0

    def test_last_bare_number(self) -> None:
        assert _extract_gsm8k_key("first 12 then 9 then 5") == 5.0

    def test_no_number(self) -> None:
        assert _extract_gsm8k_key("the cat sat on the mat") is None

    def test_42_and_42_dot_0_share_key(self) -> None:
        # Numeric equality means SC will count "42" and "42.0" as agreeing.
        assert _extract_gsm8k_key("#### 42") == _extract_gsm8k_key("#### 42.0")


# ---------------------------------------------------------------------------
# MATH answer-key extractor
# ---------------------------------------------------------------------------

class TestMathKeyExtractor:
    def test_simple_boxed(self) -> None:
        assert _extract_math_key("answer: \\boxed{17}") == "17"

    def test_collapses_whitespace(self) -> None:
        # "\\frac{1} {2}" and "\\frac{1}{2}" should map to the same key.
        a = _extract_math_key("\\boxed{\\frac{1} {2}}")
        b = _extract_math_key("\\boxed{\\frac{1}{2}}")
        assert a == b == "\\frac{1}{2}"

    def test_no_boxed(self) -> None:
        assert _extract_math_key("just plain text") is None

    def test_takes_last_boxed(self) -> None:
        # When a model writes intermediate boxed values, only the last
        # one is the final answer (matches verifier semantics).
        assert _extract_math_key(
            "first \\boxed{1} then \\boxed{2}"
        ) == "2"


# ---------------------------------------------------------------------------
# supports_answer_key_extraction + extract_answer_key dispatcher
# ---------------------------------------------------------------------------

class TestSupports:
    def test_gsm8k_supported(self) -> None:
        assert supports_answer_key_extraction("gsm8k")
        assert supports_answer_key_extraction("GSM8K")
        assert supports_answer_key_extraction(" gsm8k ")

    def test_math_supported(self) -> None:
        assert supports_answer_key_extraction("math")

    def test_humaneval_unsupported(self) -> None:
        assert not supports_answer_key_extraction("humaneval")


class TestDispatcher:
    def test_dispatches_to_gsm8k(self) -> None:
        assert extract_answer_key("gsm8k", "#### 42") == 42.0

    def test_dispatches_to_math(self) -> None:
        assert extract_answer_key("math", "\\boxed{x^2}") == "x^2"

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(KeyError, match="humaneval"):
            extract_answer_key("humaneval", "x")


# ---------------------------------------------------------------------------
# self_consistency_utility_matrix
# ---------------------------------------------------------------------------

class TestSCUtilityMatrix:
    def test_unanimous_correct_gives_all_ones(self) -> None:
        # Every cached sample agrees on the correct answer -> every SC
        # bootstrap draw picks that answer -> every entry is 1.0.
        samples = [
            ["#### 42"] * 4,   # K_h1=4, all "42"
            ["#### 7"] * 4,    # K_h1=4, all "7"
        ]
        util_h1 = np.array([
            [1.0, 1.0, 1.0, 1.0],   # all four samples correct for prompt 0
            [1.0, 1.0, 1.0, 1.0],   # all four samples correct for prompt 1
        ])
        out = self_consistency_utility_matrix(
            util_h1, samples, "gsm8k", n=2, K=8, seed=0,
        )
        assert out.shape == (2, 8)
        assert (out == 1.0).all()

    def test_majority_wrong_gives_zero(self) -> None:
        # 3 wrong samples, 1 correct -> bootstrap draws of n=2 with high
        # probability will mostly mode on the wrong answer, picking a
        # wrong rep and scoring 0.
        samples = [
            ["#### 7", "#### 7", "#### 7", "#### 42"],
        ]
        # util reflects "wrong, wrong, wrong, correct" against gt 42:
        util_h1 = np.array([[0.0, 0.0, 0.0, 1.0]])
        out = self_consistency_utility_matrix(
            util_h1, samples, "gsm8k", n=4, K=64, seed=0,
        )
        # With n=4, bootstrap draws of 4 indices from 4 columns produce
        # mostly-wrong votes; SC mode is "7" most of the time -> rep is
        # one of the wrong samples -> utility 0.
        assert out.shape == (1, 64)
        # The mean should be close to 0 (heavily skewed).
        assert out.mean() < 0.5

    def test_deterministic_under_seed(self) -> None:
        samples = [
            ["#### 1", "#### 1", "#### 2", "#### 2"],
        ]
        util_h1 = np.array([[1.0, 1.0, 0.0, 0.0]])
        a = self_consistency_utility_matrix(
            util_h1, samples, "gsm8k", n=2, K=16, seed=42,
        )
        b = self_consistency_utility_matrix(
            util_h1, samples, "gsm8k", n=2, K=16, seed=42,
        )
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_matrices(self) -> None:
        samples = [
            ["#### 1", "#### 1", "#### 2", "#### 2"],
        ]
        util_h1 = np.array([[1.0, 1.0, 0.0, 0.0]])
        a = self_consistency_utility_matrix(
            util_h1, samples, "gsm8k", n=2, K=64, seed=0,
        )
        b = self_consistency_utility_matrix(
            util_h1, samples, "gsm8k", n=2, K=64, seed=1,
        )
        # Should not be identical (very unlikely with K=64 random draws).
        assert not np.array_equal(a, b)

    def test_no_parseable_key_returns_zero(self) -> None:
        # GSM8K extractor's "last bare number" fallback means anything
        # with a digit will parse — use truly digit-free strings here.
        samples = [["the cat sat", "the cat sat", "the cat sat"]]
        util_h1 = np.array([[0.0, 0.0, 0.0]])
        out = self_consistency_utility_matrix(
            util_h1, samples, "gsm8k", n=2, K=4, seed=0,
        )
        assert out.shape == (1, 4)
        assert (out == 0.0).all()

    def test_unsupported_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="self-consistency"):
            self_consistency_utility_matrix(
                np.zeros((1, 1)), [["a"]], "humaneval", n=1, K=1, seed=0,
            )

    def test_bad_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            self_consistency_utility_matrix(
                np.zeros((1, 1)), [["#### 1"]], "gsm8k", n=0, K=1, seed=0,
            )

    def test_bad_K_raises(self) -> None:
        with pytest.raises(ValueError, match="K must be"):
            self_consistency_utility_matrix(
                np.zeros((1, 1)), [["#### 1"]], "gsm8k", n=1, K=0, seed=0,
            )

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            self_consistency_utility_matrix(
                np.zeros((2, 1)), [["#### 1"]], "gsm8k", n=1, K=1, seed=0,
            )
        with pytest.raises(ValueError, match="samples"):
            self_consistency_utility_matrix(
                np.zeros((1, 4)),
                [["#### 1", "#### 2", "#### 3"]],  # 3 != 4
                "gsm8k", n=1, K=1, seed=0,
            )
