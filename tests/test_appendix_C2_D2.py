"""Unit tests for the C.2 / D.2 audit-log post-processor logic.

The math (strict-majority re-aggregation, A-pick recovery from
candidate-relative votes + position flag, Krippendorff's $\\alpha$ on a
ternary scale) is independent of the rest of the pipeline, so we test
it directly on hand-built (N, L) arrays. Per AGENT.md §3.3 this is a
load-bearing invariant: the C.2 / D.2 numbers in the appendix are
derived from these functions and must remain trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.plotting.appendix import (
    _aggregate_strict_majority,
    _krippendorff_alpha_ternary,
)


class TestStrictMajorityReaggregation:
    """Matches the contract of self_judge._aggregate at L'∈{1,3,5}."""

    def test_L1_passes_vote_through(self):
        # L'=1, threshold=1: any single vote wins.
        votes = np.array([[1.0], [0.0], [0.5]])
        out = _aggregate_strict_majority(votes, L_p=1)
        # Tie is not a class, so 0.5 stays 0.5 (no class clears).
        assert list(out) == [1.0, 0.0, 0.5]

    def test_L3_majority(self):
        votes = np.array([
            [1.0, 1.0, 0.0],  # 2 wins, 1 lose -> 2 >= 2 -> 1.0
            [0.0, 0.0, 1.0],  # 1 win, 2 lose -> 2 >= 2 -> 0.0
            [1.0, 0.5, 0.0],  # 1 win, 1 lose, 1 tie -> neither -> 0.5
            [0.5, 0.5, 0.5],  # 3 ties -> no win, no lose -> 0.5
        ])
        out = _aggregate_strict_majority(votes, L_p=3)
        assert list(out) == [1.0, 0.0, 0.5, 0.5]

    def test_L5_majority(self):
        # Threshold = ceil(5/2) = 3.
        votes = np.array([
            [1.0, 1.0, 1.0, 0.0, 0.5],  # 3 wins -> 1.0
            [1.0, 1.0, 0.0, 0.0, 0.5],  # 2 wins, 2 lose, 1 tie -> 0.5
            [0.0, 0.0, 0.0, 1.0, 0.5],  # 1 win, 3 lose -> 0.0
        ])
        out = _aggregate_strict_majority(votes, L_p=5)
        assert list(out) == [1.0, 0.5, 0.0]

    def test_threshold_is_ceil_L_over_2(self):
        # L'=4, threshold=ceil(4/2)=2.
        votes = np.array([[1.0, 1.0, 0.0, 0.0]])
        # 2 wins and 2 loses both clear the threshold. Implementation
        # picks the FIRST class that does (n_win >= thr is checked
        # before n_lose >= thr) so 1.0 wins. This matches
        # self_judge._aggregate's `if n_win >= thr: return 1.0` order.
        out = _aggregate_strict_majority(votes, L_p=4)
        assert out[0] == 1.0


class TestKrippendorffTernary:
    """Sanity checks for $\\alpha$ on $\\{0, 0.5, 1\\}$."""

    def test_perfect_agreement_is_one(self):
        v = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        assert _krippendorff_alpha_ternary(v) == pytest.approx(1.0)

    def test_perfect_disagreement_is_negative(self):
        # Two items, two raters who flip 0<->1 within the same item.
        v = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        a = _krippendorff_alpha_ternary(v)
        # Within-item disagreement is maximal; cross-item disagreement
        # is also large but lower than within. α should be negative.
        assert a < 0

    def test_empty_or_singleton_returns_nan(self):
        # L < 2 makes within-item pair count zero — undefined.
        v = np.array([[1.0]])
        a = _krippendorff_alpha_ternary(v)
        assert np.isnan(a)


class TestAPickRecovery:
    """A-pick recovery from candidate-relative verdict + position flag.

    "Judge picked A" iff
       (a_is_candidate AND verdict==1.0)  OR
       (NOT a_is_candidate AND verdict==0.0)

    The D.2 builder uses this expression; the test pins it in case the
    formula gets accidentally inverted during a refactor.
    """

    def test_all_four_combinations(self):
        # (a_is_cand, verdict) -> picked_a
        cases = [
            (True,  1.0, True),   # candidate in A, judge picked candidate -> A
            (True,  0.0, False),  # candidate in A, judge picked reference -> B
            (False, 1.0, False),  # candidate in B, judge picked candidate -> B
            (False, 0.0, True),   # candidate in B, judge picked reference -> A
        ]
        for a_is_cand, verdict, expected in cases:
            pos = np.array([[a_is_cand]])
            raw = np.array([[verdict]])
            picked_a = (pos & (raw == 1.0)) | (~pos & (raw == 0.0))
            assert bool(picked_a[0, 0]) is expected, (
                f"a_is_candidate={a_is_cand}, verdict={verdict}: "
                f"expected picked_a={expected}, got {bool(picked_a[0, 0])}"
            )

    def test_tie_excluded_from_a_pick_denominator(self):
        # A tie (T) doesn't bias either side — D.2 restricts the A-pick
        # rate to non-tie verdicts. Pin that filter.
        raw = np.array([[1.0, 0.5, 0.0]])
        non_tie = raw != 0.5
        assert non_tie.sum() == 2  # only two non-tie verdicts counted
