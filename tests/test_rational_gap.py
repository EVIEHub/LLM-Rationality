"""Unit tests for rational-gap metrics.

Covers the closed-form invariants ($\\hat{\\mathcal{R}}_1 \\equiv 0$,
all-zeros and all-ones cases, "$1$ of $K$ passes" yielding $R = 1 - 1/K$),
shape and dtype handling, bootstrap CI determinism + behaviour on
degenerate inputs, and the saturation-curve helpers ``U_circ_at_K`` /
``U_bar_at_K`` / ``R_hat_at_K`` that build the H1 curve by direct
truncation of the $(M, n_\\text{max})$ utility matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.rational_gap import (
    BootstrapCI,
    R_hat_at_K,
    RationalGapEstimate,
    U_bar_at_K,
    U_circ_at_K,
    bootstrap_ci_over_prompts,
    compute_rational_gap,
)


# ---------------------------------------------------------------------------
# Closed-form invariants
# ---------------------------------------------------------------------------


def test_R_hat_K_is_zero_when_K_equals_one() -> None:
    """Paper invariant: $\\hat{\\mathcal{R}}_1 \\equiv 0$ — max of one
    sample equals the mean of one sample."""
    utilities = np.array([[0.0], [1.0], [1.0], [0.0]])  # M=4, K=1
    est = compute_rational_gap(utilities)
    assert est.R_hat_K == 0.0
    assert est.U_circ_K == est.U_bar_K == 0.5


def test_all_zeros_gives_zero_gap() -> None:
    utilities = np.zeros((10, 8))
    est = compute_rational_gap(utilities)
    assert est.U_circ_K == 0.0
    assert est.U_bar_K == 0.0
    assert est.R_hat_K == 0.0


def test_all_ones_gives_zero_gap() -> None:
    """When every sample is correct, max equals mean equals 1; gap is zero."""
    utilities = np.ones((10, 8))
    est = compute_rational_gap(utilities)
    assert est.U_circ_K == 1.0
    assert est.U_bar_K == 1.0
    assert est.R_hat_K == 0.0


def test_one_of_K_passes_per_prompt() -> None:
    """If exactly 1 of K samples passes per prompt, R_hat_K = 1 - 1/K
    (max=1, mean=1/K, gap=1-1/K). This is the "load-bearing" case the
    methodology memo prioritises in audits."""
    M, K = 5, 8
    utilities = np.zeros((M, K))
    utilities[:, 0] = 1.0  # exactly the first sample passes for each prompt
    est = compute_rational_gap(utilities)
    assert est.U_circ_K == 1.0
    assert est.U_bar_K == pytest.approx(1.0 / K)
    assert est.R_hat_K == pytest.approx(1.0 - 1.0 / K)


def test_per_prompt_arrays_match_aggregates() -> None:
    rng = np.random.default_rng(0)
    utilities = rng.integers(0, 2, size=(20, 16)).astype(float)
    est = compute_rational_gap(utilities)
    np.testing.assert_allclose(est.per_prompt_U_circ_K, utilities.max(axis=1))
    np.testing.assert_allclose(est.per_prompt_U_bar_K, utilities.mean(axis=1))
    np.testing.assert_allclose(
        est.per_prompt_R_hat_K, utilities.max(axis=1) - utilities.mean(axis=1)
    )
    assert est.U_circ_K == pytest.approx(est.per_prompt_U_circ_K.mean())
    assert est.R_hat_K == pytest.approx(est.per_prompt_R_hat_K.mean())


def test_continuous_utility_works() -> None:
    """The metrics module is generic over $U$; non-binary utilities should
    work without modification (per AGENT.md §3.3 forward-compatibility)."""
    utilities = np.array([[0.2, 0.7, 0.5], [0.9, 0.1, 0.4]])
    est = compute_rational_gap(utilities)
    np.testing.assert_allclose(est.per_prompt_U_circ_K, [0.7, 0.9])
    np.testing.assert_allclose(est.per_prompt_U_bar_K, [(0.2 + 0.7 + 0.5) / 3, (0.9 + 0.1 + 0.4) / 3])


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_M_and_K_recorded() -> None:
    est = compute_rational_gap(np.zeros((7, 3)))
    assert est.M == 7
    assert est.K == 3


def test_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_rational_gap(np.array([0.0, 1.0, 1.0]))


def test_rejects_3d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_rational_gap(np.zeros((2, 3, 4)))


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_rational_gap(np.zeros((0, 5)))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic_with_same_seed() -> None:
    values = np.array([0.0, 0.5, 0.7, 1.0, 0.3, 0.6])
    a = bootstrap_ci_over_prompts(values, n_resamples=500, seed=42)
    b = bootstrap_ci_over_prompts(values, n_resamples=500, seed=42)
    assert a == b


def test_bootstrap_differs_for_different_seeds() -> None:
    values = np.array([0.0, 0.5, 0.7, 1.0, 0.3, 0.6])
    a = bootstrap_ci_over_prompts(values, n_resamples=500, seed=0)
    b = bootstrap_ci_over_prompts(values, n_resamples=500, seed=1)
    # Means should be close (same underlying distribution) but CI bounds differ.
    assert a.ci_low != b.ci_low or a.ci_high != b.ci_high


def test_bootstrap_mean_close_to_sample_mean() -> None:
    """For a reasonable B, the bootstrap-of-means concentrates near the sample mean."""
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=200)
    ci = bootstrap_ci_over_prompts(values, n_resamples=2000, seed=0)
    assert ci.mean == pytest.approx(values.mean(), abs=0.01)


def test_bootstrap_ci_collapses_for_constant_values() -> None:
    """If every prompt has the same value, every resample has the same mean,
    so the CI collapses to a point."""
    values = np.full(20, 0.42)
    ci = bootstrap_ci_over_prompts(values, n_resamples=100, seed=0)
    assert ci.mean == pytest.approx(0.42)
    assert ci.ci_low == pytest.approx(0.42)
    assert ci.ci_high == pytest.approx(0.42)


def test_bootstrap_ci_brackets_mean_for_ci_inputs() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=100)
    ci = bootstrap_ci_over_prompts(values, n_resamples=1000, seed=0)
    assert ci.ci_low <= ci.mean <= ci.ci_high


def test_bootstrap_ci_default_args_match_methodology_memo() -> None:
    """Methodology memo specifies B = 1000, 95% CI as defaults."""
    values = np.array([0.0, 1.0, 0.5])
    ci = bootstrap_ci_over_prompts(values)
    assert ci.n_resamples == 1000
    assert ci.confidence == 0.95


def test_bootstrap_ci_records_confidence_level() -> None:
    values = np.array([0.0, 1.0, 0.5, 0.3])
    ci = bootstrap_ci_over_prompts(values, confidence=0.90, n_resamples=200)
    assert ci.confidence == 0.90


def test_bootstrap_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1D"):
        bootstrap_ci_over_prompts(np.zeros((10, 3)))


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci_over_prompts(np.array([]))


def test_bootstrap_rejects_invalid_confidence() -> None:
    values = np.array([0.0, 1.0])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_ci_over_prompts(values, confidence=bad)


def test_bootstrap_rejects_zero_resamples() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci_over_prompts(np.array([0.0, 1.0]), n_resamples=0)


# ---------------------------------------------------------------------------
# Integration: compute_rational_gap → bootstrap_ci_over_prompts pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Saturation-curve helpers: U_circ_at_K / U_bar_at_K / R_hat_at_K
#
# These are direct truncation of the (M, n_max) matrix to its first K
# columns — one-line transcriptions of the paper formulas. No binomial
# closed form, no expectation framing.
# ---------------------------------------------------------------------------


_TRUNCATION_FIXTURE = np.array(
    [
        [0, 0, 1, 0],  # row 0: prefix maxes [0, 0, 1, 1]
        [1, 0, 0, 0],  # row 1: prefix maxes [1, 1, 1, 1]
        [0, 0, 0, 0],  # row 2: prefix maxes [0, 0, 0, 0]
    ],
    dtype=float,
)


def test_U_circ_at_K_hand_computed() -> None:
    """M=3, n_max=4 fixture: at K=1..4, U_circ_K is 1/3, 1/3, 2/3, 2/3."""
    assert U_circ_at_K(_TRUNCATION_FIXTURE, 1) == pytest.approx(1 / 3)
    assert U_circ_at_K(_TRUNCATION_FIXTURE, 2) == pytest.approx(1 / 3)
    assert U_circ_at_K(_TRUNCATION_FIXTURE, 3) == pytest.approx(2 / 3)
    assert U_circ_at_K(_TRUNCATION_FIXTURE, 4) == pytest.approx(2 / 3)


def test_U_bar_at_K_hand_computed() -> None:
    """At K=1..4, U_bar_K is 1/3, 1/6, 2/9, 1/6 for the same fixture."""
    assert U_bar_at_K(_TRUNCATION_FIXTURE, 1) == pytest.approx(1 / 3)
    assert U_bar_at_K(_TRUNCATION_FIXTURE, 2) == pytest.approx(1 / 6)
    assert U_bar_at_K(_TRUNCATION_FIXTURE, 3) == pytest.approx(2 / 9)
    assert U_bar_at_K(_TRUNCATION_FIXTURE, 4) == pytest.approx(1 / 6)


def test_R_hat_at_K_hand_computed() -> None:
    """R_hat_K at K=1..4: 0, 1/6, 4/9, 1/2 for the fixture."""
    assert R_hat_at_K(_TRUNCATION_FIXTURE, 1) == pytest.approx(0.0)
    assert R_hat_at_K(_TRUNCATION_FIXTURE, 2) == pytest.approx(1 / 6)
    assert R_hat_at_K(_TRUNCATION_FIXTURE, 3) == pytest.approx(4 / 9)
    assert R_hat_at_K(_TRUNCATION_FIXTURE, 4) == pytest.approx(1 / 2)


def test_U_circ_at_K_is_non_decreasing_in_K() -> None:
    """Per-prompt max over the first K samples can only grow as K grows;
    averaging preserves the inequality."""
    rng = np.random.default_rng(0)
    utilities = rng.integers(0, 2, size=(50, 32)).astype(float)
    values = [U_circ_at_K(utilities, K) for K in range(1, utilities.shape[1] + 1)]
    for prev, curr in zip(values, values[1:]):
        assert curr >= prev, f"U_circ_at_K decreased: {prev} -> {curr}"


def test_K_equals_one_is_degenerate() -> None:
    """At K=1: U_circ = U_bar = mean of column 0; R_hat ≡ 0."""
    utilities = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=float,
    )
    expected = utilities[:, 0].mean()
    assert U_circ_at_K(utilities, 1) == pytest.approx(expected)
    assert U_bar_at_K(utilities, 1) == pytest.approx(expected)
    assert R_hat_at_K(utilities, 1) == 0.0


def test_K_equals_n_max_matches_full_matrix_estimator() -> None:
    """At K = n_max, the helpers reproduce the full-matrix
    ``compute_rational_gap`` aggregates."""
    rng = np.random.default_rng(0)
    utilities = rng.integers(0, 2, size=(50, 16)).astype(float)
    n_max = utilities.shape[1]
    full = compute_rational_gap(utilities)
    assert U_circ_at_K(utilities, n_max) == pytest.approx(full.U_circ_K)
    assert U_bar_at_K(utilities, n_max) == pytest.approx(full.U_bar_K)
    assert R_hat_at_K(utilities, n_max) == pytest.approx(full.R_hat_K)


def test_K_outside_valid_range_raises() -> None:
    utilities = np.zeros((3, 4), dtype=float)
    for bad_K in (0, -1, 5, 100):
        with pytest.raises(ValueError, match="K"):
            U_circ_at_K(utilities, bad_K)
        with pytest.raises(ValueError, match="K"):
            U_bar_at_K(utilities, bad_K)
        with pytest.raises(ValueError, match="K"):
            R_hat_at_K(utilities, bad_K)


def test_at_K_helpers_reject_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        U_circ_at_K(np.array([1.0, 0.0]), 1)


# ---------------------------------------------------------------------------
# Integration: compute_rational_gap → bootstrap_ci_over_prompts pipeline
# ---------------------------------------------------------------------------


def test_methodology_workflow_R_hat_K_with_CI() -> None:
    """Full per-seed workflow: utilities → compute_rational_gap →
    bootstrap_ci_over_prompts(per_prompt_R_hat_K). This is what each of the
    3 seeds produces; cross-seed mean ± std happens at a higher level."""
    rng = np.random.default_rng(0)
    M, K = 200, 16
    utilities = rng.integers(0, 2, size=(M, K)).astype(float)
    est = compute_rational_gap(utilities)
    ci = bootstrap_ci_over_prompts(est.per_prompt_R_hat_K, seed=0)
    # Sanity: bootstrap mean of R_hat_K should be close to the point estimate.
    assert ci.mean == pytest.approx(est.R_hat_K, abs=0.01)
    assert ci.ci_low <= est.R_hat_K <= ci.ci_high or abs(ci.mean - est.R_hat_K) < 0.01
