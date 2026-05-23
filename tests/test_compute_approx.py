"""Tests for the compute-approximation metric A_K (Lemma 2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.rational_gap import (
    A_at_K,
    A_K_extrapolated,
    U_circ_at_K,
    p_hat_at_K,
)


def test_A_K_equals_one_minus_U_circ():
    # Binary utilities, M=4 prompts, K=4 samples.
    u = np.array([
        [1, 0, 0, 0],   # reachable (>=1 success)
        [0, 0, 0, 0],   # unreachable (p_x = 0)
        [1, 1, 1, 1],   # always correct
        [0, 1, 0, 0],   # reachable
    ], dtype=float)
    for K in (1, 2, 3, 4):
        assert A_at_K(u, K) == pytest.approx(1.0 - U_circ_at_K(u, K))


def test_A_K_is_fraction_with_no_success_binary():
    u = np.array([
        [0, 0, 0, 0],   # never
        [0, 0, 1, 0],   # success at k=3
        [0, 0, 0, 0],   # never
        [1, 0, 0, 0],   # success at k=1
    ], dtype=float)
    # At K=4: 2 of 4 prompts have no success -> A_4 = 0.5
    assert A_at_K(u, 4) == pytest.approx(0.5)
    # At K=2: prompt 1 not yet successful -> 3 of 4 fail -> A_2 = 0.75
    assert A_at_K(u, 2) == pytest.approx(0.75)


def test_A_K_monotone_non_increasing_in_K():
    rng = np.random.default_rng(0)
    u = (rng.random((50, 16)) < 0.2).astype(float)
    vals = [A_at_K(u, K) for K in range(1, 17)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


def test_A_K_zero_when_all_reachable_and_floor_when_none():
    all_ok = np.ones((5, 8))
    assert A_at_K(all_ok, 8) == pytest.approx(0.0)
    # half the prompts entirely unreachable -> floor 0.5 at any K
    half = np.zeros((4, 8))
    half[0, 0] = 1.0
    half[1, 3] = 1.0
    assert A_at_K(half, 8) == pytest.approx(0.5)


def test_p_hat_matches_mean_and_shape():
    u = np.array([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=float)
    ph = p_hat_at_K(u, 4)
    assert ph.shape == (2,)
    assert ph == pytest.approx([0.25, 0.5])


def test_extrapolation_floors_at_unreachable_mass():
    # 3 prompts: one unreachable (p=0), two with p=0.5.
    p = np.array([0.0, 0.5, 0.5])
    # As K_target grows, (1-p)^K -> 0 for p>0, so A -> 1/3 (the p=0 mass).
    assert A_K_extrapolated(p, 1) == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert A_K_extrapolated(p, 1000) == pytest.approx(1.0 / 3, abs=1e-6)


def test_invalid_inputs_raise():
    u = np.ones((3, 4))
    with pytest.raises(ValueError):
        A_at_K(u, 0)
    with pytest.raises(ValueError):
        A_at_K(u, 5)
    with pytest.raises(ValueError):
        p_hat_at_K(np.ones(4), 1)        # not 2D
    with pytest.raises(ValueError):
        A_K_extrapolated(np.ones((2, 2)), 1)  # not 1D
