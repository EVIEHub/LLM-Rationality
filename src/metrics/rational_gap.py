"""Rational gap estimators and bootstrap uncertainty quantification.

Implements the paper's empirical estimators
(see ``project_hypotheses`` memory note):

- $U^\\circ_K(x_i) = \\max_{k\\in[K]} U(x_i, y_{i,k})$ (per-prompt reachable utility)
- $\\bar{U}_K(x_i) = \\frac{1}{K}\\sum_k U(x_i, y_{i,k})$ (per-prompt mean utility)
- $\\hat{\\mathcal{R}}_K(x_i) = U^\\circ_K(x_i) - \\bar{U}_K(x_i)$ (per-prompt rational gap)

Aggregates are simple prompt-means:

- ``U_circ_K = mean_i U_circ_K(x_i)``
- ``U_bar_K = mean_i U_bar_K(x_i)``
- ``R_hat_K = U_circ_K - U_bar_K``

The module is generic over $U$: any float-valued utility array of shape
``(M, K)`` is accepted, even though the current verifiers all return
binary ``{0.0, 1.0}`` utilities.

The primary uncertainty quantification is **bootstrap CI over prompts**
(see methodology memo): for B = 1000 resamples drawn with replacement
from the prompt set, compute the resampled mean of $\\hat{\\mathcal{R}}_K$,
and report the 95th-percentile interval. This module returns the
per-prompt array so callers can bootstrap whichever statistic they need
($U^\\circ_K$, $\\bar{U}_K$, or $\\hat{\\mathcal{R}}_K$).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RationalGapEstimate:
    """Aggregate and per-prompt rational-gap estimators for one run.

    All aggregate fields are scalar floats (means over prompts). The
    per-prompt arrays have shape ``(M,)`` and exist so callers can
    compute prompt-bootstrap CIs without recomputing the underlying
    utilities.
    """

    U_circ_K: float
    U_bar_K: float
    R_hat_K: float
    per_prompt_U_circ_K: np.ndarray
    per_prompt_U_bar_K: np.ndarray
    per_prompt_R_hat_K: np.ndarray
    M: int
    K: int


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile-based confidence interval from a bootstrap distribution.

    ``mean`` is the mean of the bootstrap distribution (i.e. the mean of
    the per-resample means), used as the point estimate for cross-seed
    aggregation per the methodology memo.
    """

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int


def compute_rational_gap(utilities: np.ndarray) -> RationalGapEstimate:
    """Compute $U^\\circ_K$, $\\bar{U}_K$, $\\hat{\\mathcal{R}}_K$ from a utility array.

    Args:
        utilities: Float array of shape ``(M, K)`` where entry ``(i, k)``
            is $U(x_i, y_{i,k})$.

    Returns:
        :class:`RationalGapEstimate` with both aggregate scalars and
        per-prompt arrays.

    Raises:
        ValueError: If ``utilities`` is not 2D or is empty.
    """
    if utilities.ndim != 2:
        raise ValueError(
            f"utilities must be 2D of shape (M, K), got shape {utilities.shape}"
        )
    if utilities.size == 0:
        raise ValueError("utilities array is empty")

    M, K = utilities.shape
    per_prompt_U_circ = utilities.max(axis=1).astype(float, copy=False)
    per_prompt_U_bar = utilities.mean(axis=1).astype(float, copy=False)
    per_prompt_R = per_prompt_U_circ - per_prompt_U_bar

    return RationalGapEstimate(
        U_circ_K=float(per_prompt_U_circ.mean()),
        U_bar_K=float(per_prompt_U_bar.mean()),
        R_hat_K=float(per_prompt_R.mean()),
        per_prompt_U_circ_K=per_prompt_U_circ,
        per_prompt_U_bar_K=per_prompt_U_bar,
        per_prompt_R_hat_K=per_prompt_R,
        M=M,
        K=K,
    )


def U_circ_at_K(utilities: np.ndarray, K: int) -> float:
    """$U^\\circ_K$ from the first $K$ columns of an $(M, n_\\text{max})$ matrix.

    $$U^\\circ_K = \\frac{1}{M} \\sum_{i=1}^{M} \\max_{k \\in [K]} U(x_i, y_{i,k})$$

    Direct one-line transcription of the paper formula — not an estimator
    of an expectation. The saturation curve at varying $K$ is built by
    calling this on the same matrix at each grid point, after sampling
    once at $n_\\text{max}$.

    Args:
        utilities: Float array of shape $(M, n_\\text{max})$.
        K: Sampling budget; ``1 <= K <= n_max``.

    Returns:
        The scalar $U^\\circ_K$.

    Raises:
        ValueError: If ``K`` is outside ``[1, n_max]`` or the input is
            not 2D.
    """
    if utilities.ndim != 2:
        raise ValueError(f"utilities must be 2D, got shape {utilities.shape}")
    n_max = utilities.shape[1]
    if K < 1 or K > n_max:
        raise ValueError(f"K={K} outside [1, {n_max}]")
    return float(utilities[:, :K].max(axis=1).mean())


def U_bar_at_K(utilities: np.ndarray, K: int) -> float:
    """$\\bar{U}_K$ from the first $K$ columns of an $(M, n_\\text{max})$ matrix.

    $$\\bar{U}_K = \\frac{1}{MK} \\sum_{i=1}^{M} \\sum_{k=1}^{K} U(x_i, y_{i,k})$$

    Args:
        utilities: Float array of shape $(M, n_\\text{max})$.
        K: Sampling budget; ``1 <= K <= n_max``.

    Returns:
        The scalar $\\bar{U}_K$.
    """
    if utilities.ndim != 2:
        raise ValueError(f"utilities must be 2D, got shape {utilities.shape}")
    n_max = utilities.shape[1]
    if K < 1 or K > n_max:
        raise ValueError(f"K={K} outside [1, {n_max}]")
    return float(utilities[:, :K].mean())


def R_hat_at_K(utilities: np.ndarray, K: int) -> float:
    """$\\hat{\\mathcal{R}}_K = U^\\circ_K - \\bar{U}_K$ from the first $K$ columns."""
    return U_circ_at_K(utilities, K) - U_bar_at_K(utilities, K)


def bootstrap_ci_over_prompts(
    per_prompt_values: np.ndarray,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap a confidence interval for the mean of ``per_prompt_values``.

    Resamples ``M`` prompts with replacement ``n_resamples`` times,
    computes the mean of each resample, and returns the empirical mean
    and the symmetric percentile interval at the requested confidence.

    Args:
        per_prompt_values: 1D float array of length $M$ (e.g. the
            ``per_prompt_R_hat_K`` field of a
            :class:`RationalGapEstimate`).
        n_resamples: $B$. Default 1000 per the methodology memo.
        confidence: Confidence level in $(0, 1)$. Default 0.95.
        seed: RNG seed; same seed → byte-identical CI for the same
            input.

    Returns:
        :class:`BootstrapCI` with the bootstrap mean and the
        $(1-\\text{confidence})/2$ and $(1+\\text{confidence})/2$
        percentiles.

    Raises:
        ValueError: If the input is not 1D, is empty, or
            ``n_resamples < 1`` or ``confidence`` is outside $(0, 1)$.
    """
    if per_prompt_values.ndim != 1:
        raise ValueError(
            f"per_prompt_values must be 1D, got shape {per_prompt_values.shape}"
        )
    M = len(per_prompt_values)
    if M == 0:
        raise ValueError("per_prompt_values is empty")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, M, size=(n_resamples, M))
    resample_means = per_prompt_values[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(resample_means, alpha))
    ci_high = float(np.quantile(resample_means, 1.0 - alpha))

    return BootstrapCI(
        mean=float(resample_means.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_resamples=n_resamples,
    )
