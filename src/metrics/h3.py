"""H3 — answer-key extractors and the self-consistency utility matrix.

H3 frames each inference procedure $\\mathcal{I}$ as its own sampling
distribution $\\pi_\\mathcal{I}$, and reports the same rational-gap
triple as H1 ($U^\\circ_K, \\bar{U}_K, \\hat{\\mathcal{R}}_K$) on $K$
samples from $\\pi_\\mathcal{I}$, computed by
:func:`src.metrics.rational_gap.compute_rational_gap`. H3 introduces
no new aggregate estimator.

For direct-sampling procedures ($\\tau \\in \\{0, 0.7, 1.0\\}$) the
$K$ samples come straight from vLLM at the chosen $\\tau$ (or from the
H1 cache for $\\tau=1$); no helpers are needed beyond what H1 provides.

For the self-consistency procedure $\\pi_{\\text{SC}(n)}$: one draw is
defined as picking $n$ underlying samples at $\\tau=1$ and outputting
the modal-key sample. The function
:func:`self_consistency_utility_matrix` returns the $(M, K)$ utility
matrix for $K$ such draws built by **bootstrap-resampling** the H1
$K_{H1}=64$ cache (the bootstrap stands in for fresh iid draws from
$\\pi_{\\tau=1}$, which would otherwise require $K \\times n$ fresh
GPU samples per prompt). Pass that matrix to
``compute_rational_gap`` to get the SC procedure's H1-shape triple.

Dataset-specific answer-key extractors (registered for GSM8K and MATH)
provide the hashable canonical key per generation used by the SC
voter. HumanEval is intentionally absent — code-level "agreement"
requires test-execution majority and is out of scope.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np

from src.verification import gsm8k as gsm8k_verifier
from src.verification import math as math_verifier
from src.verification import matharena as matharena_verifier


# Per-dataset answer-key extractor. Maps a generation string to a hashable
# canonical key, or ``None`` if no parseable answer is present. Two
# generations agree iff their keys are equal.
#
# GSM8K: float (parses #### N, \\boxed{N}, "the answer is N", or last
# bare number). Numeric equality means "42" and "42.0" agree.
#
# MATH: raw last-balanced \\boxed{...} string with whitespace collapsed.
# This is a *lower bound* on agreement: two responses that wrote
# "\\frac{1}{2}" and "0.5" both correctly will count as disagreeing,
# even though math-verify treats them as equivalent. Using math-verify
# for normalisation is too expensive at $K \\times M$ scale (parser
# timeout 5 s).
#
# HumanEval is intentionally absent — code-level "agreement" requires
# test-execution majority, out of scope.


def _extract_gsm8k_key(text: str) -> Optional[float]:
    return gsm8k_verifier.extract_answer(text)


def _extract_math_key(text: str) -> Optional[str]:
    raw = math_verifier._extract_last_boxed(text)
    if raw is None:
        return None
    # Collapse all whitespace so "\\frac{1}{2}" == "\\frac{1} {2}".
    return "".join(raw.split())


def _extract_matharena_key(text: str) -> Optional[str]:
    # MathArena uses the same \\boxed{} convention as MATH; reuse the
    # matharena verifier's extractor (which is brace-balanced).
    raw = matharena_verifier._extract_last_boxed(text)
    if raw is None:
        return None
    return "".join(raw.split())


_KEY_EXTRACTORS: dict[str, Callable[[str], Optional[object]]] = {
    "gsm8k": _extract_gsm8k_key,
    "math": _extract_math_key,
    "matharena": _extract_matharena_key,
}


def supports_answer_key_extraction(dataset: str) -> bool:
    """True if the dataset has a registered answer-key extractor."""
    return dataset.strip().lower() in _KEY_EXTRACTORS


def extract_answer_key(dataset: str, text: str) -> Optional[object]:
    """Run the dataset's answer-key extractor on a generation string.

    Returns the canonical key (float for GSM8K, normalised LaTeX string
    for MATH) or ``None`` if no parseable answer is found.

    Args:
        dataset: Dataset name (case-insensitive, whitespace-stripped).
        text: A generation string.

    Raises:
        KeyError: If no extractor is registered for ``dataset``.
    """
    key = dataset.strip().lower()
    if key not in _KEY_EXTRACTORS:
        known = ", ".join(sorted(_KEY_EXTRACTORS))
        raise KeyError(
            f"no answer-key extractor for dataset {dataset!r}; known: {known}"
        )
    return _KEY_EXTRACTORS[key](text)


def self_consistency_utility_matrix(
    util_h1: np.ndarray,
    samples_per_prompt: list[list[str]],
    dataset: str,
    n: int,
    K: int,
    seed: int,
) -> np.ndarray:
    """$(M, K)$ utility matrix for the SC$(n)$ procedure.

    Each column is one draw from $\\pi_{\\text{SC}(n)}$: pick $n$
    underlying samples at $\\tau=1$ (bootstrap-resampled from the H1
    cache), majority-vote on the dataset's answer key, output the
    modal-key sample, return its utility. The bootstrap stands in for
    drawing $n$ fresh iid samples from $\\pi_{\\tau=1}$ — exact
    equivalence holds in the $K_{H1} \\to \\infty$ limit; with the
    operating $K_{H1}=64$ it is the practical alternative to spending
    $K \\times n$ fresh GPU samples per prompt per cell.

    Args:
        util_h1: $(M, K_{H1})$ utility matrix of the H1 cell at
            $\\tau=1$. Columns are the per-prompt utilities of the
            cached samples; SC's "verify the modal-key sample" reduces
            to a lookup in this matrix.
        samples_per_prompt: $M \\times K_{H1}$ raw generation strings,
            same row order as ``util_h1``. Used only to extract answer
            keys for the voting; verifier is not re-run.
        dataset: Dataset name (must support answer-key extraction;
            see :func:`supports_answer_key_extraction`).
        n: Underlying sample count per SC draw. Must satisfy
            ``1 <= n``; values larger than $K_{H1}$ still work via
            bootstrap (samples reused across positions in a draw).
        K: Number of SC draws per prompt — defines the "$K$" of the
            output utility matrix that downstream
            ``compute_rational_gap`` will see.
        seed: RNG seed; same seed → byte-identical output for the
            same inputs.

    Returns:
        $(M, K)$ float utility array in $\\{0.0, 1.0\\}$.

    Raises:
        ValueError: On unsupported dataset, $n < 1$, $K < 1$, or shape
            mismatch between ``util_h1`` and ``samples_per_prompt``.
    """
    if not supports_answer_key_extraction(dataset):
        raise ValueError(
            f"dataset {dataset!r} does not support self-consistency; "
            f"supported: {sorted(_KEY_EXTRACTORS)}"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    M = len(samples_per_prompt)
    if util_h1.shape[0] != M:
        raise ValueError(
            f"util_h1 has {util_h1.shape[0]} rows but samples_per_prompt has {M}"
        )
    K_h1 = util_h1.shape[1]
    for i, samples in enumerate(samples_per_prompt):
        if len(samples) != K_h1:
            raise ValueError(
                f"prompt {i}: {len(samples)} samples vs util_h1's {K_h1} columns"
            )

    extractor = _KEY_EXTRACTORS[dataset.strip().lower()]
    rng = np.random.default_rng(seed)
    out = np.zeros((M, K), dtype=float)

    for i in range(M):
        # Pre-extract keys once per prompt; reused across K draws.
        keys = [extractor(s) for s in samples_per_prompt[i]]
        for k in range(K):
            indices = rng.integers(0, K_h1, size=n)
            sub_keys = [keys[j] for j in indices]
            counts: Counter = Counter(kk for kk in sub_keys if kk is not None)
            if not counts:
                # No parseable answer in this bootstrap draw -> 0 utility.
                out[i, k] = 0.0
                continue
            mode_key, _ = counts.most_common(1)[0]
            # Representative is the FIRST position in the bootstrap draw
            # whose key matches the mode. Map back to the global cache
            # column to read the precomputed utility.
            rep_local = next(
                pos for pos in range(n) if sub_keys[pos] == mode_key
            )
            rep_global = int(indices[rep_local])
            out[i, k] = float(util_h1[i, rep_global])
    return out
