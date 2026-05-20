"""Unified verifier registry for the rational-gap pipeline.

The pipeline calls into a single function ``verify(dataset, generation,
ground_truth)`` that dispatches to the per-dataset verifier. This keeps
the measurement loop dataset-agnostic — all dataset-specific knowledge
about answer extraction and equivalence checking lives in
``src/verification/<dataset>.py``.

All registered verifiers share the signature
``(generation: str, ground_truth: str) -> float`` and the float return
contract from AGENT.md §3.3. Current verifiers all return values in
``{0.0, 1.0}`` (binary $U$); the float typing keeps the interface
forward-compatible with reward-model verifiers without an API break.
"""

from __future__ import annotations

from typing import Callable, Mapping

from src.verification import gsm8k, humaneval, livecodebench, matharena
from src.verification import math as math_verifier  # avoid shadowing stdlib name in callers


Verifier = Callable[[str, str], float]


def _self_judge_marker(generation: str, ground_truth: str) -> float:
    """Sentinel for the preference-mode verifier.

    The actual self-judge logic is batched and needs a live vLLM runner;
    it lives in :mod:`src.verification.self_judge`. Cell runners detect
    ``ds_cfg['verifier'] == 'self_judge'`` and dispatch to
    ``score_matrix(...)`` instead of calling this function. If you see
    this exception, a verifier-binary code path was given a preference
    dataset — bug.
    """
    raise NotImplementedError(
        "self_judge is a batched preference verifier that requires a "
        "VllmRunner; call src.verification.self_judge.score_matrix(...) "
        "from the cell-runner preference branch instead of verify()."
    )


_REGISTRY: Mapping[str, Verifier] = {
    "gsm8k": gsm8k.verify,
    "math": math_verifier.verify,
    "humaneval": humaneval.verify,
    "matharena": matharena.verify,
    "livecodebench": livecodebench.verify,
    "self_judge": _self_judge_marker,
}


def known_datasets() -> tuple[str, ...]:
    """Return a sorted tuple of canonical dataset names with registered verifiers."""
    return tuple(sorted(_REGISTRY))


def get_verifier(dataset: str) -> Verifier:
    """Return the verifier callable for a dataset.

    Args:
        dataset: Dataset name. Lookup is case-insensitive and ignores
            surrounding whitespace, so ``"GSM8K"`` and ``" gsm8k "``
            both resolve to ``gsm8k``.

    Returns:
        A callable with signature
        ``(generation: str, ground_truth: str) -> float``.

    Raises:
        KeyError: If no verifier is registered for the given dataset.
    """
    key = dataset.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(known_datasets())
        raise KeyError(f"Unknown dataset {dataset!r}; known: {known}")
    return _REGISTRY[key]


def verify(dataset: str, generation: str, ground_truth: str) -> float:
    """Verify ``generation`` against ``ground_truth`` using the dataset's verifier.

    Convenience wrapper for ``get_verifier(dataset)(generation, ground_truth)``.

    Args:
        dataset: Dataset name; see :func:`get_verifier` for accepted forms.
        generation: The model's full output text for the prompt.
        ground_truth: The dataset-specific ground-truth string. The format
            is dataset-dependent — for HumanEval it is a check program;
            for GSM8K and MATH it is the answer expression.

    Returns:
        ``1.0`` for a passing verification, ``0.0`` otherwise. See the
        per-dataset modules for the failure modes that map to ``0.0``.
    """
    return get_verifier(dataset)(generation, ground_truth)
