"""GSM8K verifier: extract the final numeric answer from a generation and
compare it to the ground truth.

The verifier tries answer-extraction patterns in priority order: the GSM8K
canonical ``#### N``, then LaTeX ``\\boxed{N}``, then natural-language
"the answer is N", then the last bare number in the text. Within each
pattern the rightmost match wins, since models typically restate
intermediate numbers before stating the final answer.

Both the predicted and the ground-truth strings are run through the same
extraction pipeline, so this works whether the ground truth is the raw
GSM8K format (``"... #### 18"``) or a bare number (``"18"``).

Returns a ``float`` in ``{0.0, 1.0}``. The ``float`` return type matches the
general utility signature in AGENT.md §3.3; under binary $U$ on GSM8K it
coincides numerically with per-prompt correctness.

This module is a pure function — no I/O, no global state, no logging. The
audit log mandated by AGENT.md §3.3 is written by the caller (the
verification step in the pipeline), not here.
"""

from __future__ import annotations

import math
import re
from typing import Optional

# A numeric token: optional sign, optional dollar prefix, then either a
# comma-grouped integer with optional decimal, or a bare integer with
# optional decimal and optional percent. Two alternatives are needed
# because the comma-grouped form requires at least one ``,\d{3}`` group.
_NUMBER_PATTERN = (
    r"-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,234 or 1,234.56 or $1,234
    r"|"
    r"-?\$?\d+(?:\.\d+)?%?"  # 123 or 123.45 or -7 or $50 or 80%
)

# Patterns are tried in this order; first non-empty extraction wins.
_HASHED_PATTERN = re.compile(r"####\s*(" + _NUMBER_PATTERN + r")")
_BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*(" + _NUMBER_PATTERN + r")\s*\}")
_ANSWER_IS_PATTERN = re.compile(
    r"(?i)(?:the\s+)?answer\s+is[\s:=]*(" + _NUMBER_PATTERN + r")"
)
_BARE_NUMBER_PATTERN = re.compile(_NUMBER_PATTERN)


def _normalise(token: str) -> Optional[float]:
    """Convert an extracted numeric token to ``float``.

    Strips commas, leading ``$``, and trailing ``%``, then parses with
    ``float``. Returns ``None`` if the cleaned token is not a valid number.
    """
    cleaned = token.strip().replace(",", "").lstrip("$").rstrip("%")
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from a GSM8K generation or ground truth.

    Tries patterns in priority order: ``#### N``, ``\\boxed{N}``,
    "the answer is N", last bare number. Within each pattern the rightmost
    match is returned, so models that restate intermediate numbers before
    a final ``#### 42`` still parse correctly.

    Args:
        text: Free-form text potentially containing the answer.

    Returns:
        The parsed answer as a ``float``, or ``None`` if no parseable
        number is found.
    """
    for pattern in (_HASHED_PATTERN, _BOXED_PATTERN, _ANSWER_IS_PATTERN):
        matches = pattern.findall(text)
        if matches:
            value = _normalise(matches[-1])
            if value is not None:
                return value

    bare_matches = _BARE_NUMBER_PATTERN.findall(text)
    for token in reversed(bare_matches):
        value = _normalise(token)
        if value is not None:
            return value
    return None


def verify(generation: str, ground_truth: str) -> float:
    """Verify a GSM8K model generation against the ground-truth answer.

    Args:
        generation: The model's full output text for a GSM8K prompt.
        ground_truth: The ground-truth answer string. May be the raw
            GSM8K format (``"... #### 18"``) or a bare number (``"18"``);
            both are handled by the same extraction pipeline.

    Returns:
        ``1.0`` if the predicted answer matches the ground truth within
        a small numeric tolerance (so ``72`` matches ``72.0``), else
        ``0.0``. Returns ``0.0`` if either side cannot be parsed.
    """
    predicted = extract_answer(generation)
    target = extract_answer(ground_truth)
    if predicted is None or target is None:
        return 0.0
    if math.isclose(predicted, target, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0
    return 0.0
