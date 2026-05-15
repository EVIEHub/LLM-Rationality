"""MathArena verifier (AIME 2025, BRUMO 2025).

Competition-math problems from the MathArena leaderboard (Beurer-Kellner
et al., ETH 2025). Answers are short integers or symbolic expressions
(e.g. ``"70"``, ``"2^{99}"``). Models are prompted to emit their answer
inside ``\\boxed{...}``; the verifier extracts the last balanced boxed
expression and compares to the ground truth via ``math_verify``, which
handles algebraic equivalence (``2*x + 3`` matches ``3 + 2*x``;
``1/2`` matches ``0.5``).

Returns ``float`` in ``{0.0, 1.0}``. Pure function, no I/O.
"""

from __future__ import annotations

import re
from typing import Optional

import math_verify

_PARSE_TIMEOUT_SECONDS = 5
_BOXED_OPEN = re.compile(r"\\boxed\s*\{")


def _extract_last_boxed(text: str) -> Optional[str]:
    """Brace-balanced extractor for the last ``\\boxed{...}`` (handles nesting)."""
    starts = [m.end() for m in _BOXED_OPEN.finditer(text)]
    for start in reversed(starts):
        depth = 1
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
            i += 1
    return None


def verify(generation: str, ground_truth: str) -> float:
    """Verify a MathArena generation against the competition answer string.

    Args:
        generation: The model's full output text.
        ground_truth: The competition's answer string (typically an integer
            like ``"70"`` or a short LaTeX expression like ``"2^{99}"``).

    Returns:
        ``1.0`` if the predicted boxed expression is symbolically
        equivalent to the ground truth via ``math_verify``, else ``0.0``.
        Returns ``0.0`` on missing ``\\boxed{...}``, empty ``ground_truth``,
        parse failure, or any internal exception.
    """
    pred_str = _extract_last_boxed(generation)
    if pred_str is None:
        return 0.0
    gt_str = ground_truth.strip()
    if not gt_str:
        return 0.0

    try:
        gold = math_verify.parse(
            f"\\boxed{{{gt_str}}}", parsing_timeout=_PARSE_TIMEOUT_SECONDS,
        )
        pred = math_verify.parse(
            f"\\boxed{{{pred_str}}}", parsing_timeout=_PARSE_TIMEOUT_SECONDS,
        )
    except Exception:
        return 1.0 if pred_str.replace(" ", "") == gt_str.replace(" ", "") else 0.0

    if not gold or not pred:
        return 1.0 if pred_str.replace(" ", "") == gt_str.replace(" ", "") else 0.0

    try:
        is_correct = bool(math_verify.verify(gold, pred))
    except Exception:
        return 0.0
    return 1.0 if is_correct else 0.0
