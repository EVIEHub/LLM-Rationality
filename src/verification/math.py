"""MATH verifier: extract the ``\\boxed{}`` answer and check symbolic
equivalence against the ground truth using ``math-verify`` (a SymPy-based
checker robust to common LaTeX variants).

The extraction step is implemented in this module rather than delegated to
``math-verify`` because, without an explicit ``\\boxed{}`` anchor, the
library's parser extracts only the leading numeric token from bare LaTeX
like ``2\\sqrt{3}`` (silently producing ``2`` instead of ``2*sqrt(3)``).
We extract the last balanced ``\\boxed{}`` ourselves, then re-wrap both
predicted and ground-truth strings in ``\\boxed{}`` before parsing, so
``math-verify`` always sees an anchored expression on both sides.

Returns a ``float`` in ``{0.0, 1.0}``. Per AGENT.md §3.3, SymPy timeouts
and internal exceptions in ``math-verify`` must surface as ``0.0``
(incorrect), never as an exception — a single hard prompt must not abort
a 1000-prompt run. ``math-verify`` defaults to ``raise_on_error=False``
and ``timeout_seconds=5`` for both parsing and verification, but we still
wrap its calls in ``try/except`` to defend against any uncaught failure
modes inside the library or its SymPy dependency.

This module is a pure function — no I/O, no global state, no logging.
The audit log mandated by AGENT.md §3.3 is written by the caller.
"""

from __future__ import annotations

import re
from typing import Optional

import math_verify

_PARSE_TIMEOUT_SECONDS = 5
_VERIFY_TIMEOUT_SECONDS = 5
_BOXED_OPEN = re.compile(r"\\boxed\s*\{")


def _extract_last_boxed(text: str) -> Optional[str]:
    """Return the contents of the last balanced ``\\boxed{...}`` in text.

    Walks the text character-by-character with a depth counter, so nested
    braces (e.g. ``\\boxed{\\frac{1}{2}}``) are handled correctly. LaTeX
    escape sequences (``\\{``, ``\\}``, ``\\\\``) are skipped wholesale so
    they do not perturb the depth count. If the rightmost ``\\boxed{`` is
    unbalanced (no matching closing brace), the function backs off to
    earlier ``\\boxed{`` openings and returns the first balanced one
    found from the right.

    Args:
        text: The text to search.

    Returns:
        The raw content string between the matching braces (whitespace
        preserved), or ``None`` if no balanced ``\\boxed{...}`` is found.
    """
    starts = [match.end() for match in _BOXED_OPEN.finditer(text)]
    if not starts:
        return None
    for start in reversed(starts):
        depth = 1
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                # Skip the escape sequence so \{, \}, \\ do not affect depth.
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
            i += 1
    return None


def verify(generation: str, ground_truth: str) -> float:
    """Verify a MATH model generation against the LaTeX ground truth.

    The predicted answer is taken from the last balanced ``\\boxed{...}``
    in ``generation``; if no such boxed expression exists, the
    generation is considered to have failed the format requirement and
    the function returns ``0.0``. The ground truth may be wrapped in
    ``\\boxed{...}`` (as in some MATH dataset variants) or be a bare
    LaTeX expression; both are handled.

    Equivalence is delegated to ``math_verify.verify``, which compares
    SymPy-parsed expressions and handles common LaTeX equivalences
    (``\\frac{1}{2} \\leftrightarrow 0.5``, ``\\sqrt{12} \\leftrightarrow
    2\\sqrt{3}``, algebraic rearrangements, intervals, etc.).

    Args:
        generation: The model's full output text for a MATH prompt.
        ground_truth: The ground-truth answer, as a LaTeX expression or
            a string already wrapped in ``\\boxed{...}``.

    Returns:
        ``1.0`` if the predicted expression is symbolically equivalent
        to the ground truth, else ``0.0``. Returns ``0.0`` on any of:
        no ``\\boxed{}`` in the generation, empty ground truth, parse
        failure, ``math-verify`` timeout, or any internal exception.
    """
    pred_str = _extract_last_boxed(generation)
    if pred_str is None:
        return 0.0

    gt_str = _extract_last_boxed(ground_truth)
    if gt_str is None:
        gt_str = ground_truth.strip()
    if not gt_str:
        return 0.0

    # Wrap both sides in \boxed{} so math-verify's parser anchors on a
    # known marker. Without the anchor, bare LaTeX like "2\sqrt{3}" is
    # silently truncated to its leading integer.
    try:
        gold = math_verify.parse(
            f"\\boxed{{{gt_str}}}",
            parsing_timeout=_PARSE_TIMEOUT_SECONDS,
        )
        pred = math_verify.parse(
            f"\\boxed{{{pred_str}}}",
            parsing_timeout=_PARSE_TIMEOUT_SECONDS,
        )
    except Exception:
        return 0.0

    if not gold or not pred:
        return 0.0

    try:
        is_equivalent = math_verify.verify(
            gold,
            pred,
            timeout_seconds=_VERIFY_TIMEOUT_SECONDS,
        )
    except Exception:
        return 0.0

    return 1.0 if is_equivalent else 0.0
