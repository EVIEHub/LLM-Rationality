"""BBH (BIG-Bench Hard) verifier.

BBH is a suite of 23+ diverse hard-reasoning tasks (logical deduction,
temporal/causal reasoning, fallacy detection, ...). It is the non-math
diversity axis for H4's length sweep: unlike GSM8K/MATH, the reasoning
type varies across tasks, but every task still has a *short, canonical
final answer* — which is what budget forcing (s1-style two-stage commit)
needs to be verifiable.

Each ``lukaemon/bbh`` row is ``{"input": question, "target": answer}``.
The ground-truth ``target`` comes in three canonical shapes across the
subset we use:

* multiple choice  — ``"(A)"`` .. ``"(R)"``  (a single parenthesised letter)
* boolean / token  — ``"True"``/``"False"``, ``"Yes"``/``"No"``,
                      ``"valid"``/``"invalid"``
* count            — a bare integer (``object_counting``)

The verifier dispatches on the *shape of the target*, so it needs no
per-task table: it decides letter- vs token- vs number-matching from the
ground truth itself, then extracts the corresponding answer from the
model's budget-forced output (the text after the ``"Final answer:"``
commit, falling back to the last line).

Returns ``float`` in ``{0.0, 1.0}``. Pure function — no I/O, no logging
(the audit log is written by the caller, per AGENT.md §3.3).
"""

from __future__ import annotations

import re
from typing import Optional

# Phrases the model is likely to put immediately before its committed
# answer. Budget forcing appends ``"\n\nFinal answer:"``; instruction-tuned
# models also volunteer "the answer is ...". Tried in order; the LAST match
# wins so restated intermediate guesses don't shadow the final commit.
_TRIGGERS = re.compile(
    r"(?is)(?:final\s+answer|the\s+answer\s+is|answer)\s*[:\-]?\s*",
)

# A parenthesised choice letter, e.g. "(A)" or "( B )". BBH MC answers run
# A..R (reasoning_about_colored_objects has 18 options).
_PAREN_LETTER = re.compile(r"\(\s*([A-Ra-r])\s*\)")
# A bare standalone letter token (e.g. "answer is A"), as a fallback when
# the model drops the parentheses.
_BARE_LETTER = re.compile(r"(?<![A-Za-z])([A-Ra-r])(?![A-Za-z])")
_INT = re.compile(r"-?\d+")


def _final_span(text: str) -> str:
    """Return the slice of ``text`` holding the committed answer.

    Prefers the text after the last answer-trigger phrase; if none is
    present, falls back to the last non-empty line. Capped to a short
    window so a rambling tail can't smuggle in stray tokens.
    """
    matches = list(_TRIGGERS.finditer(text))
    if matches:
        span = text[matches[-1].end():]
    else:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        span = lines[-1] if lines else text
    return span[:160]


def _norm_token(s: str) -> str:
    """Lowercase and strip surrounding punctuation/quotes/whitespace."""
    return s.strip().strip("().,:;!?\"'*` ").lower()


def _extract_letter(span: str, full: str) -> Optional[str]:
    """Extract a lowercase choice letter from the answer span (then full text)."""
    for src in (span, full):
        m = _PAREN_LETTER.search(src)
        if m:
            return m.group(1).lower()
    # No parentheses anywhere — accept a lone letter only from the short span,
    # where it is much more likely to be the committed answer than prose.
    m = _BARE_LETTER.search(span)
    return m.group(1).lower() if m else None


def verify(generation: str, ground_truth: str) -> float:
    """Verify a BBH generation against the ground-truth ``target``.

    Dispatches on the shape of ``ground_truth``:

    * parenthesised letter (``"(A)"``) -> letter match
    * bare integer (``"7"``)           -> integer match
    * otherwise (``"True"``/``"Yes"``/``"valid"``/...) -> token match,
      satisfied if the canonical target token appears as a whole word in
      the committed answer span.

    Returns ``1.0`` on a match, else ``0.0``. Returns ``0.0`` when the
    model's answer cannot be located.
    """
    target_raw = ground_truth.strip()
    if not target_raw:
        return 0.0
    span = _final_span(generation)

    # --- multiple choice: target is "(X)" ---
    paren = _PAREN_LETTER.fullmatch(target_raw)
    if paren:
        gold = paren.group(1).lower()
        pred = _extract_letter(span, generation)
        return 1.0 if pred == gold else 0.0

    # --- count: target is a bare integer ---
    if _INT.fullmatch(target_raw):
        gold = int(target_raw)
        ints = _INT.findall(span) or _INT.findall(generation)
        return 1.0 if ints and int(ints[-1]) == gold else 0.0

    # --- token: True/False, Yes/No, valid/invalid, word answers ---
    gold = _norm_token(target_raw)
    if not gold:
        return 0.0
    pred = _norm_token(span)
    if pred == gold:
        return 1.0
    # Allow the gold token to sit inside a short phrase ("the answer is yes").
    if re.search(rf"(?<![a-z]){re.escape(gold)}(?![a-z])", span.lower()):
        return 1.0
    return 0.0
