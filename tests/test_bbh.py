"""Tests for the BBH verifier (src/verification/bbh.py).

These exercise the three ground-truth shapes the verifier dispatches on
(parenthesised choice letter, boolean/token, bare integer) against
realistic budget-forced model outputs — including the awkward cases that
make naive string matching fail: restated intermediate guesses, missing
parentheses, and gold tokens that are substrings of longer words.

This is a load-bearing correctness contract for an H4 dataset, so per the
project rule these run live and are never skipped.
"""

from __future__ import annotations

import pytest

from src.verification import bbh
from src.verification.interface import verify as verify_dispatch


# --- multiple choice: target "(X)" ----------------------------------------

@pytest.mark.parametrize(
    "generation, target, expected",
    [
        # Clean budget-forced commit.
        ("...so the order is fixed.\n\nFinal answer: (B)", "(B)", 1.0),
        # Wrong letter.
        ("\n\nFinal answer: (A)", "(B)", 0.0),
        # Model restates an intermediate guess, then commits — last wins.
        ("Maybe (A)? No. The answer is (C).", "(C)", 1.0),
        # Parentheses dropped by the model — accept lone letter in the span.
        ("\n\nFinal answer: D", "(D)", 1.0),
        # 'answer is (Q)' style, double-digit option range (A..R).
        ("Reasoning...\n\nFinal answer: (Q)", "(Q)", 1.0),
        # Span has prose before the letter.
        ("\n\nFinal answer: The correct option is (E).", "(E)", 1.0),
        # No locatable letter -> miss.
        ("\n\nFinal answer: I am not sure.", "(B)", 0.0),
    ],
)
def test_multiple_choice(generation, target, expected):
    assert bbh.verify(generation, target) == expected


# --- boolean / token: True/False, Yes/No, valid/invalid -------------------

@pytest.mark.parametrize(
    "generation, target, expected",
    [
        ("\n\nFinal answer: True", "True", 1.0),
        ("\n\nFinal answer: False", "True", 0.0),
        ("evaluating...\n\nFinal answer: Yes", "Yes", 1.0),
        ("\n\nFinal answer: No", "Yes", 0.0),
        ("\n\nFinal answer: valid", "valid", 1.0),
        # 'invalid' must NOT count as a match for gold 'valid' (substring trap).
        ("\n\nFinal answer: invalid", "valid", 0.0),
        ("\n\nFinal answer: invalid", "invalid", 1.0),
        # Gold token embedded in a short phrase.
        ("\n\nFinal answer: the answer is yes", "yes", 1.0),
        # Case-insensitive.
        ("\n\nFinal answer: NO", "No", 1.0),
    ],
)
def test_boolean_token(generation, target, expected):
    assert bbh.verify(generation, target) == expected


# --- count: bare integer (object_counting) --------------------------------

@pytest.mark.parametrize(
    "generation, target, expected",
    [
        ("I count 1, 2, ... that's 8 items.\n\nFinal answer: 8", "8", 1.0),
        ("\n\nFinal answer: 15", "15", 1.0),
        ("\n\nFinal answer: 7", "8", 0.0),
        # Intermediate counts present; final commit is what matters.
        ("So far 3, then 4 more.\n\nFinal answer: 7", "7", 1.0),
    ],
)
def test_count(generation, target, expected):
    assert bbh.verify(generation, target) == expected


def test_no_trigger_falls_back_to_last_line():
    # Budget forcing always appends "Final answer:", but be robust if the
    # model emits a bare last line instead.
    assert bbh.verify("Step 1 ...\nStep 2 ...\n(C)", "(C)") == 1.0
    assert bbh.verify("reasoning\nNo", "No") == 1.0


def test_registered_in_interface():
    # The dispatch layer must route 'bbh' to this verifier.
    assert verify_dispatch("bbh", "\n\nFinal answer: (A)", "(A)") == 1.0
    assert "bbh" in __import__(
        "src.verification.interface", fromlist=["known_datasets"]
    ).known_datasets()
