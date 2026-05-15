"""Unit tests for the unified verifier registry.

Covers dispatch correctness for each registered dataset, case- and
whitespace-insensitive lookup, error handling for unknown datasets,
the listing API, and uniform return-type contract.
"""

from __future__ import annotations

import pytest

from src.verification import gsm8k, humaneval
from src.verification import math as math_verifier
from src.verification.interface import get_verifier, known_datasets, verify


# ---------------------------------------------------------------------------
# Dispatch correctness — each dataset routes to the right verifier
# ---------------------------------------------------------------------------


def test_dispatches_to_gsm8k() -> None:
    assert verify("gsm8k", "#### 42", "42") == 1.0
    assert verify("gsm8k", "#### 42", "43") == 0.0


def test_dispatches_to_math() -> None:
    assert verify("math", "\\boxed{\\frac{1}{2}}", "0.5") == 1.0
    assert verify("math", "\\boxed{42}", "43") == 0.0


def test_dispatches_to_humaneval() -> None:
    gen = "def candidate_func():\n    return 1\n"
    gt = "def check(c):\n    assert c() == 1\ncheck(candidate_func)\n"
    assert verify("humaneval", gen, gt) == 1.0


# ---------------------------------------------------------------------------
# Lookup ergonomics
# ---------------------------------------------------------------------------


def test_lookup_is_case_insensitive() -> None:
    assert verify("GSM8K", "#### 42", "42") == 1.0
    assert verify("Math", "\\boxed{42}", "42") == 1.0
    assert verify("HumanEval", "x=1\n", "assert x == 1\n") == 1.0


def test_lookup_strips_whitespace() -> None:
    assert verify("  gsm8k  ", "#### 42", "42") == 1.0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_dataset_raises_key_error() -> None:
    with pytest.raises(KeyError) as excinfo:
        verify("unknown_dataset", "x", "y")
    assert "unknown_dataset" in str(excinfo.value)


def test_unknown_dataset_error_lists_known_names() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_verifier("nope")
    msg = str(excinfo.value)
    for name in ("gsm8k", "math", "humaneval"):
        assert name in msg


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


def test_known_datasets_lists_all_registered() -> None:
    assert set(known_datasets()) == {
        "gsm8k", "math", "humaneval",
        "matharena", "livecodebench",
    }


def test_known_datasets_is_sorted_tuple() -> None:
    names = known_datasets()
    assert isinstance(names, tuple)
    assert list(names) == sorted(names)


def test_get_verifier_returns_correct_callable() -> None:
    assert get_verifier("gsm8k") is gsm8k.verify
    assert get_verifier("math") is math_verifier.verify
    assert get_verifier("humaneval") is humaneval.verify


# ---------------------------------------------------------------------------
# Uniform return-type contract across registered verifiers
# ---------------------------------------------------------------------------


def test_all_registered_verifiers_return_float_on_empty_inputs() -> None:
    """Every registered verifier should accept ``(str, str)`` and return
    ``float`` even on empty inputs (which uniformly map to 0.0)."""
    for name in known_datasets():
        result = verify(name, "", "")
        assert isinstance(result, float), f"{name} returned {type(result).__name__}"
        assert result == 0.0, f"{name} returned {result} for empty inputs"


# ---------------------------------------------------------------------------
# Top-level package re-export
# ---------------------------------------------------------------------------


def test_top_level_package_exposes_verify() -> None:
    """``from src.verification import verify`` should work and resolve
    to the same dispatcher."""
    from src.verification import verify as pkg_verify

    assert pkg_verify is verify
