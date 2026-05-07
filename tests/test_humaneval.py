"""Unit tests for the HumanEval verifier.

Covers correct/incorrect solutions, error paths (syntax, runtime,
undefined name, failed assertion, empty), the 5-second timeout
contract, environment and cwd isolation, and the float-typed return.
"""

from __future__ import annotations

import os

from src.verification.humaneval import verify


# ---------------------------------------------------------------------------
# Correct / incorrect solutions
# ---------------------------------------------------------------------------


def test_correct_solution_returns_one(humaneval_check_factory) -> None:
    gen = "def candidate_func(a, b):\n    return a + b\n"
    gt = humaneval_check_factory("assert candidate(2, 3) == 5\nassert candidate(0, 0) == 0\n")
    assert verify(gen, gt) == 1.0


def test_wrong_solution_returns_zero(humaneval_check_factory) -> None:
    gen = "def candidate_func(a, b):\n    return a - b\n"
    gt = humaneval_check_factory("assert candidate(2, 3) == 5\n")
    assert verify(gen, gt) == 0.0


def test_multiple_assertions_all_must_pass(humaneval_check_factory) -> None:
    gen = "def candidate_func(x):\n    return x * 2\n"
    gt = humaneval_check_factory(
        "assert candidate(1) == 2\n"
        "assert candidate(2) == 4\n"
        "assert candidate(3) == 6\n"
    )
    assert verify(gen, gt) == 1.0


def test_one_failing_assertion_fails_overall(humaneval_check_factory) -> None:
    gen = "def candidate_func(x):\n    return x * 2 if x != 3 else 999\n"
    gt = humaneval_check_factory(
        "assert candidate(1) == 2\n"
        "assert candidate(2) == 4\n"
        "assert candidate(3) == 6\n"
    )
    assert verify(gen, gt) == 0.0


def test_solution_with_imports(humaneval_check_factory) -> None:
    gen = "import math\ndef candidate_func(x):\n    return math.sqrt(x)\n"
    gt = humaneval_check_factory("assert abs(candidate(4) - 2.0) < 1e-9\n")
    assert verify(gen, gt) == 1.0


def test_solution_with_helper_function(humaneval_check_factory) -> None:
    gen = (
        "def _helper(x):\n    return x * x\n"
        "def candidate_func(x):\n    return _helper(x) + 1\n"
    )
    gt = humaneval_check_factory("assert candidate(3) == 10\nassert candidate(0) == 1\n")
    assert verify(gen, gt) == 1.0


def test_recursive_solution(humaneval_check_factory) -> None:
    gen = (
        "def candidate_func(n):\n"
        "    return 1 if n <= 1 else n * candidate_func(n - 1)\n"
    )
    gt = humaneval_check_factory("assert candidate(5) == 120\nassert candidate(0) == 1\n")
    assert verify(gen, gt) == 1.0


def test_solution_returning_list(humaneval_check_factory) -> None:
    gen = "def candidate_func(n):\n    return list(range(n))\n"
    gt = humaneval_check_factory("assert candidate(3) == [0, 1, 2]\nassert candidate(0) == []\n")
    assert verify(gen, gt) == 1.0


def test_string_manipulation_solution(humaneval_check_factory) -> None:
    gen = "def candidate_func(s):\n    return s.upper()\n"
    gt = humaneval_check_factory("assert candidate('hi') == 'HI'\n")
    assert verify(gen, gt) == 1.0


# ---------------------------------------------------------------------------
# Error paths: syntax, runtime, undefined name, empty
# ---------------------------------------------------------------------------


def test_syntax_error_returns_zero(humaneval_check_factory) -> None:
    gen = "def candidate_func(a, b)\n    return a + b\n"  # missing colon
    gt = humaneval_check_factory("assert candidate(2, 3) == 5\n")
    assert verify(gen, gt) == 0.0


def test_runtime_exception_returns_zero(humaneval_check_factory) -> None:
    gen = "def candidate_func(a, b):\n    raise ValueError('oops')\n"
    gt = humaneval_check_factory("assert candidate(2, 3) == 5\n")
    assert verify(gen, gt) == 0.0


def test_undefined_function_returns_zero(humaneval_check_factory) -> None:
    gen = "x = 42\n"  # never defines candidate_func
    gt = humaneval_check_factory("assert candidate(2, 3) == 5\n")
    assert verify(gen, gt) == 0.0


def test_empty_generation_returns_zero(humaneval_check_factory) -> None:
    gt = humaneval_check_factory("assert candidate(2, 3) == 5\n")
    assert verify("", gt) == 0.0


def test_empty_ground_truth_returns_zero() -> None:
    """An empty test program exits 0 vacuously; we must fail closed
    rather than report a vacuous pass."""
    gen = "def candidate_func():\n    return 1\n"
    assert verify(gen, "") == 0.0
    assert verify(gen, "   \n  ") == 0.0  # whitespace-only also fails closed
    assert verify("", "") == 0.0


def test_failing_bare_assertion_returns_zero(humaneval_check_factory) -> None:
    gen = "def candidate_func():\n    return 0\n"
    gt = humaneval_check_factory("assert candidate() == 1\n")
    assert verify(gen, gt) == 0.0


# ---------------------------------------------------------------------------
# Timeout contract (AGENT.md §3.3: 5-second timeout)
# ---------------------------------------------------------------------------


def test_timeout_returns_zero(humaneval_check_factory, monkeypatch) -> None:
    """An infinite loop in the candidate must yield 0.0, not hang the run.

    We patch the module-level timeout down to 0.5s to keep the test fast.
    """
    import src.verification.humaneval as hv

    monkeypatch.setattr(hv, "_TIMEOUT_SECONDS", 0.5)
    gen = "def candidate_func():\n    \n    while True:\n        pass\n"
    gt = humaneval_check_factory("candidate()\n")
    assert verify(gen, gt) == 0.0


def test_default_timeout_is_five_seconds() -> None:
    """Per AGENT.md §3.3, the documented timeout is 5 seconds."""
    import src.verification.humaneval as hv

    assert hv._TIMEOUT_SECONDS == 5.0


# ---------------------------------------------------------------------------
# Sandbox isolation: cwd, env
# ---------------------------------------------------------------------------


def test_does_not_pollute_parent_cwd(humaneval_check_factory, tmp_path, monkeypatch) -> None:
    """A candidate that writes a file should not leave artefacts in the
    caller's cwd — the subprocess runs in its own temp dir."""
    monkeypatch.chdir(tmp_path)
    pre = set(os.listdir("."))
    gen = (
        "def candidate_func():\n"
        "    open('side_effect.txt', 'w').write('boo')\n"
        "    return 1\n"
    )
    gt = humaneval_check_factory("assert candidate() == 1\n")
    verify(gen, gt)
    post = set(os.listdir("."))
    assert pre == post


def test_parent_env_vars_do_not_leak(humaneval_check_factory, monkeypatch) -> None:
    """A custom env var set in the parent must not be visible inside the
    subprocess — only PATH is forwarded."""
    monkeypatch.setenv("RATIONAL_GAP_TEST_LEAK_CANARY", "leaked")
    gen = (
        "import os\n"
        "def candidate_func():\n"
        "    return os.environ.get('RATIONAL_GAP_TEST_LEAK_CANARY', 'not-leaked')\n"
    )
    gt = humaneval_check_factory("assert candidate() == 'not-leaked'\n")
    assert verify(gen, gt) == 1.0


# ---------------------------------------------------------------------------
# Return-type contract from AGENT.md §3.3
# ---------------------------------------------------------------------------


def test_return_type_is_float_on_pass(humaneval_check_factory) -> None:
    gen = "def candidate_func():\n    return 1\n"
    gt = humaneval_check_factory("assert candidate() == 1\n")
    result = verify(gen, gt)
    assert isinstance(result, float)
    assert result == 1.0


def test_return_type_is_float_on_fail(humaneval_check_factory) -> None:
    gen = "def candidate_func():\n    return 0\n"
    gt = humaneval_check_factory("assert candidate() == 1\n")
    result = verify(gen, gt)
    assert isinstance(result, float)
    assert result == 0.0
