"""Tests for src.verification.livecodebench (stdin + functional kernels)."""

from __future__ import annotations

import json

import pytest

from src.verification.livecodebench import _extract_python, verify


def _bundle(tests: list[dict], starter_code: str = "") -> str:
    return json.dumps({"tests": tests, "starter_code": starter_code})


class TestExtractPython:
    def test_strip_python_fence(self) -> None:
        assert _extract_python("```python\nprint(42)\n```") == "print(42)\n"

    def test_strip_bare_fence(self) -> None:
        assert _extract_python("```\nprint(42)\n```") == "print(42)\n"

    def test_no_fence_returns_text(self) -> None:
        assert _extract_python("print(42)") == "print(42)"


# ---------------------------------------------------------------------------
# stdin tests
# ---------------------------------------------------------------------------

class TestStdin:
    def test_passing(self) -> None:
        gen = "n = int(input())\nprint(n * 2)"
        bundle = _bundle([
            {"input": "5\n", "output": "10\n", "testtype": "stdin"},
            {"input": "7\n", "output": "14\n", "testtype": "stdin"},
        ])
        assert verify(gen, bundle) == 1.0

    def test_one_failing_returns_zero(self) -> None:
        gen = "n = int(input())\nprint(n * 2)"
        bundle = _bundle([
            {"input": "5\n", "output": "10\n", "testtype": "stdin"},
            {"input": "7\n", "output": "13\n", "testtype": "stdin"},  # wrong
        ])
        assert verify(gen, bundle) == 0.0

    def test_runtime_error_returns_zero(self) -> None:
        gen = "1 / 0"
        bundle = _bundle([
            {"input": "", "output": "anything", "testtype": "stdin"},
        ])
        assert verify(gen, bundle) == 0.0

    def test_strips_trailing_whitespace(self) -> None:
        # Expected has trailing newline; actual has trailing newline; should match.
        gen = "print(42)"
        bundle = _bundle([
            {"input": "", "output": "42\n", "testtype": "stdin"},
        ])
        assert verify(gen, bundle) == 1.0


# ---------------------------------------------------------------------------
# functional tests
# ---------------------------------------------------------------------------

class TestFunctional:
    def test_passing_int_return(self) -> None:
        gen = """
class Solution:
    def add(self, a, b):
        return a + b
"""
        starter = "class Solution:\n    def add(self, a, b):\n        pass"
        bundle = _bundle(
            [{"input": "[2, 3]", "output": "5", "testtype": "functional"}],
            starter_code=starter,
        )
        assert verify(gen, bundle) == 1.0

    def test_failing(self) -> None:
        gen = """
class Solution:
    def add(self, a, b):
        return a - b   # wrong
"""
        starter = "class Solution:\n    def add(self, a, b):\n        pass"
        bundle = _bundle(
            [{"input": "[2, 3]", "output": "5", "testtype": "functional"}],
            starter_code=starter,
        )
        assert verify(gen, bundle) == 0.0

    def test_list_return(self) -> None:
        gen = """
class Solution:
    def doubled(self, xs):
        return [x * 2 for x in xs]
"""
        starter = "class Solution:\n    def doubled(self, xs):\n        pass"
        bundle = _bundle(
            [{"input": "[[1, 2, 3]]", "output": "[2, 4, 6]",
              "testtype": "functional"}],
            starter_code=starter,
        )
        assert verify(gen, bundle) == 1.0

    def test_missing_starter_method_returns_zero(self) -> None:
        gen = "class Solution: pass"
        bundle = _bundle(
            [{"input": "[1]", "output": "1", "testtype": "functional"}],
            starter_code="",  # no method to discover
        )
        assert verify(gen, bundle) == 0.0


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_ground_truth(self) -> None:
        assert verify("print(42)", "") == 0.0

    def test_malformed_bundle(self) -> None:
        assert verify("print(42)", "not json") == 0.0

    def test_no_tests(self) -> None:
        assert verify("print(42)", json.dumps({"tests": []})) == 0.0

    def test_unknown_testtype(self) -> None:
        bundle = _bundle([{"input": "", "output": "", "testtype": "exotic"}])
        assert verify("print(42)", bundle) == 0.0
