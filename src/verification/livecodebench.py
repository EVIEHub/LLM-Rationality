"""LiveCodeBench verifier (Jain et al. 2024, contest_date filtered).

LiveCodeBench tests come in two flavours per row:

  - ``stdin`` testtype: the candidate program is run as a top-level
    script; ``input`` is piped to stdin and the program's stdout must
    match ``output`` (modulo trailing whitespace). Typical for AtCoder /
    Codeforces tasks.
  - ``functional`` testtype: the candidate defines a ``Solution`` class
    with a method. The verifier instantiates ``Solution()``, calls the
    method on parsed arguments, and compares the return value to the
    expected output. Typical for LeetCode tasks.

The pipeline (in scripts/run_h1.py) packages the per-row test cases
into a JSON string under the synthetic ``test_cases_json`` field
(combining ``public_test_cases`` + ``private_test_cases`` + the row's
``starter_code`` for functional dispatch). That JSON is the
``ground_truth`` argument here.

Returns ``1.0`` only if **all** test cases pass within the timeout,
``0.0`` otherwise.

Sandboxing follows the AGENT.md §3.3 contract: temp cwd, full
environment passthrough (LiveCodeBench tests freely use sys, math,
collections, etc.), new process session, wall-clock timeout per test.
Code is **never** ``exec()``'d in the parent process.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

_PER_TEST_TIMEOUT_SECONDS: float = 10.0
_TOTAL_TIMEOUT_SECONDS: float = 60.0


def _extract_python(text: str) -> str:
    """Strip Markdown code fences if present; otherwise return the text."""
    # Common patterns from chat models: ```python\n...\n```
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    return text


def _run_stdin_test(
    code: str, test_input: str, test_output: str, timeout: float
) -> bool:
    """Pipe ``test_input`` to ``code`` via stdin; compare stdout to ``test_output``."""
    expected = test_output.rstrip("\n").rstrip()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                input=test_input,
                timeout=timeout,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                cwd=tmpdir,
                start_new_session=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
    if result.returncode != 0:
        return False
    actual = result.stdout.rstrip("\n").rstrip()
    return actual == expected


def _build_functional_invoker(code: str, starter_code: str) -> str:
    """Build a small driver appended to ``code`` that reads
    JSON-encoded args from stdin and prints the return value of
    ``Solution().<method>(*args)``.

    The method name is discovered from ``starter_code`` (the
    ``def <method>(self, ...)`` line)."""
    m = re.search(r"def\s+(\w+)\s*\(\s*self", starter_code)
    if not m:
        return ""  # Caller will short-circuit to 0.
    method_name = m.group(1)
    driver = f"""

if __name__ == "__main__":
    import sys, json
    _args_raw = sys.stdin.read().strip()
    _args = json.loads(_args_raw) if _args_raw.startswith(("[", "{{")) else [_args_raw]
    _instance = Solution()
    _result = _instance.{method_name}(*_args) if isinstance(_args, list) else _instance.{method_name}(**_args)
    print(json.dumps(_result))
"""
    return code + "\n" + driver


def _run_functional_test(
    code: str, starter_code: str, test_input: str, test_output: str, timeout: float
) -> bool:
    """Run a Solution()-style functional test."""
    program = _build_functional_invoker(code, starter_code)
    if not program:
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                [sys.executable, "-c", program],
                input=test_input,
                timeout=timeout,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                cwd=tmpdir,
                start_new_session=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
    if result.returncode != 0:
        return False
    actual = result.stdout.strip()
    # Compare as JSON-decoded values when both sides parse; fall back to
    # stripped-string equality (LeetCode test outputs are usually one of
    # int / list / str literals).
    try:
        return json.loads(actual) == json.loads(test_output.strip())
    except (json.JSONDecodeError, ValueError):
        return actual == test_output.strip()


def verify(generation: str, ground_truth: str) -> float:
    """Verify a LiveCodeBench generation against its test bundle.

    Args:
        generation: The model's code (may contain Markdown fences).
        ground_truth: A JSON string with the shape
            ``{"tests": [{"input", "output", "testtype"}, ...],
               "starter_code": "<...>"}``.

    Returns:
        ``1.0`` iff every test case passes within its timeout; ``0.0``
        otherwise (test failure, timeout, parse failure, malformed
        ``ground_truth``).
    """
    if not ground_truth.strip():
        return 0.0
    try:
        bundle = json.loads(ground_truth)
    except json.JSONDecodeError:
        return 0.0
    tests = bundle.get("tests", [])
    starter_code = bundle.get("starter_code", "")
    if not tests:
        return 0.0

    code = _extract_python(generation)

    import time
    t0 = time.time()
    for t in tests:
        if time.time() - t0 > _TOTAL_TIMEOUT_SECONDS:
            return 0.0
        test_input = t.get("input", "")
        test_output = t.get("output", "")
        testtype = t.get("testtype", "stdin")
        if testtype == "stdin":
            ok = _run_stdin_test(
                code, test_input, test_output, _PER_TEST_TIMEOUT_SECONDS,
            )
        elif testtype == "functional":
            ok = _run_functional_test(
                code, starter_code, test_input, test_output,
                _PER_TEST_TIMEOUT_SECONDS,
            )
        else:
            return 0.0  # Unknown testtype.
        if not ok:
            return 0.0
    return 1.0
