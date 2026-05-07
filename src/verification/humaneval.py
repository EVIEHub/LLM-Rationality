"""HumanEval verifier: execute the model's code completion concatenated
with the ground-truth check program in a sandboxed subprocess. Returns
1.0 if the subprocess exits 0 within the timeout, else 0.0.

Per AGENT.md §3.3 the model output is **never** ``exec()``'d in the
parent process. We use a subprocess with:

- a 5-second wall-clock timeout (``subprocess.run(timeout=...)``);
- a fresh temporary working directory, so the candidate cannot read or
  pollute the repository or outputs tree;
- a minimal environment containing only ``PATH``, so secrets and
  HuggingFace tokens in the parent environment do not leak into the
  candidate's process;
- a new session (``start_new_session=True``), so SIGINT in the parent
  does not propagate, and the subprocess can be cleanly torn down.

This is **not** a full sandbox: a malicious candidate can still touch
the network, read ``/tmp``, fork (and any forked grandchildren outlive
the timeout), or consume CPU/RAM up to the OS limits. The HumanEval
benchmark is composed of benign algorithmic problems, so the threat
model is "buggy or runaway code", not "adversarial code".

The ``ground_truth`` argument is the check program that, when appended
to the generation, asserts correctness — typically a ``check(candidate)``
function plus a final ``check(<entry_point>)`` call. The HumanEval data
loader is responsible for assembling that string.

This module is a pure function modulo the bounded subprocess side
effect. No I/O outside the subprocess's temp dir, no global state, no
logging — the audit log mandated by AGENT.md §3.3 is written by the
caller.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

# Per AGENT.md §3.3.
_TIMEOUT_SECONDS: float = 5.0


def _restricted_env() -> dict[str, str]:
    """Minimal environment for the candidate subprocess.

    Only ``PATH`` is forwarded, so the Python interpreter can find any
    shared libraries it needs at startup. Secrets, ``HOME``,
    ``PYTHONPATH``, and HuggingFace tokens in the parent environment
    are deliberately not propagated.
    """
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def verify(generation: str, ground_truth: str) -> float:
    """Verify a HumanEval generation by executing it against a check program.

    Args:
        generation: The model's code completion (a Python source string
            that should define the target function).
        ground_truth: A Python source string that, when executed *after*
            the generation, asserts correctness. Must include the call
            that exercises the candidate function (e.g.
            ``check(<entry_point>)``).

    Returns:
        ``1.0`` if the combined program exits 0 within the timeout,
        else ``0.0``. Returns ``0.0`` on timeout, non-zero exit (failed
        assertion, raised exception, syntax error, undefined name),
        any subprocess-level error, or an empty ``ground_truth`` (an
        empty test program would exit 0 vacuously; we fail closed).
    """
    if not ground_truth.strip():
        return 0.0
    program = generation + "\n" + ground_truth

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                [sys.executable, "-c", program],
                timeout=_TIMEOUT_SECONDS,
                capture_output=True,
                env=_restricted_env(),
                cwd=tmpdir,
                start_new_session=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 0.0
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0

    return 1.0 if result.returncode == 0 else 0.0
