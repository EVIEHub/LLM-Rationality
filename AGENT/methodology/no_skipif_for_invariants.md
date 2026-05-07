---
name: No skipif on load-bearing methodological invariants
description: Tests that exercise paper-load-bearing invariants (determinism, equivalence, etc.) must run live in their target environment — never gated by @pytest.mark.skipif.
type: feedback
---

Rule: Never gate a test of a load-bearing methodological invariant behind `@pytest.mark.skipif(...)`. If the invariant cannot be tested in the current environment, pause the build and resume in the target environment rather than ship a skipped test.

**Why:** A skipped test produces a green CI badge while leaving the actual guarantee unverified. For a paper-citable codebase, "correctness > convenience" (AGENT.md §1) means we accept build delay rather than a verification gap. The user explicitly stated this on 2026-05-07 when offered the choice between (a) shipping the vLLM determinism test behind `skipif(no_gpu)` or (b) deferring 3.8 entirely to a GPU box: "The determinism invariant is too important to ship behind skipif."

**How to apply:**
- Examples of load-bearing invariants in this codebase: vLLM determinism with fixed seed (AGENT.md §3.1), `R_hat_1 ≡ 0` closed form, math-verify symbolic equivalence on canonical LaTeX cases, verifier audit-log completeness.
- When such a test is environment-blocked, propose pausing the relevant build step rather than adding skipif. If the user explicitly chooses skipif anyway, save *that* choice as a deviation note — but do not assume it.
- Routine environment markers (e.g. skipping a Linux-only test on Windows when the project targets Linux only, skipping a slow performance benchmark on PR CI) are fine — those are not paper-load-bearing.
