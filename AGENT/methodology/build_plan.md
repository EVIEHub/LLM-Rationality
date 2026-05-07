---
name: Build phasing — verifier-first plan (mostly historical)
description: Three-phase build plan agreed 2026-05-07. Verifiers and their tests came first (no GPU), then pipeline plumbing, then sampling and end-to-end smoke test.
type: project
---

This is largely historical now — Phases 1 and 2 are complete and 3.9 is complete. Recorded for future agents who want to understand *why* the codebase has its current shape.

**Phase 1 (no GPU):** ✅ done
1. ✅ `src/verification/gsm8k.py` + `tests/test_gsm8k.py` (37 tests)
2. ✅ `src/verification/math.py` + `tests/test_math.py` (40 tests; uses `math-verify`)
3. ✅ `src/verification/humaneval.py` + `tests/test_humaneval.py` (21 tests; subprocess sandbox, 5s timeout)
4. ✅ `src/verification/interface.py` — unified verifier registry (12 tests)

**Phase 2 (no GPU):** ✅ done
5. ✅ `src/pipeline/paths.py` — load `configs/paths.yaml` (13 tests)
6. ✅ `src/pipeline/cache.py` — gzipped JSONL with versioned cache keys (18 tests)
7. ✅ `src/pipeline/logging_utils.py` — run logger + verifier audit log + compute log (12 tests)

**Phase 3 (GPU — partial):** 1 of 3 done
8. ⏸ `src/sampling/vllm_runner.py` — thin vLLM wrapper; **deferred to GPU box**
9. ✅ `src/metrics/rational_gap.py` — $\hat{\mathcal{R}}_K$ + bootstrap CI + saturation-curve helpers (30 tests)
10. ⏸ `scripts/smoke_test.py` — Qwen-1.5B on 10 GSM8K prompts at $K=4$, end-to-end; **deferred to GPU box**

**Why:** Verifiers are the highest-risk component (false-negative risk dominates measurement error per the audit memo) and the easiest to test in isolation without GPUs. The user explicitly ordered this phasing. With the GPU phases now blocked, GPU-prep work (configs/, conftest fixtures, the H1 experiment config) was done in advance so the GPU session has a clear runway.

**How to apply:** When new agents pick up the GPU work, do **not** retroactively rebuild Phase 1/2; trust the existing tests and read [TODO.md](../TODO.md) for the current entry point.
