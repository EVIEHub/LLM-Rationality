# Evaluator Report — `rational-gap-of-LLM-reasoning`

**Evaluator:** automated review against [AGENT/AGENT.md](AGENT/AGENT.md) and [AGENT/TODO.md](AGENT/TODO.md)
**Date:** 2026-05-07
**Scope:** read-only audit of the repository as it stands. No code was modified.
**Tests:** `pytest tests/` → **153 passed in ~5 s** (locally reproduced).

---

## 1. Verdict (one paragraph)

The shipped code is a careful, well-documented Phase-1+early-Phase-2 build that adheres tightly to the AGENT.md hard rules. Every module visible on disk has matching tests with positive/negative/edge coverage, and the design choices recorded in module docstrings line up with the methodology commitments in AGENT.md. The most material gap is **between TODO.md and reality**: the TODO claims "Phase 1 complete (steps 1–4 of 10), 110 tests passing", but Phase 2.5 (paths), 2.6 (cache), and 2.7 (logging) are also already shipped with green tests, and the actual test count is 153. The README is forward-looking (it documents components that have not yet been built); this is acceptable as a paper-facing artefact but should not mislead readers into thinking sampling/scripts/figures already work. Verdict: **methodologically sound, code quality high, documentation drift is the only real issue.**

---

## 2. Compliance with AGENT.md (section-by-section)

| AGENT.md §                           | Status     | Evidence                                                                                                                                                                                                                                                                |
| ------------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §3.1 Sampling                        | N/A (Phase 3) | No `src/sampling/` yet — correctly deferred per the phased plan.                                                                                                                                                                                                       |
| §3.2 Caching (key, immutability)     | ✅ Pass    | [src/pipeline/cache.py:53-62](src/pipeline/cache.py#L53-L62) `CacheKey` carries all 9 required fields; [src/pipeline/cache.py:128-134](src/pipeline/cache.py#L128-L134) refuses overwrite; atomic `.tmp` + `replace`; [tests/test_cache.py](tests/test_cache.py) verifies. |
| §3.3 Verifier purity & float return  | ✅ Pass    | All three verifiers are pure (no I/O / globals); audit logging is at the call site via [src/pipeline/logging_utils.py:92-116](src/pipeline/logging_utils.py#L92-L116). Return-type contract enforced by parametric tests in [tests/test_interface.py:98-104](tests/test_interface.py#L98-L104). |
| §3.3 HumanEval sandboxing            | ✅ Pass    | [src/verification/humaneval.py:78-92](src/verification/humaneval.py#L78-L92): subprocess + 5 s timeout + temp cwd + minimal env + `start_new_session=True`. Honestly disclaimed as "not a full sandbox".                                                                |
| §3.3 SymPy timeouts must not raise   | ✅ Pass    | [src/verification/math.py:116-138](src/verification/math.py#L116-L138) wraps both `math_verify.parse` and `math_verify.verify` in `try/except`; covered by `test_verify_returns_zero_on_*` in [tests/test_math.py:127-152](tests/test_math.py#L127-L152).               |
| §3.4 Paths / no hardcoding           | ✅ Pass    | [src/pipeline/paths.py](src/pipeline/paths.py) is the single source of truth; `pathlib.Path` everywhere; outputs configured outside repo via `paths.yaml`.                                                                                                              |
| §3.5 Reproducibility                 | N/A (Phase 3) | No entry-point scripts yet.                                                                                                                                                                                                                                            |
| §3.6 Logging                         | ✅ Pass    | `setup_run_logger`, `log_verifier_decision`, `log_compute` all present and tested.                                                                                                                                                                                       |
| §4.2 Forbidden alternatives          | ✅ Pass    | `grep -E 'pass_at_k\|pass@k\|pass1\|pass_rate\|accuracy\|success_rate'` over `src/`, `tests/`, `configs/` returns no hits.                                                                                                                                              |
| §4.2 Paper-symbol naming             | ✅ Pass    | `K`, `U`, `R_hat_K`, `U_circ_K`, `U_bar_K` referenced in docstrings. Code uses `K` for sample budget.                                                                                                                                                                    |
| §5 Bare `except:`                    | ✅ Pass    | Source tree contains no bare `except`. Specific exception classes only (`subprocess.TimeoutExpired`, `OSError`, `ValueError`, `json.JSONDecodeError`, `RuntimeError`, etc.).                                                                                              |
| §6.2 ≥10+10 verifier tests           | ✅ Exceeds | GSM8K 14+11, MATH 14+11, HumanEval 9+6 (with extra error/sandbox cases). All three datasets exceed the floor.                                                                                                                                                            |
| §10 Scope guards                     | ✅ Pass    | No training/fine-tuning code, no reward models, binary `U` only.                                                                                                                                                                                                          |

---

## 3. Code Quality — module-by-module

### 3.1 [src/verification/gsm8k.py](src/verification/gsm8k.py)
- Four-pattern extraction chain (`####` → `\boxed{}` → "the answer is X" → bare last number) matches AGENT.md §8.2 exactly.
- Rightmost-match wins per pattern — appropriate because models often restate intermediate numbers.
- Numeric normalisation strips commas, leading `$`, trailing `%`. Comparison via `math.isclose(rel_tol=1e-9, abs_tol=1e-9)` so `72` ↔ `72.0`.
- One subtle observation: `math.isclose(rel_tol=1e-9, abs_tol=1e-9)` is strict enough that floating-point answers produced by intermediate rounding (e.g. `0.1 + 0.2 == 0.30000000000000004` vs. ground-truth `0.3`) can fail. For GSM8K this is fine — answers are integer or short-decimal. Worth re-examining if the verifier is ever reused for tasks with computed decimals.

### 3.2 [src/verification/math.py](src/verification/math.py)
- The decision to do `\boxed{...}` extraction locally before delegating to `math-verify` is correct and well-justified in the docstring (parser silently truncates `2\sqrt{3}` → `2` without an anchor).
- Balanced-brace walker handles nested braces, escape sequences, and falls back from rightmost-unbalanced to earlier balanced openings — solid.
- Defence-in-depth around `math_verify.parse` and `math_verify.verify` is appropriate; covered by monkey-patched exception tests.
- Empty `\boxed{}` and empty ground truth both fail closed.

### 3.3 [src/verification/humaneval.py](src/verification/humaneval.py)
- Subprocess hardening is correct: temp cwd, minimal env (`PATH` only), `start_new_session=True`, 5 s timeout. `_TIMEOUT_SECONDS` is a module constant so tests can monkey-patch it down.
- Empty-ground-truth → `0.0` is a load-bearing fail-closed (vacuous-pass bug noted in TODO; regression test `test_empty_ground_truth_returns_zero` covers it).
- Honestly documented as "not a full sandbox" (network, `/tmp`, fork bombs, RAM limits not constrained). Consistent with the HumanEval threat model of buggy code rather than adversarial code.
- Test `test_parent_env_vars_do_not_leak` actively asserts the env restriction works — this is a strong test, not a paper assumption.

### 3.4 [src/verification/interface.py](src/verification/interface.py)
- Registry is small and explicit; case-insensitive + whitespace-stripping lookup.
- `KeyError` for unknown datasets surfaces the registered names (self-diagnosing failure).
- Uniform return-contract test (`test_all_registered_verifiers_return_float_on_empty_inputs`) is exactly the right shape for a registry — it caught the HumanEval vacuous-pass bug per TODO §1.4.

### 3.5 [src/pipeline/cache.py](src/pipeline/cache.py)
- `CacheKey` is a frozen dataclass with `__post_init__` numeric canonicalisation so `temperature=1` and `temperature=1.0` produce identical fingerprints — good defence against silent cache misses from incidental type drift.
- 16-hex SHA-256 prefix (~64 bits) is sufficient for cache-key fingerprinting; embedded full-key header in the file is the actual collision defence.
- Atomic write (`.tmp` + `replace`) with exception-safe cleanup using `BaseException`. The broad catch is justified here — it's a finally-style cleanup, not an error swallow, and the exception is re-raised.
- Read path validates header version and full key; 16 high-quality test cases including negative cases (mismatched key, format mismatch, empty file, malformed header).

### 3.6 [src/pipeline/paths.py](src/pipeline/paths.py)
- Recursive interpolation with cycle detection (`seen` frozenset) and missing-name reporting.
- Outer fixed-point loop is mildly redundant given that `_resolve_string` already recurses inside the substitution callback. This is harmless — recursion fully resolves any value in one pass — but a future tidy-up could collapse the two layers.
- Tilde expansion happens at `Path.expanduser()` call sites in `load_paths`, consistent with the docstring.
- Tests cover happy paths, the shipped template file, missing keys, undefined references, cycles, and non-mapping roots.

### 3.7 [src/pipeline/logging_utils.py](src/pipeline/logging_utils.py)
- `setup_run_logger` correctly closes and removes pre-existing handlers (otherwise repeated entry-point invocations would accumulate file descriptors).
- File handler captures DEBUG; stream handler is configurable. Matches AGENT.md §3.6 exactly.
- `log_verifier_decision` and `log_compute` are pure-append JSONL writers — simple, durable, easy to audit. UTC timestamps by default.

---

## 4. Documentation Drift — issues to flag

These are not code bugs, but they will mislead a reader who trusts the docs.

### 4.1 [AGENT/TODO.md](AGENT/TODO.md) is stale
- Header claims **"Phase 1 complete (steps 1–4 of 10). 110 tests passing."** Actual state on disk:
  - Phase 2.5 (`paths.py`) — **shipped**, 14 tests in [tests/test_paths.py](tests/test_paths.py).
  - Phase 2.6 (`cache.py`) — **shipped**, 16 tests in [tests/test_cache.py](tests/test_cache.py).
  - Phase 2.7 (`logging_utils.py`) — **shipped**, 12 tests in [tests/test_logging_utils.py](tests/test_logging_utils.py).
  - Total tests: **153 passing**, not 110.
- The "Pending → Phase 2 — pipeline plumbing (no GPU)" section still lists 2.5, 2.6, 2.7 as `[ ]` even though their code, requirements (`pyyaml`), and tests are all in. The TODO needs a refresh to mark Phase 2 done and shift the next milestones (Phase 3.8/3.9/3.10) into focus.

### 4.2 [README.md](README.md) is forward-looking
The README documents the project as if it were complete. As of this snapshot:
- **Missing config files** that the README references: `configs/models.yaml`, `configs/datasets.yaml`, `configs/experiments/h{1..4}.yaml`. Only `configs/paths.template.yaml` exists. AGENT.md §4.3 mandates these.
- **Missing source trees**: `src/sampling/`, `src/metrics/`, `src/plotting/`, `scripts/`. The README's Quick Start (`python -m scripts.smoke_test`), Running Experiments (`bash scripts/run_h{N}.sh`), and `python -m src.plotting.generate_all` will all fail today.
- **Verifier audit log path**: README §Verification claims "Every verification decision is logged to `${outputs_root}/logs/verifier/`". The plumbing for this exists (`log_verifier_decision`), but no caller invokes it yet because the sampling/measurement loop has not been built. The claim is true *of the design*, not of a runnable system.
- This is acceptable for a paper-facing README that previews the final repo, but I would recommend either:
  (a) adding a "Status" badge / one-line note that the runnable surface is currently the verifier and pipeline-plumbing layer, or
  (b) trimming the README until the corresponding code lands.

### 4.3 Minor — `__init__.py` files
- `src/__init__.py` is empty (1-byte file). Fine for Python 3.10+ namespace, but consistency would prefer either a docstring or removal.

---

## 5. Test Coverage Summary

| File                                                          | Tests | Notes                                                                              |
| ------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------- |
| [tests/test_gsm8k.py](tests/test_gsm8k.py)                    | 37    | 14 positive, 11 negative, 4 priority, 5 extraction edge cases, 3 return-type.      |
| [tests/test_math.py](tests/test_math.py)                      | 40    | 14 positive, 11 negative, 10 extraction-helper, 3 robustness, 2 return-type.       |
| [tests/test_humaneval.py](tests/test_humaneval.py)            | 21    | 9 correctness, 6 error paths, 2 timeout, 2 sandbox, 2 return-type.                 |
| [tests/test_interface.py](tests/test_interface.py)            | 12    | Dispatch, lookup, error paths, registry surface, uniform return.                   |
| [tests/test_paths.py](tests/test_paths.py)                    | 14    | Interpolation, ~ expansion, missing/undefined/cyclic refs, frozen, ensure_dirs.    |
| [tests/test_cache.py](tests/test_cache.py)                    | 16    | Fingerprint, path layout, roundtrip, immutability, key/version mismatch, streaming. |
| [tests/test_logging_utils.py](tests/test_logging_utils.py)    | 13    | Run logger, verifier audit log, compute log, timestamp formats.                    |
| **Total**                                                     | **153** | All passing.                                                                       |

Coverage is appropriately weighted toward correctness-critical surfaces: the verifiers (98 tests, ~64 % of the suite) are the modules whose bugs would directly invalidate paper results, exactly as AGENT.md §1 prescribes ("a measurement bug here invalidates paper results, so correctness > convenience").

---

## 6. Recommendations (in priority order)

1. **Refresh [AGENT/TODO.md](AGENT/TODO.md)** to reflect that Phase 2.5/2.6/2.7 are shipped; update "110 tests passing" to "153 tests passing"; promote Phase 3.8/3.9/3.10 to the active block. (Documentation only — no code change.)
2. **Add a Status/Phase note to [README.md](README.md)** — one or two lines telling the reader which scripts and modules are runnable today. Today's README implies a finished system.
3. **Plan the Phase-3 cache-key encoding for H3 and H4 explicitly.** AGENT.md §3.2 lists the required cache-key fields, and §6.3 says the H3 procedure name and kwargs must be encoded — but the current `CacheKey` dataclass has no slot for "inference procedure" or "max reasoning length". Either the H3/H4 runs will need a separate cache-key class, or `CacheKey` will need to grow optional fields with a careful version bump. Worth deciding before Phase 3 starts so the cache layout doesn't churn after samples land.
4. **(Nice-to-have)** Tighten `_resolve_string` in [src/pipeline/paths.py](src/pipeline/paths.py): the outer fixed-point loop is redundant given recursive substitution.

---

## 7. Conclusion

The current code base meets every hard rule in AGENT.md that is in scope for the work shipped to date, with strong test coverage and disciplined adherence to the project's terminology and methodology commitments. There are no correctness concerns to raise. The remaining issues are documentation-only: [AGENT/TODO.md](AGENT/TODO.md) understates the progress, and [README.md](README.md) overstates it. Both are easy to fix and neither blocks Phase 3.
