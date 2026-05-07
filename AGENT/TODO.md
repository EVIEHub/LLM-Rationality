# TODO — Rational Gap Measurement Codebase

**Companion to:** [AGENT.md](AGENT.md), [HANDOFF.md](HANDOFF.md), [methodology/](methodology/)
**Last update:** 2026-05-07
**Status:** Phases 1 & 2 complete; Phase 3.9 (metrics + saturation helpers) complete; GPU-prep work complete. 12 of 14 build deliverables done. **209 tests passing.** Blocked on 3.8 (vLLM runner) and 3.10 (smoke test) — both require a GPU + vLLM and were deferred to a cloud-GPU session. Per the no-skipif-for-invariants rule, the determinism test in 3.8 will run live in the target environment, not gated behind skipif.

> ## 🚦 NEW SESSION ON A GPU BOX? START HERE
>
> If you are picking up this project on a fresh GPU box, **read [HANDOFF.md](HANDOFF.md) first**. It contains:
> - Environment setup steps (vLLM install, model download, paths.yaml, sanity tests).
> - Detailed specs for Phase 3.8 (vLLM runner) and 3.10 (smoke test).
> - The phase plan from H1 production through to manual audit and finalisation.
> - All open design questions with recommended decisions.
>
> Then read [methodology/](methodology/) — seven short rule files that this codebase obeys. They are not in AGENT.md but are load-bearing.
>
> This file ([TODO.md](TODO.md)) is the **progress log**; read after HANDOFF.md.

This document tracks build progress against the phased plan agreed in design
discussion. Hard rules are in [AGENT.md](AGENT.md). Methodology rules are in
[methodology/](methodology/). The GPU-session playbook is in [HANDOFF.md](HANDOFF.md).

---

## Done

### Phase 1.1 — GSM8K verifier *(2026-05-07)*

**Code shipped**
- [src/verification/gsm8k.py](src/verification/gsm8k.py) — pure function
  `verify(generation: str, ground_truth: str) -> float`. Returns
  `1.0`/`0.0` (binary $U$); `float` signature is forward-compatible with
  reward-model verifiers per AGENT.md §3.3.
- [tests/test_gsm8k.py](tests/test_gsm8k.py) — **37 tests passing**
  (14 positive, 11 negative, 4 priority-ordering, 5 extraction edge cases,
  3 return-type contract). Exceeds AGENT.md §6.2's ≥10+10 requirement.
- Scaffolding: [requirements.txt](requirements.txt) (pytest only for
  Phase 1), [pytest.ini](pytest.ini) (`pythonpath = .` so
  `from src.verification.gsm8k` resolves), [src/__init__.py](src/__init__.py),
  [src/verification/__init__.py](src/verification/__init__.py).

**Design choices recorded**
- Four-pattern priority chain matches AGENT.md §8.2:
  `####` → `\boxed{}` → "the answer is X" → last bare number.
- Rightmost match wins within each pattern — handles models that restate
  intermediate numbers before stating the final answer.
- Numeric normalisation strips commas, leading `$`, trailing `%`;
  comparison via `math.isclose(rel_tol=1e-9, abs_tol=1e-9)` so `72`
  matches `72.0`.
- Verifier is purely functional — **no logging inside**. The audit log
  required by AGENT.md §3.3 is written at the call site (will land in
  Phase 2.7).
- Fractions like `\boxed{1/2}` correctly fail GSM8K extraction and fall
  through to `0.0`; fraction support is MATH territory (Phase 1.2,
  SymPy-based).

### Phase 1.2 — MATH verifier *(2026-05-07)*

**Code shipped**
- [src/verification/math.py](src/verification/math.py) — pure function
  `verify(generation, ground_truth) -> float` using `math-verify` 0.9.0
  for symbolic equivalence.
- [tests/test_math.py](tests/test_math.py) — **40 tests passing**
  (14 positive, 11 negative, 10 extraction-helper, 3 timeout/exception
  robustness, 2 return-type contract).
- [requirements.txt](requirements.txt): added `math-verify>=0.9.0`.

**Design choices recorded**
- Extraction is done locally (balanced-brace walk) before delegating to
  `math-verify`. Without an explicit `\boxed{}` anchor, the library's
  parser silently extracts only the leading numeric token from bare
  LaTeX (e.g. `2\sqrt{3}` → `2`). Wrapping both sides in `\boxed{...}`
  before parsing makes extraction deterministic.
- Balanced-brace walker handles nested braces (`\boxed{\frac{a}{b}}`),
  LaTeX escape sequences (`\{`, `\}`, `\\`), multiple `\boxed{}`
  (rightmost-balanced wins), and falls back to earlier `\boxed{`
  openings if the rightmost is unbalanced.
- Both `math_verify.parse` and `math_verify.verify` are wrapped in
  `try/except` even though they default to `raise_on_error=False` —
  defence-in-depth for AGENT.md §3.3's "SymPy timeouts must default to
  false, never raise" rule. Confirmed working via two monkey-patched
  exception tests.
- LaTeX equivalence verified end-to-end for: fractions ↔ decimals
  (`\frac{1}{2}` ↔ `0.5` ↔ `1/2`), radical simplification (`\sqrt{12}`
  ↔ `2\sqrt{3}`), algebraic commutativity (`x^2+1` ↔ `1+x^2`), mixed
  numbers (`1.5` ↔ `\frac{3}{2}`), constants (`\pi`), and intervals.

### Phase 1.3 — HumanEval verifier *(2026-05-07)*

**Code shipped**
- [src/verification/humaneval.py](src/verification/humaneval.py) — pure
  function `verify(generation, ground_truth) -> float`. Executes
  `generation + "\n" + ground_truth` in a subprocess (AGENT.md §3.3).
- [tests/test_humaneval.py](tests/test_humaneval.py) — **21 tests
  passing** (9 correctness, 6 error paths, 2 timeout, 2 sandbox
  isolation, 2 return-type contract).

**Design choices recorded**
- Subprocess hardening: `subprocess.run` with `timeout=5.0`,
  `capture_output=True`, `cwd=tempfile.TemporaryDirectory()`, `env`
  containing only `PATH` (no `HOME`, `PYTHONPATH`, or HF tokens),
  `start_new_session=True`. Documented as "not a full sandbox" — threat
  model is buggy/runaway code, not adversarial.
- Bug found and fixed during 1.4 integration: empty `ground_truth`
  caused a vacuous pass (subprocess exits 0 with no assertions run).
  Now guarded with `if not ground_truth.strip(): return 0.0`. Failing
  closed matches AGENT.md §9 ("prefer the more conservative option").
  Regression test `test_empty_ground_truth_returns_zero` covers this.
- The `ground_truth` is the check-program suffix (`def check(candidate)`
  + assertions + `check(<entry_point>)`). The dataset loader is
  responsible for assembling it; the verifier just runs it.
- Timeout test patches `_TIMEOUT_SECONDS` down to 0.5s to keep the test
  suite fast (otherwise infinite-loop test would add 5s per run).

### Phase 1.4 — Unified verifier registry *(2026-05-07)*

**Code shipped**
- [src/verification/interface.py](src/verification/interface.py) —
  `verify(dataset, generation, ground_truth) -> float` dispatcher,
  plus `get_verifier(dataset)` and `known_datasets()`. Registry maps
  `gsm8k`, `math`, `humaneval` to their per-module `verify` functions.
- [src/verification/__init__.py](src/verification/__init__.py) —
  re-exports the dispatcher API so callers can write
  `from src.verification import verify`.
- [tests/test_interface.py](tests/test_interface.py) — **12 tests
  passing** (3 dispatch correctness, 2 lookup ergonomics, 2 error
  paths, 2 registry surface, 1 dispatcher identity, 1 uniform return
  contract, 1 top-level re-export).

**Design choices recorded**
- Lookup is case-insensitive and whitespace-trimming (`"GSM8K"` and
  `" gsm8k "` both resolve). Unknown dataset → `KeyError` listing the
  registered names so the failure is self-diagnosing.
- Registry uses function references, not module objects, so dispatch
  goes straight to `verify` without an extra attribute lookup per
  call.
- The MATH module is imported as `math as math_verifier` in
  `interface.py` to avoid any future ambiguity between our submodule
  and stdlib `math` (the package path `src.verification.math` makes
  the actual import unambiguous, but the alias keeps reader cognitive
  load low).
- Uniform return contract verified by a parametric test: every
  registered verifier returns `float == 0.0` on `("", "")`. This
  caught the HumanEval vacuous-pass bug above.

### Phase 2.5 — Path configuration *(2026-05-07)*

**Code shipped**
- [src/pipeline/paths.py](src/pipeline/paths.py) — `Paths` frozen
  dataclass + `load_paths(path=None)`. Resolves `${name}`
  interpolation iteratively to a fixed point and expands `~`.
  `Paths.ensure_dirs()` creates all five dirs idempotently.
- [configs/paths.template.yaml](configs/paths.template.yaml) — user
  template; gitignored target is `configs/paths.yaml`.
- [.gitignore](.gitignore) — excludes `configs/paths.yaml`,
  `__pycache__`, `.pytest_cache`, virtual envs.
- [tests/test_paths.py](tests/test_paths.py) — **13 tests passing**.
- `requirements.txt`: added `pyyaml>=6.0`.

**Design choices recorded**
- `${name}` interpolation iterates to fixed point so chains
  (`samples_dir` → `outputs_root`) and even longer chains resolve.
  Cyclic references raise `ValueError` with the offending chain.
- Tilde expansion via `Path.expanduser()` — `~/rational_gap_outputs`
  works without manual `$HOME` substitution.
- `Paths` is frozen; tests verify mutation raises and that
  `ensure_dirs()` is idempotent.
- The `FileNotFoundError` message tells the user exactly how to fix
  it (copy template → paths.yaml).

### Phase 2.6 — Sample cache *(2026-05-07)*

**Code shipped**
- [src/pipeline/cache.py](src/pipeline/cache.py) — `CacheKey` frozen
  dataclass + `cache_path()`, `cache_exists()`, `write_cache()`,
  `read_cache()`. File format is gzipped JSONL with a header line
  recording the full cache key + format version.
- [tests/test_cache.py](tests/test_cache.py) — **18 tests passing**.

**Design choices recorded**
- `CacheKey` includes all nine fields from AGENT.md §3.2:
  `model, dataset, K, temperature, top_p, top_k, max_tokens, seed,
  prompt_template_version`. Per-field perturbation tests confirm each
  changes the fingerprint.
- `__post_init__` canonicalises numeric types (`int(K)`,
  `float(temperature)`, …) so `temperature=1` and `temperature=1.0`
  produce identical fingerprints.
- Filename includes readable hints (`v1_gsm8k_Qwen2.5-7B_K4_<hash>`)
  for `ls`-friendly debugging; the 16-hex-char SHA-256 prefix
  guarantees uniqueness.
- Atomic write via `.tmp` sibling + `Path.replace()`; a crashed
  write removes the `.tmp` and leaves no half-file (verified by a
  fault-injection test).
- `read_cache()` validates the embedded header against the requested
  key — defends against the (vanishingly unlikely) hash collision
  and against manually-renamed files.
- Header records `_format_version`. Bumping `CACHE_FORMAT_VERSION`
  changes the filename prefix, so old and new formats coexist
  without silent invalidation per AGENT.md §3.2.

### Phase 2.7 — Logging utilities *(2026-05-07)*

**Code shipped**
- [src/pipeline/logging_utils.py](src/pipeline/logging_utils.py) —
  `setup_run_logger()`, `log_verifier_decision()`, `log_compute()`.
- [tests/test_logging_utils.py](tests/test_logging_utils.py) —
  **12 tests passing**.

**Design choices recorded**
- `setup_run_logger()` configures the **root** logger, so any
  module-level `logging.getLogger(__name__)` flows into both the
  per-run file (DEBUG) and stderr (configurable, default INFO).
  AGENT.md §3.6 mandates DEBUG to file.
- Repeat calls close prior handlers and replace them — no FD leaks,
  no duplicate log lines.
- `log_verifier_decision()` writes one JSONL line per verifier call
  to `${logs_dir}/verifier/{dataset}_log.jsonl`. Schema is
  caller-defined; the function just appends.
- `log_compute()` writes to `${logs_dir}/compute_budget.jsonl` with
  `timestamp` (ISO-8601 UTC), `experiment`, `gpu_hours`, optional
  `metadata`.

### Phase 3.9 — Rational gap metrics *(2026-05-07)*

**Code shipped**
- [src/metrics/rational_gap.py](src/metrics/rational_gap.py) —
  `compute_rational_gap(utilities)` returning a
  `RationalGapEstimate` (aggregate scalars + per-prompt arrays);
  `bootstrap_ci_over_prompts(per_prompt_values)` returning a
  `BootstrapCI` (mean + percentile interval).
- [tests/test_rational_gap.py](tests/test_rational_gap.py) —
  **22 tests passing**, including closed-form invariants.
- `requirements.txt`: added `numpy>=1.24`.

**Design choices recorded**
- Generic over $U$: accepts any float-valued `(M, K)` array, even
  though current verifiers all return `{0.0, 1.0}`. A
  `test_continuous_utility_works` test pins this contract.
- Closed-form invariants verified as tests:
  $\hat{\mathcal{R}}_1 \equiv 0$, all-zeros → $0$, all-ones → $0$,
  "exactly $1$ of $K$ passes per prompt" → $1 - 1/K$.
- `RationalGapEstimate` exposes per-prompt arrays so callers can
  bootstrap whichever statistic they need ($U^\circ_K$, $\bar{U}_K$,
  $\hat{\mathcal{R}}_K$). The metric module does not bootstrap
  internally — the caller chooses, per the methodology memo.
- `bootstrap_ci_over_prompts()` defaults to $B = 1000$ and 95% CI
  per the methodology memo. RNG is `np.random.default_rng(seed)`
  with vectorised resampling (`(B, M)` index matrix) for speed.
- `BootstrapCI.mean` is the mean of the bootstrap distribution,
  used as the per-seed point estimate; cross-seed mean ± std happens
  one level up.

### GPU-prep — Configs and shared test fixtures *(2026-05-07)*

While 3.8/3.10 wait for GPU access, four pieces of scaffolding that
feed into them landed:

**Code shipped**
- [configs/models.yaml](configs/models.yaml) — registry of 9 model
  aliases: full Tülu-3 trajectory (`tulu3-8b-base/sft/dpo/rlvr` for
  H2), H1 cross-model panel (`tulu3-8b-rlvr`, `qwen2.5-7b-instruct`,
  `llama3.1-8b-instruct`), and three Qwen-2.5 small variants
  (0.5B/1.5B/3B) for fast smoke testing.
- [configs/datasets.yaml](configs/datasets.yaml) — registry for
  `gsm8k`, `math`, `humaneval` with chat- and few-shot templates,
  verifier registry references, prompt/ground-truth field names, and
  `prompt_template_version`.
- [configs/experiments/h1.yaml](configs/experiments/h1.yaml) — H1
  concrete config: 3 models × 3 datasets × 3 seeds at K_max=64 with
  the saturation curve grid `{1, 2, 4, 8, 16, 32, 64}`. Sampling
  defaults pinned to AGENT.md §3.1; bootstrap pinned to methodology
  memo (B=1000, 95%).
- [tests/conftest.py](tests/conftest.py) — shared fixtures:
  `paths_factory`, `cache_key_factory`, `humaneval_check_factory`,
  `binary_utility_array` (the load-bearing `[1,0,0,0]`-style row
  pinned for downstream test reuse).
- [tests/test_configs.py](tests/test_configs.py) — **26 tests
  passing**: structural validation + cross-file reference checks
  (every H1 model alias must exist in models.yaml; every dataset's
  `verifier` must resolve via `verification/interface.py`).
- Refactored [tests/test_cache.py](tests/test_cache.py) and
  [tests/test_humaneval.py](tests/test_humaneval.py) to consume the
  conftest fixtures, removing per-file helper duplication.

**Design choices recorded**
- `tulu3-8b-base` is `meta-llama/Llama-3.1-8B` (the actual pretrained
  base on which Tülu-3 is built) with `prompt_mode: few_shot`;
  H2 must use few-shot prompting to elicit responses comparable to
  the SFT/DPO/RLVR stages, since base has no chat template.
- H1 sampling is done **once per (model, dataset, seed) at K_max=64**;
  the saturation curve at $K' \le 64$ is computed by direct truncation
  to the first $K'$ columns of the $(M, K_\text{max})$ utility matrix
  via `U_circ_at_K`, `U_bar_at_K`, `R_hat_at_K` in
  [src/metrics/rational_gap.py](src/metrics/rational_gap.py). These
  are one-line transcriptions of the paper formulas — **no** binomial
  closed forms, **no** "expectation" framing. With $M \approx 1000+$
  prompts the per-prompt-variance reduction from a closed form would
  be negligible compared to between-prompt variance (which the
  bootstrap CI captures), and the paper does not adopt expectation
  framing for $U^\circ_K$.
- Few-shot examples files (`configs/few_shot/{gsm8k_8shot,
  math_4shot, humaneval_3shot}.txt`) are referenced in
  `configs/datasets.yaml` but not yet created — they are H2-only
  artefacts and will be added when H2 lands.
- `test_h1_all_models_use_chat_mode` enforces that H1 stays
  instruct-only (mixing a base model would conflate "no alignment"
  with "gap exists", which is H2 territory).
- The MATH HF dataset id (`EleutherAI/hendrycks_math`) was chosen
  over the legacy `hendrycks/competition_math` since the EleutherAI
  fork is more reliably maintained on the Hub. To be confirmed at
  data-load time during 3.10.

### AGENT.md edits *(2026-05-07)*

- §3.3 verifier signature: `-> int` → `-> float` with extensibility note.
- §4.2 `U` row: return type aligned to `float`.
- §4.2 paper-symbol mappings: `R_hat`/`U_max_K`/`U_mean_K` →
  `R_hat_K`/`U_circ_K`/`U_bar_K` to match paper LaTeX directly.
- §4.2 forbidden-alternatives bullet (added by author) locks
  `pass_at_k`, `pass@k`, `accuracy`, etc. out of the codebase.

---

## Pending — GPU-session work

> Detailed specs and acceptance criteria for each item below live in
> [HANDOFF.md](HANDOFF.md). This list is just the queue; HANDOFF.md is
> the playbook.

### Immediate (unblock by running 3.8 + 3.10)
- [ ] **3.8** [src/sampling/vllm_runner.py](src/sampling/vllm_runner.py)
  — `VllmRunner` class wrapping `vllm.LLM`. `n=K` not a Python loop;
  explicit `seed=`; defaults from AGENT.md §3.1; clean VRAM release.
  Determinism test runs **live**, never `skipif`. See HANDOFF §3.
- [ ] **3.10** [scripts/smoke_test.py](scripts/smoke_test.py) — end-to-end
  on Qwen-2.5-1.5B-Instruct, 10 GSM8K prompts, K=4. Loads paths/configs,
  formats prompts, samples, caches, verifies, logs, computes
  $\hat{\mathcal{R}}_K$ + bootstrap CI, prints. See HANDOFF §4.

### H1 production (after 3.10 passes)
- [ ] **H1 runner** `scripts/run_h1.py` — 3 models × 3 datasets ×
  3 seeds at $K_\text{max}{=}64$. Saturation curve via
  `U_circ_at_K` / `U_bar_at_K` / `R_hat_at_K` direct truncation
  (no binomial, no expectation framing). See HANDOFF §5 and
  [methodology/no_expectation_framing.md](methodology/no_expectation_framing.md).
- [ ] **H1 figure** `src/plotting/h1.py` — saturation curve PDF
  with bootstrap CI bands.

### H2 (Tülu trajectory)
- [ ] **Author few-shot files** under `configs/few_shot/`:
  `gsm8k_8shot.txt`, `math_4shot.txt`, `humaneval_3shot.txt`. Bump
  `prompt_template_version` to `v2` in `configs/datasets.yaml`.
- [ ] **`configs/experiments/h2.yaml`** following the shape of
  `h1.yaml`. 4 models (base + SFT + DPO + RLVR) × 2 datasets
  (GSM8K, MATH) × 3 seeds.
- [ ] **H2 runner + figure** — plot $U^\circ_K$ and $\bar{U}_K$
  separately along the trajectory. See HANDOFF §6.

### H3 (inference mechanisms)
- [ ] **Extend `CacheKey`** with `inference_procedure: str = "direct"`
  and `inference_kwargs: tuple[tuple[str, str], ...] = ()`; bump
  `CACHE_FORMAT_VERSION` 1 → 2. See HANDOFF §7 for the rationale.
- [ ] **`src/sampling/inference_procedures.py`** — registry of
  `direct(τ)`, `cot`, `self_consistency`, `mcts`. Oracle BoN is **not**
  in the registry — it is the reachable upper bound, drawn as a
  reference line (see [methodology/hypotheses.md](methodology/hypotheses.md) §H3).
- [ ] **`configs/experiments/h3.yaml`**, runner, figure.

### H4 (reasoning length)
- [ ] **Extend `CacheKey`** with `max_reasoning_length: int | None = None`.
- [ ] **`budget_forced(...)`** in `inference_procedures.py` —
  two-stage budget forcing à la s1.
- [ ] **`configs/experiments/h4.yaml`**, runner, figure (7 length values).

### Manual audit (per [methodology/audit_and_uncertainty.md](methodology/audit_and_uncertainty.md))
- [ ] `tests/manual_audit_gsm8k.md` — 50 random + 50 load-bearing.
- [ ] `tests/manual_audit_math.md` — 50 random + 50 load-bearing.
- [ ] `tests/manual_audit_humaneval.md` — 50 random + 50 load-bearing.

### Finalisation
- [ ] **Trim [README.md](README.md)** to reflect what is actually
  runnable when the project ships (it currently previews the final
  state but several scripts don't exist yet).
- [ ] **`compute_budget.jsonl` summary** in TODO.md or README — total
  GPU-hours sanity-checked against the README's ~85 GPU-hour estimate.
- [ ] **Final commit** anchoring the published state.

---

## Methodology commitments to honour throughout

Reminders only — full versions are in [methodology/](methodology/), one
file per rule. Read those, not this short list, before making decisions.

- **Terminology** ([methodology/terminology.md](methodology/terminology.md)): code, log fields, and prose use `U_circ_K`, `U_bar_K`, `R_hat_K`. Never `pass_at_k` / `pass@K`.
- **K is a first-class axis** ([methodology/hypotheses.md](methodology/hypotheses.md)) — H1's headline figure is the saturation curve $\hat{\mathcal{R}}_K$ vs $K$, not a bar at a single $K$.
- **No expectation framing for $U^\circ_K$** ([methodology/no_expectation_framing.md](methodology/no_expectation_framing.md)) — saturation curves are direct truncation. No binomial closed forms.
- **H2 tests the strong claim** ([methodology/hypotheses.md](methodology/hypotheses.md)) — alignment does not *eliminate* the gap. Log $U^\circ_K$ and $\bar{U}_K$ separately.
- **H3 — oracle BoN is the upper bound** — not a candidate inference procedure. The reported H3 metric is the residual gap $U^\circ_K - u_I$ for each deployable procedure $I$.
- **Verifier audit is stratified** ([methodology/audit_and_uncertainty.md](methodology/audit_and_uncertainty.md)) — 50 random + 50 load-bearing per dataset.
- **Uncertainty quantification** — bootstrap CI over prompts (B = 1000, 95%) reported on top of the 3-seed mean ± std.
- **No skipif on load-bearing invariants** ([methodology/no_skipif_for_invariants.md](methodology/no_skipif_for_invariants.md)) — the vLLM determinism test runs live on the GPU box, not skipped.

---

## Open questions / deferred decisions

These are flagged so the GPU session doesn't burn cycles re-discovering
them. Make a decision, document it here, and move on. See HANDOFF §12
for fuller treatment.

1. **MATH HF dataset id** — currently `EleutherAI/hendrycks_math` in `configs/datasets.yaml`. Confirm at first load on the GPU box; if HF moved it, switch to the canonical id and bump `prompt_template_version` only if the field names change too.
2. **vLLM version pin** — once you find a vLLM version that gives byte-identical samples for fixed seed, pin it exactly in `requirements.txt`. Record the pin here.
3. **Few-shot file content** (H2 prerequisite) — use the published Wei et al. 8-shot CoT set for GSM8K; pick a published source for MATH and HumanEval and cite in the file headers.
4. **`CacheKey` extension for H3 / H4** — bake when H3 starts. Plan in HANDOFF §7. Default values keep existing caches valid, but `CACHE_FORMAT_VERSION` bumps from 1 → 2.
5. **H1 panel completeness vs. GPU size** — if the box can't run Llama-3.1-8B, drop it from the panel and document the deviation. Tülu-3-RLVR + Qwen-2.5-7B is still informative.

---

## Resolved (kept here for the audit trail)

- **GPU environment for Phase 3.8 and 3.10 (resolved 2026-05-07):** the prior session's environment was CPU-only. Decision: defer 3.8 + 3.10 to a cloud GPU box; the determinism test runs live, never behind `skipif`. Documentation handed off in [HANDOFF.md](HANDOFF.md) and [methodology/](methodology/).
