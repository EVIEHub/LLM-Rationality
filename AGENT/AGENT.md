# AGENT.md — Coding Agent Instructions

This document instructs coding agents (Claude Code, Cursor, Aider, etc.) on how to work in this repository. Read this entirely before making any code changes.

> **Companion docs (read in order on first contact):**
> 1. This file (`AGENT/AGENT.md`) — hard rules.
> 2. [`AGENT/TODO.md`](TODO.md) — current build status and pending queue.
> 3. [`AGENT/HANDOFF.md`](HANDOFF.md) — session brief / playbook for new agents (especially anyone picking up the work on a new machine, e.g. a cloud GPU box).
> 4. [`AGENT/methodology/`](methodology/) — seven short rule files of methodological commitments (terminology, hypothesis framings, audit and uncertainty protocols, no-skipif, no-expectation-framing). These are not in this file but are load-bearing.

## 1. Project Identity

This repository implements measurement experiments for an academic paper on LLM rational gap. **Code quality and methodological correctness take priority over feature speed.** A measurement bug here invalidates paper results, so correctness > convenience.

## 2. What This Code Does

The pipeline performs four operations in sequence:

1. **Sample** $K$ trajectories per prompt from a fixed model $\pi_\theta$ via vLLM
2. **Verify** each trajectory against ground truth, producing $U(x, y) \in \{0, 1\}$
3. **Aggregate** into rational gap $\widehat{\mathcal{R}}_K = \frac{1}{N}\sum_i \max_k U(x_i, y_{i,k}) - \frac{1}{NK}\sum_{i,k} U(x_i, y_{i,k})$
4. **Plot** results into figures and tables

All operations are deterministic given a fixed seed.

## 3. Hard Rules (Never Violate)

### 3.1 Sampling
- **Never modify $\pi_\theta$ during measurement.** The experiments measure properties of frozen models. Quantization, LoRA adapters, or in-place weight modification break the framework.
- **Default sampling parameters**: `temperature=1.0`, `top_p=1.0`, `top_k=-1`, **no truncation**. Truncation at sample time changes the support and corrupts $U^\circ_K$. Only override these in H3 (where varying inference is the point) and clearly document.
- **Always seed explicitly.** Every `runner.sample()` call must pass `seed=...`. Never rely on default randomness.
- **vLLM's `n=K` parameter** is the correct way to sample $K$ trajectories per prompt. Never call `generate()` in a loop $K$ times — it is 5–10× slower and breaks reproducibility.

### 3.2 Caching
- **Cache key must include all parameters that affect output**: `(model, dataset, K, temperature, top_p, top_k, max_tokens, seed, prompt_template_version)`. Missing any of these creates silent bugs.
- **Never modify a cached file in place.** Treat `data/samples/*.jsonl.gz` as immutable. Recompute metrics; do not patch samples.
- **Cache hit must be byte-deterministic.** If you change cache key format, bump a version field, do not silently invalidate old caches.

### 3.3 Verification
- **Verifiers are pure functions**: `verify(generation: str, ground_truth: str) -> float`. No side effects, no model calls, no global state. Current verifiers return values in `{0.0, 1.0}` (binary $U$); the `float` signature preserves extensibility to reward-model verifiers without an API break, and matches the `metrics` module which is general over $U$.
- **Every verifier decision is logged** to `${logs_dir}/verifier/{dataset}_log.jsonl`. Do not skip logging "for performance" — the audit trail is necessary for paper review.
- **Code execution (HumanEval) must be sandboxed**: subprocess + 5-second timeout + restricted environment. Never `exec()` model output in the parent process.
- **SymPy timeouts** (used in MATH verifier) must default to false (incorrect), not raise an exception. Crashing on a single hard prompt should not abort 1000-prompt runs.

### 3.4 Paths
- **Never hardcode paths.** Read everything from `src/pipeline/paths.py`, which loads `configs/paths.yaml`.
- **All outputs go outside the repo** (configured by `outputs_root`). Writing inside the repo is a bug.
- **Use `pathlib.Path`**, not string concatenation, for cross-platform safety.

### 3.5 Reproducibility
- **No randomness without explicit seeding.** This includes `numpy`, `random`, `torch`, vLLM. `set_seed()` at the top of every entry point.
- **Variance is reported across 3 seeds** by default. Do not silently default to 1 seed; if compute-constrained, document explicitly.
- **No notebook-style state.** All scripts must be runnable end-to-end from `python -m scripts.<name>`. Notebooks are for exploration only and never imported by other code.

### 3.6 Logging
- **Every entry-point script writes to `logs/runs/<timestamp>_<experiment>.log`** via `setup_run_logger()`. No silent runs.
- **Log compute usage** via `log_compute(...)` at the end of each run. The cumulative budget in `logs/compute_budget.jsonl` is reported in the paper.
- **Log levels**: INFO for milestones, DEBUG for per-prompt details (off by default), WARNING for recoverable anomalies, ERROR for failures.

## 4. Code Style

### 4.1 Python
- Python 3.10+. Use type hints for all public functions.
- Format with `black` (line length 100). Lint with `ruff`.
- Imports: stdlib → third-party → local, alphabetical within groups.
- Docstrings: Google style, on every public function and class.

### 4.2 Naming Conventions
- **Symbols match the paper**:
  - `K` = number of samples per prompt (per-prompt budget)
  - `M` = number of prompts in dataset
  - `L` = max reasoning length (for H4)
  - `U` = utility function, returning `float`; current binary verifiers return values in {0.0, 1.0}
  - `R_hat_K` = $\hat{\mathcal{R}}_K$
  - `U_circ_K` = $U^\circ_K$
  - `U_bar_K` = $\bar{U}_K$
  - `pi_theta` = the policy (in code: `model` or `runner`)
- **Files match modules**: `gsm8k.py` for GSM8K verifier, `tulu3.py` for Tülu-3 model loading, etc.
- **Avoid abbreviations** that are not in the paper: `acc`, `gen`, `traj` are fine; `cfg`, `mgr` are not.
- **Forbidden alternatives**: `pass_at_k`, `pass@k`, `pass1`, `pass_rate`, `accuracy`, `success_rate`. These either belong to the LLM evaluation literature or denote different quantities; mixing them with paper symbols creates silent inconsistencies when paper formulas are revised.

### 4.3 Configuration
- All experiment hyperparameters live in `configs/experiments/h{N}.yaml`. Do not hardcode in scripts.
- All model paths live in `configs/models.yaml` with HuggingFace IDs and short aliases.
- All dataset paths and prompt templates live in `configs/datasets.yaml`.
- Loading config: use `pyyaml` directly; do not introduce a config framework (Hydra/OmegaConf) for this scale of project.

## 5. Forbidden Practices

The following will be rejected on review:

- **Modifying paper math**: $\hat{\mathcal{R}}_K$ formula, definitions of $U^\circ_K$, $\bar{U}_K$. These are paper artifacts; if they appear wrong, raise an issue, do not silently change.
- **Reading from external networks at runtime** (e.g., `huggingface_hub` downloads inside the measurement loop). All datasets and model weights must be pre-downloaded to `${raw_data_dir}` and the local model cache.
- **Catching exceptions silently** with bare `except:`. Always catch specific types and log.
- **Commenting out tests** to make CI pass. Either fix the test or document why skipped.
- **Writing to the repository at runtime** (e.g., creating files in `src/`). All runtime writes go to the configured outputs directory.
- **Adding heavy dependencies** without discussion. The current dependency footprint is intentionally minimal.

## 6. Workflow

### 6.1 Adding a New Experiment Variant
1. Add config to `configs/experiments/h{N}.yaml`
2. Reuse existing `sampling/` and `verification/` modules; do not duplicate logic
3. Add a script `scripts/run_h{N}_<variant>.sh`
4. Add unit tests for any new pure function
5. Document in README under "Running Experiments"

### 6.2 Adding a New Verifier
1. Implement as pure function in `src/verification/<dataset>.py`
2. Add ≥20 unit test cases in `tests/test_verifier_<dataset>.py` (10 positive + 10 negative)
3. Register in `src/verification/interface.py`
4. Hand-verify 50 random samples from a real model run; commit to a `tests/manual_audit_<dataset>.md`

### 6.3 Adding a New Inference Procedure (H3)
1. Implement in `src/sampling/inference_procedures.py` as a function `procedure(runner, prompt, K, **kwargs) -> List[str]`
2. Add to `configs/experiments/h3.yaml` with all hyperparameters
3. Cache key must include the procedure name and all kwargs
4. Update H3 figure generator to include the new bar

### 6.4 Modifying the Cache Format
1. Bump cache version in `src/pipeline/cache.py`
2. Old caches become invalid (this is expected); add migration script if data is too expensive to regenerate
3. Document the change in CHANGELOG.md

## 7. Testing

- Run unit tests: `pytest tests/`
- Smoke test the pipeline: `python -m scripts.smoke_test --num-prompts 10 --K 2`
- A change is not complete until smoke test passes.

## 8. Common Pitfalls

These have already burned us; do not repeat:

1. **Greedy decoding ignores `seed`** — vLLM treats `temperature=0` as deterministic, but multiple `n` will all be identical. Greedy with `K>1` is wasteful; set `K=1` for greedy.
2. **GSM8K answer extraction has multiple formats** — `####`, `\boxed{}`, "the answer is X", standalone last number. The verifier tries them in order. Do not assume a single format.
3. **MATH ground truth contains LaTeX** — `\frac{1}{2}`, `\sqrt{12}`, etc. SymPy equivalence is required; string match fails on equivalent forms.
4. **vLLM `n=K` and `seed` interaction** — at temperature > 0, the K samples differ; at temperature = 0, they are identical. Explicitly check this in tests.
5. **Cache file collisions** — the cache key must encode all parameters. A cache hit on the wrong configuration silently corrupts results.
6. **GPU memory after model swap** — vLLM does not always release VRAM cleanly. After loading a new model, run `del runner; torch.cuda.empty_cache(); gc.collect()` before instantiating the next.

## 9. When in Doubt

Prefer the more conservative, more reproducible, more loggable option, even if slower. This is research code that other researchers will read and try to reproduce. Speed is not the bottleneck; correctness is.

If the right behavior is unclear, **leave a `# TODO(human): ...` comment** explaining the question and stop. Do not guess on methodology.

## 10. Out of Scope

This codebase deliberately does not implement:
- Training or fine-tuning (we measure frozen models only)
- Multi-node distributed inference (single-node only)
- Reward models (binary verifiable utility only)
- Continuous utility (binary $U \in \{0, 1\}$ only)

Requests to add these belong in a separate paper / repository.