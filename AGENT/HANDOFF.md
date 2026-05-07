# HANDOFF.md — Cloud-GPU Session Brief

**Audience:** the next coding agent (Claude or otherwise) who picks up this project on a fresh cloud GPU box. The previous session's per-project memory was machine-local and is **not** available here. Everything load-bearing has been moved into this repository.

**Author of this doc:** the prior CPU-only session (2026-05-07).

---

## 0 — Read these in order before any code change

1. [AGENT.md](AGENT.md) — hard rules (10 sections). Read in full.
2. [TODO.md](TODO.md) — current build status and the queue of remaining work.
3. **This file** ([HANDOFF.md](HANDOFF.md)) — environment setup + step-by-step playbook.
4. [methodology/](methodology/) — seven short files of methodological commitments carried across sessions. The directory has its own [README](methodology/README.md) describing read order.

The whole reading set is ≈ 30 minutes. Do not skip it: there are decisions in `methodology/` (e.g. terminology, no-expectation-framing, no-skipif-for-invariants) that have already been corrected once and would be a regression to re-introduce.

---

## 1 — State at handoff (2026-05-07)

| Phase | Status | Tests |
|---|---|---|
| 1.1 GSM8K verifier | ✅ done | 37 |
| 1.2 MATH verifier (math-verify) | ✅ done | 40 |
| 1.3 HumanEval verifier (sandbox) | ✅ done | 21 |
| 1.4 Verifier registry | ✅ done | 12 |
| 2.5 Paths config | ✅ done | 13 |
| 2.6 Sample cache | ✅ done | 18 |
| 2.7 Logging utilities | ✅ done | 12 |
| 3.9 Rational-gap metrics + saturation helpers | ✅ done | 30 |
| GPU-prep: configs/models.yaml | ✅ done | covered by test_configs |
| GPU-prep: configs/datasets.yaml | ✅ done | covered by test_configs |
| GPU-prep: configs/experiments/h1.yaml | ✅ done | covered by test_configs |
| GPU-prep: tests/conftest.py shared fixtures | ✅ done | — |
| **3.8 vLLM runner** | ⏸ blocked here | needs GPU |
| **3.10 smoke test** | ⏸ blocked here | needs GPU |
| H1 production run | ⏸ post-3.10 | — |
| H2 (Tülu trajectory) | ⏸ needs few-shot files + CacheKey re-check | — |
| H3 (inference mechanisms) | ⏸ needs CacheKey extension | — |
| H4 (reasoning length) | ⏸ needs CacheKey extension | — |
| Figures and tables | ⏸ post-experiments | — |
| Manual audit | ⏸ post-experiments | — |
| Finalise (README polish, compute total) | ⏸ post-everything | — |

**Test count at handoff: 209.** First action on the GPU box is to clone, install, and confirm `pytest tests/` reports 209 passed. Anything other than 209 is a regression to investigate before going further.

---

## 2 — Environment setup on the GPU box

```bash
# 1. Clone
git clone <repo-url> rational-gap-of-LLM-reasoning
cd rational-gap-of-LLM-reasoning

# 2. Python venv + minimal deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # pytest, math-verify, pyyaml, numpy

# 3. Install vLLM + PyTorch matching the box's CUDA stack.
#    Do NOT pin a vllm version blindly — check the CUDA version on the box
#    first (`nvidia-smi`), then follow https://docs.vllm.ai for the
#    correct install command. Once you find a working version, ADD it to
#    requirements.txt with an exact pin (e.g. vllm==0.6.3).
pip install vllm
# (PyTorch is pulled in transitively by vllm; verify CUDA-enabled below.)

# 4. Verify CUDA
python -c "import torch; print('cuda?', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
# Expected: True / >=1

# 5. Configure local paths
cp configs/paths.template.yaml configs/paths.yaml
# Edit configs/paths.yaml — set outputs_root to a writable dir OUTSIDE
# the repo with ≥ 20 GB free (e.g. /workspace/rg_outputs on RunPod).

# 6. Pre-download model weights (NOT inside the measurement loop, per
#    AGENT.md §5 "no network reads during sampling").
huggingface-cli login        # if needed for gated models (Llama)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
# … and the rest of the models you'll run; the full list is in
# configs/models.yaml. For the smoke test, only qwen2.5-1.5b-instruct
# is required.

# 7. Smoke-test the existing test suite
pytest tests/
# Expected: 209 passed in ~5s. If anything other than 209, STOP and
# investigate — every previous session's work depended on this baseline.
```

If the box has multiple GPUs, default vLLM tensor-parallel = 1 unless you have a specific reason to shard (the H1 panel at 8B fits comfortably on one A100/A800).

---

## 3 — Phase 3.8 — vLLM runner

**Module path:** `src/sampling/vllm_runner.py` (does not exist yet)

### 3.8.1 Interface (suggested — refine as you build)

```python
# src/sampling/vllm_runner.py

from __future__ import annotations
from dataclasses import dataclass

# Lazy-import vllm/torch so unit tests that don't need a live LLM
# (e.g. parameter-packing tests with a mock) can import this module
# without CUDA being present. Real instantiation imports them.

@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters per AGENT.md §3.1.

    Defaults are the H1/H2 baseline. H3 explicitly varies these.
    """
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 2048


class VllmRunner:
    """Thin wrapper around vllm.LLM enforcing the project's sampling rules.

    - `n=K` is used for K samples per prompt; never a Python loop
      (AGENT.md §3.1, §8.4).
    - `seed` is mandatory on every call (no default randomness).
    - VRAM is released cleanly on `close()` (AGENT.md §8.6).
    """

    def __init__(self, hf_id: str, **vllm_kwargs):
        from vllm import LLM
        self.hf_id = hf_id
        self.llm = LLM(model=hf_id, **vllm_kwargs)

    def sample(
        self,
        prompts: list[str],
        K: int,
        seed: int,
        config: SamplingConfig | None = None,
    ) -> list[list[str]]:
        """Sample K completions per prompt. Returns shape (M, K) of strings."""
        from vllm import SamplingParams
        cfg = config or SamplingConfig()
        params = SamplingParams(
            n=K,
            seed=seed,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        # Map back to (M, K). vLLM returns one RequestOutput per prompt;
        # each has .outputs of length K.
        return [[o.text for o in req.outputs] for req in outputs]

    def close(self) -> None:
        """Release VRAM; per AGENT.md §8.6 this is required between models."""
        import gc
        import torch
        del self.llm
        torch.cuda.empty_cache()
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

### 3.8.2 Tests (`tests/test_vllm_runner.py`)

The determinism test is **load-bearing** and per [methodology/no_skipif_for_invariants.md](methodology/no_skipif_for_invariants.md) runs live on this GPU box. **Never** wrap it in `@pytest.mark.skipif`.

| Test | Purpose | Needs GPU? |
|---|---|---|
| `test_sample_returns_M_by_K_strings` | shape contract | yes (or mock) |
| `test_seed_byte_determinism` | same model + prompts + seed → byte-identical samples across two runs | **yes, live** |
| `test_default_sampling_matches_AGENT_3_1` | when `config=None`, params are temperature=1.0, top_p=1.0, top_k=-1, n=K, seed=<passed> (mockable) | mock |
| `test_n_equals_K_not_loop` | mocks vllm.LLM; asserts SamplingParams(n=K) is built once, not K times (AGENT.md §3.1) | mock |
| `test_close_releases_VRAM` | check `torch.cuda.memory_allocated()` decreases after `close()` (small but non-zero delta acceptable) | yes |
| `test_greedy_with_K_above_one_warns_or_errors` | AGENT.md §8.1: temperature=0 with K>1 is wasteful; pick warn or refuse and pin via test | mock |

For the determinism test, use a tiny model (qwen2.5-0.5b-instruct) and a 4-prompt fixture so the test takes seconds, not minutes. Compare bytes of the joined output strings, not just substring equality.

### 3.8.3 Common pitfalls (enumerated in AGENT.md §8 — re-read before coding)

- **VRAM not freed across models** — the `close()` pattern above is required when iterating across H1's three models in one process. Better: load one model, run all prompts × seeds × datasets for that model, close, then load the next.
- **vLLM seed semantics may drift across versions** — once you find a vLLM version that produces byte-identical samples for `(model, prompts, seed)`, **pin it** in `requirements.txt`. Document the pinned version in [TODO.md](TODO.md).
- **Greedy** (`temperature=0`) with `K>1` is wasteful (samples identical). For H3's "direct sampling at τ=0", run with `K=1` and replicate.

### 3.8.4 Acceptance

3.8 is complete when:
- The 6 tests above pass live on the GPU box (not skipped, not mocked-only for the determinism one).
- `pytest tests/` total is 209 + (new tests).
- TODO.md updated to record the pinned vLLM version and any deviations from the suggested interface.

---

## 4 — Phase 3.10 — smoke test

**Script path:** `scripts/smoke_test.py` (does not exist yet)

**Command (the contract):**
```bash
python -m scripts.smoke_test --model qwen2.5-1.5b-instruct --dataset gsm8k --num-prompts 10 --K 4
```

### 4.1 What it must do (12 ordered steps)

1. Parse CLI args (`--model`, `--dataset`, `--num-prompts`, `--K`, `--seed=0`).
2. `paths = src.pipeline.paths.load_paths()`.
3. `setup_run_logger(paths.logs_dir, experiment="smoke_<model>_<dataset>")`.
4. Load model entry from `configs/models.yaml`; load dataset entry from `configs/datasets.yaml`.
5. Load the dataset via HF Datasets (`datasets.load_dataset(hf_id, hf_config, split=split)`); take first `num_prompts`.
6. **Format prompts**:
   - If `model.prompt_mode == "chat"`: build a list of `{"role": "system", ...}, {"role": "user", ...}` messages using `dataset.templates.chat.system` and `dataset.templates.chat.user_template.format(**fields)`. Apply via `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. (vLLM doesn't apply chat templates itself; you must pre-format.)
   - If `prompt_mode == "few_shot"`: prepend the few-shot examples file content (these don't exist yet — H2 work; the smoke test only uses chat-mode models so this branch isn't exercised here).
7. Construct a `CacheKey(model=model.hf_id, dataset="gsm8k", K=K, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=...,  seed=0, prompt_template_version="v1")`.
8. Check `cache_exists(paths.samples_dir, key)`; if yes, read; if no, sample via `VllmRunner` and `write_cache`. Each cached record schema:
   ```json
   {"prompt_id": "<idx>", "prompt": "<formatted>", "ground_truth": "<from dataset>", "samples": ["...", "...", "...", "..."]}
   ```
9. For each (prompt, sample) pair: `u = verify(dataset_alias, sample, ground_truth)`; log via `log_verifier_decision(paths.logs_dir, dataset_alias, {...})`.
10. Build `(M, K)` utility array from per-prompt utilities; `est = compute_rational_gap(arr)`.
11. `ci = bootstrap_ci_over_prompts(est.per_prompt_R_hat_K, n_resamples=1000, confidence=0.95, seed=0)`.
12. Print the table-format string from [methodology/audit_and_uncertainty.md](methodology/audit_and_uncertainty.md):
    `R_hat_K = {est.R_hat_K:.3f} [{ci.ci_low:.3f}, {ci.ci_high:.3f}] (95% bootstrap CI over prompts)`.
    Also log `U_circ_K`, `U_bar_K` separately. Call `log_compute(paths.logs_dir, experiment="smoke_qwen_gsm8k", gpu_hours=...)`.

### 4.2 HumanEval ground-truth assembly

When you eventually smoke-test HumanEval, the dataset's `test` field is a `def check(candidate)` body — but the trailing `check(<entry_point>)` invocation is **not** in the field. The dataset loader must assemble:
```python
ground_truth = sample["test"] + f"\ncheck({sample['entry_point']})\n"
```
Then pass `ground_truth` to the verifier as-is. This is documented in `configs/datasets.yaml` `humaneval.entry_point_field`.

### 4.3 Acceptance

3.10 is complete when:
- The full command above completes in < 5 minutes on a small GPU.
- It writes a cache file under `${samples_dir}/` and a verifier audit log under `${logs_dir}/verifier/gsm8k_log.jsonl`.
- It prints a non-zero `R_hat_K` (Qwen-1.5B on GSM8K should show a noticeable gap — if you see 0.0 or 1.0, something is wrong with extraction or sampling).
- A new test in `tests/test_smoke_integration.py` runs it on K=2 + 3 prompts and asserts the artefacts exist.
- AGENT.md §7 says "no change is complete until smoke test passes" — this is the smoke test, so passing it unblocks H1.

---

## 5 — Phase 4 — H1 production run

After 3.10 passes:

```bash
# Run H1 across the panel; the runner script does not yet exist.
# Build it in scripts/run_h1.py (or .sh wrapping a Python entry point).
# Read configs/experiments/h1.yaml — it pins all the parameters.
```

### Design intent

- 3 models × 3 datasets × 3 seeds = 27 sampling runs at K_max = 64.
- Use the cache: a re-run after a verifier change should be near-instant (sampling is the expensive step).
- For each (model, dataset, seed), build a `(M, K_max=64)` utility array via the verifier; save it to `${results_dir}/h1/<cell>.npz` along with the bootstrap-CI summary.
- For the saturation-curve plot, use `U_circ_at_K` / `U_bar_at_K` / `R_hat_at_K` from `src/metrics/rational_gap.py` at each $K' \in \{1, 2, 4, 8, 16, 32, 64\}$. **No binomial closed form, no expectation framing** — see [methodology/no_expectation_framing.md](methodology/no_expectation_framing.md).
- Bootstrap CI per (model, dataset, seed) over prompts (B=1000, 95%); cross-seed mean ± std on top.

### Acceptance for H1

- `${results_dir}/h1/` contains 9 cell-summary JSONs plus 1 panel-summary JSON.
- `${results_dir}/figures/h1_saturation.pdf` shows R_hat_K vs K, one curve per (model, dataset) with CI bands.
- Compute logged via `log_compute()`.

---

## 6 — Phase 5 — H2 (Tülu-3 trajectory)

**Hypothesis (verbatim from [methodology/hypotheses.md](methodology/hypotheses.md)):** alignment does not *eliminate* the gap. Along SFT → DPO → RLVR, $\bar{U}_K$ rises monotonically while $U^\circ_K$ may stagnate or decline (distributional sharpening narrows the reachable region).

### Pre-flight (must do before sampling)

1. **Author few-shot files.** `configs/datasets.yaml` references three files that do not yet exist:
   - `configs/few_shot/gsm8k_8shot.txt` — 8 GSM8K examples in "Question / Answer" format (use the canonical Wei et al. CoT examples).
   - `configs/few_shot/math_4shot.txt` — 4 MATH examples in "Problem / Solution" format with `\boxed{}` answers.
   - `configs/few_shot/humaneval_3shot.txt` — 3 HumanEval-style examples (the base model needs in-context examples to follow the function-completion convention).
   Each file is plain text; the dataset loader concatenates the file's contents in front of the user template. Bump `prompt_template_version` to `v2` in `configs/datasets.yaml` since this changes the prompt — old cache entries become invalid.
2. **Re-confirm Tülu trajectory weights are downloadable.** `meta-llama/Llama-3.1-8B` is gated; ensure `huggingface-cli login` has access. Tülu-3 SFT/DPO/RLVR are at `allenai/Llama-3.1-Tulu-3-8B-{SFT,DPO,}` (RLVR is the unsuffixed final).

### Design intent

- 4 models (base + SFT + DPO + RLVR) × 2 datasets (GSM8K, MATH; HumanEval is excluded from H2) × 3 seeds × K_max = 64.
- Plot $U^\circ_K$ and $\bar{U}_K$ separately as bars or twinned lines per stage — do NOT collapse to just $\hat{\mathcal{R}}_K$ since the strong form of H2 distinguishes "gap shrinks because $U^\circ_K$ shrinks" from "gap shrinks because $\bar{U}_K$ catches up".
- Configuration in `configs/experiments/h2.yaml` (does not exist yet — author it following h1.yaml's shape).

---

## 7 — Phase 6 — H3 (inference mechanisms)

**Hypothesis:** deployable inference procedures (CoT, self-consistency, MCTS) close a measurable fraction of the residual gap $U^\circ_K - u_I$ to the reachable upper bound.

### Pre-flight: extend `CacheKey`

The current `CacheKey` has no slot for "inference procedure" or its kwargs. Two clean options:

**Option A — Optional fields on existing CacheKey** (recommended):
```python
inference_procedure: str = "direct"          # "direct" | "cot" | "self_consistency" | "mcts" | "best_of_n_oracle"
inference_kwargs: tuple[tuple[str, str], ...] = ()  # frozen sorted (name, repr) pairs; default empty
```
- Default values keep existing caches valid (same fingerprint).
- Bump `CACHE_FORMAT_VERSION` from 1 → 2 anyway, because the on-disk header schema gains the field — old files will then fail the version check and force recompute. Per AGENT.md §3.2 this is the correct behaviour.
- Add tests: every (procedure, kwargs) combo gives a unique fingerprint; default values match a Phase-3.10 cache.

**Option B — Sibling `H3CacheKey`**:
- Cleaner separation but doubles the cache machinery. Reject unless A causes real problems.

Go with A. Implement, regenerate the cache (small if smoke-only at this point), update `tests/test_cache.py`, update `tests/conftest.py` `cache_key_factory` defaults.

### Design intent for H3

- One $\pi_\theta$ (Tülu-3-RLVR) × 3 datasets × 3 seeds × five procedures.
- Procedures (per [methodology/hypotheses.md](methodology/hypotheses.md)):
  1. Direct sampling at τ=0 (greedy; K=1 only — see AGENT.md §8.1).
  2. Direct sampling at τ=0.7.
  3. Direct sampling at τ=1.0 (this is the H1 baseline).
  4. CoT prompting (system prompt nudges step-by-step reasoning).
  5. Self-consistency: 8 CoT samples → majority vote.
  6. MCTS with budget-matched search (Tree-of-Thought-style; specify branching/depth in config).
  7. Oracle Best-of-N at N=K (drawn as a horizontal reference line, **not** a candidate procedure — see hypotheses.md).
- Per-procedure utility `u_I` is a scalar per (prompt, seed); aggregate as `mean over prompts`. Plot residual gap $U^\circ_K - u_I$ with bootstrap CI.
- Configuration in `configs/experiments/h3.yaml`. Implementation in `src/sampling/inference_procedures.py`. Each procedure registered into a registry similar to verifier registry.

---

## 8 — Phase 7 — H4 (reasoning length)

**Hypothesis:** $\hat{\mathcal{R}}_K(L)$ as a function of max reasoning length $L \in \{0, 64, 128, 256, 512, 1024, 2048\}$ on MATH and GSM8K via two-stage budget forcing. Relationship may be non-monotonic.

### Pre-flight: extend `CacheKey` again

Add `max_reasoning_length: int | None = None` (None for non-H4 runs). Same approach as H3 — default keeps existing caches valid. The H3 bump to v2 already covered the format version; H4 can either piggyback (still v2) or bump to v3 if H3 has already shipped before H4 starts.

### Design intent

- Two-stage budget forcing (à la s1 paper): first stage generates up to $L$ reasoning tokens; if not finished, force a `</think>` (or model-appropriate) closing token then sample the answer.
- Implementation lives in `src/sampling/inference_procedures.py` as `budget_forced(...)`.
- One $\pi_\theta$ (Tülu-3-RLVR) × 2 datasets × 3 seeds × 7 length values × K=64.
- Plot `R_hat_K(L)` vs `L`; expect non-monotonic shape.

---

## 9 — Phase 8 — figures and tables

`src/plotting/generate_all.py` (does not exist yet). Each hypothesis has its own figure spec; design as you go but commit to:

- **One PDF and one PNG per figure.**
- **Matplotlib only**; no seaborn or plotnine (keep dependencies minimal per AGENT.md §5).
- **Bootstrap CI bands** on every line plot; format from `audit_and_uncertainty.md`.
- **No emoji, no decorative styling**; this is a paper figure.

Inputs: the per-cell JSON summaries from each hypothesis run. Outputs: `${results_dir}/figures/{h1,h2,h3,h4}_<name>.{pdf,png}`.

---

## 10 — Phase 9 — Manual audit

Per [methodology/audit_and_uncertainty.md](methodology/audit_and_uncertainty.md): per dataset, hand-verify 100 prompts.

- 50 sampled uniformly at random from a real model run (use the H1 cache).
- 50 *load-bearing*: prompts where exactly 1 of K samples passes the automated verifier. These are the cases where the verifier matters most.

For each prompt, record in `tests/manual_audit_<dataset>.md`:
- prompt id, generation excerpt, ground truth, automated verifier output, **your** judgement, and any disagreement note.

Acceptance: three audit files committed; any disagreements > 5% of the audit set trigger a verifier patch + a re-run.

---

## 11 — Phase 10 — Finalisation

Done when:

- ✅ All 4 hypotheses (H1–H4) have results JSONs in `${results_dir}/h{N}/`.
- ✅ All figures rendered to `${results_dir}/figures/`.
- ✅ Three `tests/manual_audit_<dataset>.md` files committed.
- ✅ `tests/` runs green with the final test count documented in TODO.md.
- ✅ [README.md](../README.md) "Status" section updated to reflect what is now runnable. The current README is forward-looking (per the evaluator REPORT.md) — trim it down to what is actually runnable when the project ships.
- ✅ [TODO.md](TODO.md) shows every phase as `✅ done` with final test count and total compute.
- ✅ `${logs_dir}/compute_budget.jsonl` reflects the cumulative GPU-hours spent (sanity-check against the README's ~85 GPU-hour estimate).
- ✅ `git log` is clean: meaningful commit messages, no `--amend` rewrites of pushed commits, no `--no-verify` bypasses.
- ✅ A final commit titled "Final results: H1-H4 + figures + audit" anchors the published state.

---

## 12 — Open design questions you will hit

These are flagged so you don't burn cycles re-discovering them. Make a decision, document it in TODO.md, and move on.

1. **MATH HF dataset id.** `configs/datasets.yaml` uses `EleutherAI/hendrycks_math`. Confirm at first load — if HF moved it, switch to `hendrycks/competition_math` or whatever the current canonical id is. The field names (`problem`, `solution`) are likely stable across forks but verify.
2. **vLLM version pin.** Once you find a version that gives byte-identical samples for fixed seed, pin it exactly in `requirements.txt`. Document the pin in TODO.md with a one-line note.
3. **Few-shot file content.** The Wei et al. 8-shot CoT for GSM8K is the standard; pick a published source, cite it in the file's header comment, and don't make up new examples.
4. **CacheKey extension timing.** Don't bolt on H3 fields until you actually start H3 — but do read §7 above so you know how to do it cleanly when the time comes.
5. **Models you can't run on this box.** If the GPU isn't big enough for Llama-3.1-8B (consumer GPUs), drop Llama-3.1-8B from the H1 panel and document the deviation. The Tülu-3-RLVR + Qwen-2.5-7B comparison is still informative.

---

## 13 — Reading map (the whole repo)

| Path | Purpose |
|---|---|
| [AGENT.md](AGENT.md) | Hard rules. The contract. |
| [TODO.md](TODO.md) | Current status, phase-by-phase. |
| [HANDOFF.md](HANDOFF.md) | This file. |
| [methodology/](methodology/) | Methodological commitments. Read all 7 files. |
| [../README.md](../README.md) | User-facing description (forward-looking; trim at finalisation). |
| [../src/verification/](../src/verification/) | GSM8K, MATH, HumanEval verifiers + registry. Done, tested. |
| [../src/pipeline/](../src/pipeline/) | Paths, cache, logging. Done, tested. |
| [../src/metrics/](../src/metrics/) | `R_hat_K` + saturation helpers + bootstrap CI. Done, tested. |
| [../src/sampling/](../src/sampling/) | **Empty.** Build `vllm_runner.py` and later `inference_procedures.py` here. |
| [../src/plotting/](../src/plotting/) | **Empty.** Build figure generators here. |
| [../scripts/](../scripts/) | **Empty.** Build `smoke_test.py`, `run_h{1..4}.py` here. |
| [../configs/](../configs/) | `models.yaml`, `datasets.yaml`, `experiments/h1.yaml`. H2/H3/H4 configs to author. |
| [../tests/](../tests/) | All current tests, plus `conftest.py` with shared fixtures. |
| [../requirements.txt](../requirements.txt) | Minimal: pytest, math-verify, pyyaml, numpy. Add vllm + pinned torch when you install. |

---

## 14 — Communication protocol with the user

The user is the paper author (see [methodology/user.md](methodology/user.md)). When in doubt:
- **Pause and ask** rather than guess on methodology. The user has corrected several assumptions during prior sessions; re-correcting later is more expensive than asking now.
- **Surface deviations** (e.g. dropping Llama-3.1-8B from H1 because the GPU is too small) explicitly in chat AND record them in TODO.md.
- **Quote the paper formula** when explaining design choices, not the implementation. Code follows the paper, not the other way round.

Good luck. The methodology is solid; the code so far is solid; finishing this is mostly disciplined execution.
