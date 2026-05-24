"""H1 cell runner — sampling + verification + saturation curve for one
(model, dataset, seed) cell.

Per HANDOFF.md §5 and configs/experiments/h1.yaml: H1 samples once at
K_max per cell, then derives the saturation curve at K' <= K_max by
direct truncation of the (M, K_max) utility matrix to its first K'
columns. **No** binomial closed forms, **no** "expectation" framing —
see methodology/no_expectation_framing.md.

Each cell writes:

    ${samples_dir}/v1_<dataset>_<model>_K<K>_<hash>.jsonl.gz
        — gzipped JSONL of per-prompt samples (CacheKey-deduplicated).
    ${results_dir}/h1/<model_alias>_<dataset>_seed<S>.json
        — per-cell summary: aggregates, saturation curve, bootstrap CI,
          per-prompt R_hat_K array (for downstream cross-seed and
          cross-cell aggregation).
    ${logs_dir}/verifier/<dataset>_log.jsonl
        — append: one record per (prompt, sample) verifier call.
    ${logs_dir}/compute_budget.jsonl
        — append: gpu_hours for this cell's sampling.

Usage:
    python -m scripts.run_h1 --model tulu3-8b-rlvr --dataset gsm8k \\
        --seed 0 --K 64 --num-prompts 200

Default K=64, num-prompts=null (use the full test split). For the
minimal-cell sanity check before running the full panel, set
--num-prompts to a smaller integer (e.g. 200) so wall-time is bounded.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from src.metrics.rational_gap import (
    R_hat_at_K,
    U_bar_at_K,
    U_circ_at_K,
    bootstrap_ci_over_prompts,
    compute_rational_gap,
)
from src.pipeline.cache import CacheKey, cache_exists, read_cache, write_cache
from src.pipeline.logging_utils import log_compute, log_verifier_decision, setup_run_logger
from src.pipeline.paths import load_paths
from src.sampling.vllm_runner import SamplingConfig, VllmRunner
from src.verification.interface import verify as verify_dispatch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _format_chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _assemble_ground_truth(row: dict[str, Any], ds_cfg: dict[str, Any]) -> str:
    """Build the verifier's ``ground_truth`` argument from a dataset row.

    HumanEval: append ``check(<entry_point>)`` to the ``test`` field.
    LiveCodeBench: bundle public + private test cases (parsed from
        JSON-encoded strings) plus the ``starter_code`` field into a
        single JSON blob the verifier can re-parse.
    self_judge (preference mode): the reference y^+ is the assistant
        message text. UltraFeedback's `chosen` is a list of role-content
        dicts; extract the assistant turn. Otherwise treat as raw string.
    Everything else: just the raw ground-truth field.
    """
    if ds_cfg.get("verifier") == "livecodebench":
        public = row.get("public_test_cases", "[]")
        private = row.get("private_test_cases", "[]")
        try:
            tests = json.loads(public) + json.loads(private)
        except (json.JSONDecodeError, TypeError):
            tests = []
        return json.dumps({
            "tests": tests,
            "starter_code": row.get("starter_code", ""),
        })
    if ds_cfg.get("verifier") == "self_judge":
        chosen = row[ds_cfg["ground_truth_field"]]
        if isinstance(chosen, list):
            for msg in chosen:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""  # malformed row
        return str(chosen)
    gt = row[ds_cfg["ground_truth_field"]]
    if "entry_point_field" in ds_cfg:
        gt = gt + f"\ncheck({row[ds_cfg['entry_point_field']]})\n"
    return gt


def _saturation_grid(K_max: int) -> list[int]:
    """Powers of 2 up to K_max, with K_max itself appended."""
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= K_max]
    if K_max not in grid:
        grid.append(K_max)
    return sorted(set(grid))


def main() -> None:
    parser = argparse.ArgumentParser(description="H1 cell runner")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset", required=True,
        choices=["gsm8k", "math", "humaneval", "matharena", "livecodebench", "ultrafeedback", "alpaca_eval"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=64, help="K_max; samples per prompt")
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="If set, truncate the test split to this many prompts (for minimal-cell runs).",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument(
        "--api-concurrency", type=int, default=10,
        help="Max in-flight requests for backend:api models (proxy rate-limit ceiling).",
    )
    parser.add_argument(
        "--judge", choices=["self", "api"], default="self",
        help="Preference judge backend: 'self' (vLLM, strict-self) or 'api' (hosted model).",
    )
    parser.add_argument(
        "--judge-model", default="deepseek-v4-flash",
        help="models.yaml alias of the API judge when --judge api.",
    )
    parser.add_argument(
        "--judge-local-model", default=None,
        help="models.yaml alias of a FIXED local vLLM judge for --judge self "
             "(e.g. qwen2.5-14b-instruct). Default None = strict-self (the "
             "candidate model judges itself). Use a fixed independent judge to "
             "re-judge preference cells without self-preference bias.",
    )
    parser.add_argument(
        "--judge-L", type=int, default=None,
        help="Override judge calls per pair (default: ds_cfg.judge_L, or 3 for API).",
    )
    args = parser.parse_args()

    paths = load_paths()
    paths.ensure_dirs()
    h1_dir = paths.results_dir / "h1"
    h1_dir.mkdir(parents=True, exist_ok=True)

    experiment = f"h1_{args.model}_{args.dataset}_seed{args.seed}"
    _root, log_file = setup_run_logger(paths.logs_dir, experiment)
    log = logging.getLogger(__name__)
    log.info(
        "Starting %s | K=%d, num_prompts=%s, seed=%d, max_tokens=%d",
        experiment, args.K, args.num_prompts, args.seed, args.max_tokens,
    )

    models_cfg = _load_yaml(_REPO_ROOT / "configs" / "models.yaml")
    datasets_cfg = _load_yaml(_REPO_ROOT / "configs" / "datasets.yaml")

    if args.model not in models_cfg:
        raise SystemExit(f"Unknown model alias {args.model!r}; known: {sorted(models_cfg)}")
    model_cfg = models_cfg[args.model]
    ds_cfg = datasets_cfg[args.dataset]
    if model_cfg["prompt_mode"] != "chat":
        raise SystemExit(
            f"H1 is chat-mode only; {args.model} is {model_cfg['prompt_mode']!r}. "
            f"H2 covers few-shot mode (see HANDOFF §6)."
        )

    # --- load + slice dataset --------------------------------------------
    if "subsets" in ds_cfg:
        # Multi-source dataset (e.g. MathArena: AIME 2025 + BRUMO 2025).
        # Each entry in `subsets` is either:
        #   - a config name to combine with ds_cfg["hf_id"], OR
        #   - a full "org/dataset" path when ds_cfg["hf_id"] is null.
        # Take prompts_per_subset from each split, then concatenate.
        from datasets import Value, concatenate_datasets
        parts = []
        for subset in ds_cfg["subsets"]:
            if ds_cfg.get("hf_id"):
                log.info("Loading dataset: %s / %s / split=%s",
                         ds_cfg["hf_id"], subset, ds_cfg["split"])
                part = load_dataset(ds_cfg["hf_id"], subset, split=ds_cfg["split"])
            else:
                log.info("Loading dataset: %s / split=%s", subset, ds_cfg["split"])
                part = load_dataset(subset, split=ds_cfg["split"])
            # Different subsets can have different inferred types for the
            # same field (e.g. AIME 2025's `answer` is int64, BRUMO 2025's
            # is string). Coerce the prompt + ground-truth columns to
            # string so concatenate_datasets succeeds; the verifier reads
            # them as strings anyway.
            for col in (ds_cfg.get("prompt_field"), ds_cfg.get("ground_truth_field")):
                if col and col in part.features:
                    feat = part.features[col]
                    if not (isinstance(feat, Value) and feat.dtype == "string"):
                        part = part.cast_column(col, Value("string"))
            n = ds_cfg.get("prompts_per_subset")
            if n is not None:
                part = part.select(range(min(n, len(part))))
            parts.append(part)
        raw_ds = concatenate_datasets(parts)
    elif ds_cfg.get("hf_data_file"):
        # Script-based HF datasets (e.g. AlpacaEval) aren't loadable by the
        # current datasets lib; pull the raw JSON file from the hub directly.
        from huggingface_hub import hf_hub_download
        fp = hf_hub_download(ds_cfg["hf_id"], ds_cfg["hf_data_file"], repo_type="dataset")
        log.info("Loading hub file: %s/%s", ds_cfg["hf_id"], ds_cfg["hf_data_file"])
        raw_ds = load_dataset("json", data_files=fp, split="train")
    else:
        ds_kwargs: dict[str, Any] = {"path": ds_cfg["hf_id"], "split": ds_cfg["split"]}
        if ds_cfg.get("hf_config"):
            ds_kwargs["name"] = ds_cfg["hf_config"]
        log.info("Loading dataset: %s", ds_kwargs)
        raw_ds = load_dataset(**ds_kwargs)
    # Optional date filter — used by LiveCodeBench to keep only
    # post-cutoff problems for contamination-resistant evaluation.
    min_date = ds_cfg.get("min_contest_date")
    if min_date is not None:
        from datetime import datetime
        cutoff = datetime.fromisoformat(min_date)
        n_before = len(raw_ds)
        raw_ds = raw_ds.filter(
            lambda r: r.get("contest_date") is not None
                      and r["contest_date"] >= cutoff,
        )
        log.info("min_contest_date=%s filter: %d -> %d",
                 min_date, n_before, len(raw_ds))
    if args.num_prompts is not None:
        raw_ds = raw_ds.select(range(min(args.num_prompts, len(raw_ds))))
    raw_inputs = [dict(row) for row in raw_ds]
    # Decode byte-string fields (some HF datasets return bytes). Apply
    # to known string fields only.
    for row in raw_inputs:
        for f in (ds_cfg.get("prompt_field"), ds_cfg.get("ground_truth_field")):
            if f and isinstance(row.get(f), (bytes, bytearray)):
                row[f] = row[f].decode("utf-8")
    log.info("Using %d prompts", len(raw_inputs))

    key = CacheKey(
        model=model_cfg["hf_id"],
        dataset=args.dataset,
        K=args.K,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_template_version=ds_cfg["prompt_template_version"],
        num_prompts=len(raw_inputs),
    )
    log.info("Cache key fingerprint: %s", key.fingerprint())

    # --- format prompts --------------------------------------------------
    # backend:api models (H5 hosted subjects) have no local tokenizer —
    # the hosted endpoint applies its own chat template, so we pass the
    # raw user text and hand the system prompt to ApiRunner.
    is_api = model_cfg.get("backend") == "api"
    if is_api and ds_cfg.get("verifier") == "self_judge":
        raise SystemExit(
            "backend:api models support deterministic-verifier datasets only "
            "(H1/H3 of H5); self_judge preference verification is not wired "
            "for API backends."
        )
    chat_tmpl = ds_cfg["templates"]["chat"]
    prompt_field = ds_cfg["prompt_field"]

    if is_api:
        tokenizer = None
        log.info("API backend (%s); skipping local tokenizer", model_cfg["hf_id"])
    else:
        log.info("Loading tokenizer for %s", model_cfg["hf_id"])
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"])

    prompts: list[str] = []
    raw_user_prompts: list[str] = []   # raw user question, used by self_judge verifier
    ground_truths: list[str] = []
    for row in raw_inputs:
        raw_q = row[prompt_field]
        if isinstance(raw_q, (bytes, bytearray)):
            raw_q = raw_q.decode("utf-8")
        raw_user_prompts.append(str(raw_q))
        user_text = chat_tmpl["user_template"].format(**{prompt_field: raw_q})
        if is_api:
            prompts.append(user_text)  # hosted model applies its own template
        else:
            prompts.append(_format_chat_prompt(tokenizer, chat_tmpl["system"], user_text))
        ground_truths.append(_assemble_ground_truth(row, ds_cfg))

    # --- sample ----------------------------------------------------------
    if cache_exists(paths.samples_dir, key):
        log.info("Cache hit; reading samples")
        records = list(read_cache(paths.samples_dir, key))
        sample_seconds = 0.0
    else:
        log.info("Cache miss — sampling K=%d completions per prompt", args.K)
        t0 = time.time()
        cfg = SamplingConfig(
            temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens,
        )
        if is_api:
            from src.sampling.api_runner import ApiModelSpec, ApiRunner
            spec = ApiModelSpec.from_model_cfg(model_cfg)
            resume_path = paths.samples_dir / f"apiresume_{key.fingerprint()}.jsonl"
            log.info("API sampling: %d prompts x K=%d, concurrency=%d, resume=%s",
                     len(prompts), args.K, args.api_concurrency, resume_path)
            with ApiRunner(
                spec, system=chat_tmpl["system"], resume_path=resume_path,
                concurrency=args.api_concurrency,
            ) as runner:
                samples_list = runner.sample(
                    prompts, K=args.K, seed=args.seed, config=cfg,
                )
        else:
            with VllmRunner(
                model_cfg["hf_id"],
                gpu_memory_utilization=args.gpu_mem,
                enforce_eager=True,
            ) as runner:
                samples_list = runner.sample(
                    prompts, K=args.K, seed=args.seed, config=cfg,
                )
        sample_seconds = time.time() - t0
        log.info("Sampling done in %.1f s", sample_seconds)

        records = [
            {
                "prompt_id": f"{args.dataset}_{i}",
                "prompt": prompts[i],
                "ground_truth": ground_truths[i],
                "samples": samples_list[i],
            }
            for i in range(len(prompts))
        ]
        write_cache(paths.samples_dir, key, records)
        log_compute(
            paths.logs_dir,
            experiment=experiment,
            gpu_hours=sample_seconds / 3600.0,
            metadata={
                "model": args.model,
                "dataset": args.dataset,
                "num_prompts": len(prompts),
                "K": args.K,
                "seed": args.seed,
                "stage": "sampling",
            },
        )

    # --- verify ----------------------------------------------------------
    # Two code paths:
    #
    # 1. Preference mode (ds_cfg["verifier"] == "self_judge"): each
    #    (prompt, sample) pair is judged by an LLM (the same model under
    #    test, for H1 strict self) against the human-preferred reference
    #    y^+. L=5 i.i.d. judge calls per pair with random A/B position;
    #    ternary outcome in {0, 0.5, 1} by strict majority. Batched
    #    through vLLM. See src/verification/self_judge.py.
    #
    # 2. Verifier-binary mode (gsm8k, math, humaneval, matharena,
    #    livecodebench): parallelised across threads since code-execution
    #    verifiers spawn subprocesses dominated by I/O wait; pure-Python
    #    verifiers also benefit from parallel CPU work on the M*K pairs
    #    at no correctness cost.
    #
    # Per AGENT.md §3.3 every verifier decision is logged. log_verifier_
    # decision appends single short JSON lines (< PIPE_BUF), so
    # concurrent appends from threads are atomic.
    M = len(records)
    samples_per_prompt = [rec["samples"] for rec in records]

    if ds_cfg.get("verifier") == "self_judge" and args.judge == "api":
        # --- preference mode: hosted API judge (no GPU) ---
        from src.sampling.api_runner import ApiModelSpec
        from src.verification.api_judge import score_matrix_api
        judge_cfg = models_cfg[args.judge_model]
        spec = ApiModelSpec.from_model_cfg(judge_cfg)
        L_judge = int(args.judge_L or ds_cfg.get("judge_L", 3))
        resume = paths.samples_dir / f"apijudge_{key.fingerprint()}_{args.judge_model}.jsonl"
        log.info(
            "Preference mode (API judge=%s): %d * %d * L=%d = %d judge calls",
            args.judge_model, M, args.K, L_judge, M * args.K * L_judge,
        )
        outcome = score_matrix_api(
            spec,
            raw_prompts=raw_user_prompts,
            candidates=samples_per_prompt,
            references=ground_truths,
            L=L_judge,
            seed=args.seed,
            concurrency=args.api_concurrency,
            resume_path=resume,
        )
        utility = outcome.utility.astype(float)
        log.info(
            "API judge done: parse_failure_rate=%.4f, n_calls=%d",
            outcome.parse_failure_rate, outcome.n_judge_calls,
        )
        for i in range(M):
            for k in range(args.K):
                log_verifier_decision(
                    paths.logs_dir, args.dataset,
                    {
                        "experiment": experiment,
                        "prompt_id": records[i]["prompt_id"],
                        "k": k,
                        "utility": float(utility[i, k]),
                        "raw_verdicts": [float(v) for v in outcome.raw_verdicts[i, k]],
                        "a_is_candidate": [bool(b) for b in outcome.a_is_candidate[i, k]],
                        "judge": args.judge_model,
                        "seed": args.seed,
                    },
                )
    elif ds_cfg.get("verifier") == "self_judge":
        # --- preference mode: batched vLLM self-judge ---
        from src.verification.self_judge import score_matrix
        L_judge = int(args.judge_L or ds_cfg.get("judge_L", 5))
        # Strict-self by default; a fixed independent local judge if requested
        # (re-judges cached candidates without the candidate's self-preference).
        if args.judge_local_model:
            judge_cfg = models_cfg[args.judge_local_model]
            judge_hf_id = judge_cfg["hf_id"]
            judge_tok = AutoTokenizer.from_pretrained(judge_hf_id)
            log.info("Fixed local judge: %s (%s)", args.judge_local_model, judge_hf_id)
        else:
            judge_hf_id = model_cfg["hf_id"]
            judge_tok = tokenizer
        log.info(
            "Preference mode: %d * %d * L=%d = %d judge calls via vLLM",
            M, args.K, L_judge, M * args.K * L_judge,
        )
        with VllmRunner(
            judge_hf_id,
            gpu_memory_utilization=args.gpu_mem,
            enforce_eager=True,
        ) as judge_runner:
            outcome = score_matrix(
                judge_runner=judge_runner,
                judge_tokenizer=judge_tok,
                raw_prompts=raw_user_prompts,
                candidates=samples_per_prompt,
                references=ground_truths,
                L=L_judge,
                seed=args.seed,
            )
        utility = outcome.utility.astype(float)
        log.info(
            "Judge done: parse_failure_rate=%.4f, n_calls=%d",
            outcome.parse_failure_rate, outcome.n_judge_calls,
        )
        # Audit log: one record per (prompt, sample) summarising L votes.
        # ``a_is_candidate`` is the per-call position-coin record needed by
        # appendix D.2 (position-bias rate) and C.2 (L-sensitivity
        # re-aggregation); see :class:`JudgeOutcome` for the contract.
        for i in range(M):
            for k in range(args.K):
                log_verifier_decision(
                    paths.logs_dir,
                    args.dataset,
                    {
                        "experiment": experiment,
                        "prompt_id": records[i]["prompt_id"],
                        "k": k,
                        "utility": float(utility[i, k]),
                        "raw_verdicts": [float(v) for v in outcome.raw_verdicts[i, k]],
                        "a_is_candidate": [bool(b) for b in outcome.a_is_candidate[i, k]],
                        "seed": args.seed,
                    },
                )

    else:
        # --- verifier-binary mode (existing path) ---
        import os
        from concurrent.futures import ThreadPoolExecutor

        log.info("Verifying %d (prompt, sample) pairs", M * args.K)
        utility = np.zeros((M, args.K), dtype=float)

        tasks: list[tuple[int, int, str, str, str]] = []
        for i, rec in enumerate(records):
            for k, sample in enumerate(rec["samples"]):
                tasks.append((i, k, sample, rec["ground_truth"], rec["prompt_id"]))

        def _verify_task(t: tuple[int, int, str, str, str]) -> tuple[int, int, float, str]:
            i, k, sample, gt, prompt_id = t
            return i, k, verify_dispatch(args.dataset, sample, gt), prompt_id

        n_workers = min(32, (os.cpu_count() or 4))
        log.info("Verifying with %d worker threads", n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for i, k, u, prompt_id in pool.map(_verify_task, tasks):
                utility[i, k] = u
                log_verifier_decision(
                    paths.logs_dir,
                    args.dataset,
                    {
                        "experiment": experiment,
                        "prompt_id": prompt_id,
                        "k": k,
                        "utility": u,
                        "seed": args.seed,
                    },
                )

    # --- aggregate at K_max + saturation curve + bootstrap CI ------------
    est = compute_rational_gap(utility)
    ci = bootstrap_ci_over_prompts(
        est.per_prompt_R_hat_K, n_resamples=1000, confidence=0.95, seed=args.seed,
    )

    K_grid = _saturation_grid(args.K)
    saturation = []
    for K_prime in K_grid:
        saturation.append(
            {
                "K": K_prime,
                "U_circ_K": U_circ_at_K(utility, K_prime),
                "U_bar_K": U_bar_at_K(utility, K_prime),
                "R_hat_K": R_hat_at_K(utility, K_prime),
            }
        )

    # --- save per-cell results JSON --------------------------------------
    results = {
        "experiment": experiment,
        "model": args.model,
        "model_hf_id": model_cfg["hf_id"],
        "dataset": args.dataset,
        "seed": args.seed,
        "M": int(est.M),
        "K_max": int(est.K),
        "cache_key_fingerprint": key.fingerprint(),
        "aggregates_at_K_max": {
            "U_circ_K": est.U_circ_K,
            "U_bar_K": est.U_bar_K,
            "R_hat_K": est.R_hat_K,
        },
        "bootstrap_R_hat_K_at_K_max": {
            "mean": ci.mean,
            "ci_low": ci.ci_low,
            "ci_high": ci.ci_high,
            "confidence": ci.confidence,
            "n_resamples": ci.n_resamples,
        },
        "saturation_curve": saturation,
        "per_prompt_R_hat_K": est.per_prompt_R_hat_K.tolist(),
        "per_prompt_U_circ_K": est.per_prompt_U_circ_K.tolist(),
        "per_prompt_U_bar_K": est.per_prompt_U_bar_K.tolist(),
        "sampling_seconds": sample_seconds,
    }
    # Tag the filename with the judge when it is NOT strict-self, so an
    # independent-judge re-run (fixed local or API) sits alongside the
    # original strict-self result instead of overwriting it — both are
    # needed for the judge-robustness comparison.
    if ds_cfg.get("verifier") == "self_judge" and args.judge == "api":
        judge_tag = f"_judge-{args.judge_model}"
    elif ds_cfg.get("verifier") == "self_judge" and args.judge_local_model:
        judge_tag = f"_judge-{args.judge_local_model}"
    else:
        judge_tag = ""
    results["judge"] = (
        args.judge_model if args.judge == "api"
        else (args.judge_local_model or "strict-self")
    )
    out_path = h1_dir / f"{args.model}_{args.dataset}{judge_tag}_seed{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2))

    # --- print summary ---------------------------------------------------
    print()
    print(f"=== H1 cell: {experiment}  (M={est.M}, K_max={est.K}) ===")
    print(f"  At K=K_max:")
    print(f"    U_circ_K = {est.U_circ_K:.3f}")
    print(f"    U_bar_K  = {est.U_bar_K:.3f}")
    print(
        f"    R_hat_K  = {est.R_hat_K:.3f}  "
        f"[{ci.ci_low:.3f}, {ci.ci_high:.3f}] (95% bootstrap CI over prompts)"
    )
    print()
    print(f"  Saturation curve  K -> R_hat_K:")
    for row in saturation:
        print(f"    K={row['K']:>4}: R_hat_K = {row['R_hat_K']:.3f}  "
              f"(U_circ={row['U_circ_K']:.3f}, U_bar={row['U_bar_K']:.3f})")
    print()
    if sample_seconds > 0:
        print(f"  sampling: {sample_seconds:.1f} s ({sample_seconds/3600:.3f} GPU-hr)")
    print(f"  results:  {out_path}")
    print(f"  log:      {log_file}")
    print()

    log.info(
        "Done. R_hat_K (at K_max=%d) = %.3f [%.3f, %.3f]",
        args.K, est.R_hat_K, ci.ci_low, ci.ci_high,
    )


if __name__ == "__main__":
    main()
