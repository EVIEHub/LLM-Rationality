"""H3 cell runner — rational gap of one inference mechanism on one
(model, dataset, seed) cell.

Each H3 cell evaluates ONE inference procedure pi at sampling budget K
on one (model, dataset, seed) tuple. Cell output is the same shape as
H1: aggregates {U_circ_K, U_bar_K, R_hat_K} computed by
``compute_rational_gap`` on K samples drawn from pi, plus a
prompt-bootstrap CI on R_hat_K.

Procedures and their natural budgets:

    --procedure direct --tau 1.0
        Reuses H1's K=64 cache (zero new GPU work).
    --procedure direct --tau 0.0 (greedy)
        Deterministic: K=1 sample is sufficient; R_hat_K=0 by construction.
    --procedure direct --tau 0.7
        Fresh K=64 stochastic sampling required; wall-time comparable
        to an H1 cell at the same K.
    --procedure sc --sc-n N
        Self-consistency over N underlying samples per draw. Reuses the
        H1 K=64 cache via bootstrap-resampled draws; no GPU work.

Output:
    ${samples_dir}/v2_<dataset>_<model>_K<K>_<hash>.jsonl.gz   — sample cache
    ${results_dir}/h3/<model>_<dataset>_t<tau>_seed<S>.json    — direct cell
    ${results_dir}/h3/<model>_<dataset>_sc_n<N>_seed<S>.json   — SC cell

Usage:
    python -m scripts.run_h3 --model tulu3-8b-rlvr --dataset gsm8k \\
        --seed 0 --K 64 --num-prompts 1319 --procedure direct --tau 0.7

    python -m scripts.run_h3 --model tulu3-8b-rlvr --dataset gsm8k \\
        --seed 0 --K 64 --num-prompts 1319 --procedure sc --sc-n 8
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

from src.metrics.h3 import self_consistency_utility_matrix
from src.metrics.rational_gap import (
    bootstrap_ci_over_prompts,
    compute_rational_gap,
)
from src.pipeline.cache import CacheKey, cache_exists, read_cache, write_cache
from src.pipeline.logging_utils import (
    log_compute,
    log_verifier_decision,
    setup_run_logger,
)
from src.pipeline.paths import load_paths
from src.sampling.inference_procedures import direct_sample
from src.sampling.vllm_runner import VllmRunner
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
    gt = row[ds_cfg["ground_truth_field"]]
    if "entry_point_field" in ds_cfg:
        gt = gt + f"\ncheck({row[ds_cfg['entry_point_field']]})\n"
    return gt


def _verify_matrix(
    samples_per_prompt: list[list[str]],
    ground_truths: list[str],
    dataset: str,
    *,
    paths: Any,
    log_tag: str,
    seed: int,
) -> np.ndarray:
    """Return (M, K) utility array; logs every (prompt, k) verifier decision.

    Parallelised across threads — same rationale as scripts/run_h1.py.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    M = len(samples_per_prompt)
    K = len(samples_per_prompt[0]) if M else 0
    util = np.zeros((M, K), dtype=float)
    if M == 0:
        return util

    tasks = [
        (i, k, sample, ground_truths[i])
        for i, samples in enumerate(samples_per_prompt)
        for k, sample in enumerate(samples)
    ]

    def _verify_task(t):
        i, k, sample, gt = t
        return i, k, verify_dispatch(dataset, sample, gt)

    n_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, k, u in pool.map(_verify_task, tasks):
            util[i, k] = u
            log_verifier_decision(
                paths.logs_dir,
                dataset,
                {
                    "experiment": log_tag,
                    "prompt_id": f"{dataset}_{i}",
                    "k": k,
                    "utility": u,
                    "seed": seed,
                },
            )
    return util


def _saturation_grid(K_max: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= K_max]
    if K_max not in grid:
        grid.append(K_max)
    return sorted(set(grid))


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 cell runner")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset", required=True,
        choices=["gsm8k", "math", "humaneval", "matharena", "livecodebench"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=64,
                        help="Sampling budget per prompt for the chosen procedure.")
    parser.add_argument("--procedure", choices=["direct", "sc"], default="direct",
                        help="Inference procedure pi. 'direct' samples at fixed "
                             "tau (--tau required). 'sc' is self-consistency over "
                             "N underlying tau=1 samples (--sc-n required).")
    parser.add_argument("--tau", type=float, default=None,
                        help="For --procedure direct: temperature defining pi. "
                             "Use 0 for greedy (R_hat_K = 0 by construction). "
                             "1.0 reuses the H1 cache for this cell.")
    parser.add_argument("--sc-n", type=int, default=None,
                        help="For --procedure sc: number of underlying samples per "
                             "SC draw (the 'n' in SC(n)). Each H3 SC cell builds "
                             "K bootstrap draws from the H1 K=64 cache.")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    if args.procedure == "direct":
        if args.tau is None:
            raise SystemExit("--procedure direct requires --tau")
        if args.tau < 0:
            raise SystemExit(f"--tau must be >= 0, got {args.tau}")
    elif args.procedure == "sc":
        if args.sc_n is None or args.sc_n < 1:
            raise SystemExit("--procedure sc requires --sc-n >= 1")

    paths = load_paths()
    paths.ensure_dirs()
    h3_dir = paths.results_dir / "h3"
    h3_dir.mkdir(parents=True, exist_ok=True)

    if args.procedure == "direct":
        proc_tag = f"t{args.tau}"
    else:
        proc_tag = f"sc_n{args.sc_n}"
    experiment = f"h3_{args.model}_{args.dataset}_{proc_tag}_seed{args.seed}"
    _root, log_file = setup_run_logger(paths.logs_dir, experiment)
    log = logging.getLogger(__name__)
    log.info(
        "Starting %s | procedure=%s K=%d num_prompts=%s seed=%d",
        experiment, args.procedure, args.K, args.num_prompts, args.seed,
    )

    models_cfg = _load_yaml(_REPO_ROOT / "configs" / "models.yaml")
    datasets_cfg = _load_yaml(_REPO_ROOT / "configs" / "datasets.yaml")
    if args.model not in models_cfg:
        raise SystemExit(f"Unknown model alias {args.model!r}; known: {sorted(models_cfg)}")
    model_cfg = models_cfg[args.model]
    ds_cfg = datasets_cfg[args.dataset]
    if model_cfg["prompt_mode"] != "chat":
        raise SystemExit(
            f"H3 (this MVP) is chat-mode only; {args.model} is {model_cfg['prompt_mode']!r}."
        )

    # --- load + slice dataset --------------------------------------------
    if "subsets" in ds_cfg:
        from datasets import Value, concatenate_datasets
        parts = []
        for subset in ds_cfg["subsets"]:
            if ds_cfg.get("hf_id"):
                part = load_dataset(ds_cfg["hf_id"], subset, split=ds_cfg["split"])
            else:
                part = load_dataset(subset, split=ds_cfg["split"])
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
    else:
        ds_kwargs: dict[str, Any] = {"path": ds_cfg["hf_id"], "split": ds_cfg["split"]}
        if ds_cfg.get("hf_config"):
            ds_kwargs["name"] = ds_cfg["hf_config"]
        raw_ds = load_dataset(**ds_kwargs)
    min_date = ds_cfg.get("min_contest_date")
    if min_date is not None:
        from datetime import datetime
        cutoff = datetime.fromisoformat(min_date)
        raw_ds = raw_ds.filter(
            lambda r: r.get("contest_date") is not None
                      and r["contest_date"] >= cutoff,
        )
    if args.num_prompts is not None:
        raw_ds = raw_ds.select(range(min(args.num_prompts, len(raw_ds))))
    raw_inputs = [dict(row) for row in raw_ds]
    log.info("Using %d prompts", len(raw_inputs))

    # --- format prompts --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"])
    chat_tmpl = ds_cfg["templates"]["chat"]
    prompt_field = ds_cfg["prompt_field"]
    prompts: list[str] = []
    ground_truths: list[str] = []
    for row in raw_inputs:
        user_text = chat_tmpl["user_template"].format(**{prompt_field: row[prompt_field]})
        prompts.append(_format_chat_prompt(tokenizer, chat_tmpl["system"], user_text))
        ground_truths.append(_assemble_ground_truth(row, ds_cfg))

    # ---------------------------------------------------------------------
    # Build the (M, K_effective) utility matrix for the chosen procedure.
    # ---------------------------------------------------------------------
    if args.procedure == "direct":
        # Greedy is deterministic: K=1 is the natural budget; storing K>1
        # of identical samples would just inflate the cache.
        K_effective = 1 if args.tau == 0.0 else args.K
        key = CacheKey(
            model=model_cfg["hf_id"],
            dataset=args.dataset,
            K=K_effective,
            temperature=args.tau,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            seed=args.seed,
            prompt_template_version=ds_cfg["prompt_template_version"],
            num_prompts=len(raw_inputs),
        )
        log.info("Cache key fingerprint: %s", key.fingerprint())

        if cache_exists(paths.samples_dir, key):
            log.info("Cache hit; reading samples")
            records = list(read_cache(paths.samples_dir, key))
            sample_seconds = 0.0
        else:
            log.info("Cache miss — sampling K=%d at tau=%s", K_effective, args.tau)
            t0 = time.time()
            with VllmRunner(
                model_cfg["hf_id"],
                gpu_memory_utilization=args.gpu_mem,
                enforce_eager=True,
            ) as runner:
                samples_list = direct_sample(
                    runner, prompts, K=K_effective, seed=args.seed,
                    temperature=args.tau, max_tokens=args.max_tokens,
                )
            sample_seconds = time.time() - t0
            log.info(
                "Sampling done in %.1f s (%.2f GPU-hr)",
                sample_seconds, sample_seconds / 3600,
            )
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
                    "model": args.model, "dataset": args.dataset,
                    "num_prompts": len(prompts), "K": K_effective, "tau": args.tau,
                    "seed": args.seed, "stage": "h3_sampling",
                },
            )

        samples_per_prompt = [rec["samples"] for rec in records]
        util = _verify_matrix(
            samples_per_prompt, ground_truths, args.dataset,
            paths=paths, log_tag=experiment, seed=args.seed,
        )

    else:
        # SC procedure. Read the H1 K=64 cache (must exist), verify it
        # to get util_h1, then build the SC utility matrix as K bootstrap
        # draws of n underlying samples.
        K_effective = args.K
        h1_key = CacheKey(
            model=model_cfg["hf_id"],
            dataset=args.dataset,
            K=64,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            seed=args.seed,
            prompt_template_version=ds_cfg["prompt_template_version"],
            num_prompts=len(raw_inputs),
        )
        if not cache_exists(paths.samples_dir, h1_key):
            raise SystemExit(
                f"SC needs the H1 K=64 tau=1 cache for this cell. Missing "
                f"fingerprint: {h1_key.fingerprint()}. Run scripts/run_h1 first."
            )
        h1_records = list(read_cache(paths.samples_dir, h1_key))
        samples_per_prompt = [rec["samples"] for rec in h1_records]
        log.info("H1 cache hit: %d prompts x K=%d samples", len(h1_records), 64)
        util_h1 = _verify_matrix(
            samples_per_prompt, ground_truths, args.dataset,
            paths=paths, log_tag=f"{experiment}_h1verify", seed=args.seed,
        )
        log.info(
            "Building SC(%d) utility matrix: K=%d bootstrap draws per prompt",
            args.sc_n, K_effective,
        )
        util = self_consistency_utility_matrix(
            util_h1, samples_per_prompt, args.dataset,
            n=args.sc_n, K=K_effective, seed=args.seed,
        )

    # --- compute the H1-style triple via compute_rational_gap -----------
    est = compute_rational_gap(util)
    ci = bootstrap_ci_over_prompts(
        est.per_prompt_R_hat_K, n_resamples=1000, confidence=0.95,
        seed=args.seed,
    )
    log.info(
        "Aggregates@K=%d %s: U_circ_K=%.3f U_bar_K=%.3f R_hat_K=%.3f",
        K_effective, proc_tag, est.U_circ_K, est.U_bar_K, est.R_hat_K,
    )

    # Saturation curve over K' <= K_effective. For greedy K=1 this is
    # just the K=1 row (R_hat_K=0).
    K_grid = _saturation_grid(K_effective)
    saturation = []
    from src.metrics.rational_gap import (
        R_hat_at_K, U_bar_at_K, U_circ_at_K,
    )
    for k_prime in K_grid:
        saturation.append({
            "K": k_prime,
            "U_circ_K": U_circ_at_K(util, k_prime),
            "U_bar_K": U_bar_at_K(util, k_prime),
            "R_hat_K": R_hat_at_K(util, k_prime),
        })

    # --- write per-cell result ------------------------------------------
    out_path = h3_dir / f"{args.model}_{args.dataset}_{proc_tag}_seed{args.seed}.json"
    payload: dict[str, Any] = {
        "experiment": experiment,
        "model": args.model,
        "model_hf_id": model_cfg["hf_id"],
        "dataset": args.dataset,
        "seed": args.seed,
        "K": K_effective,
        "procedure": args.procedure,
        "M": est.M,
        "aggregates_at_K_max": {
            "U_circ_K": est.U_circ_K,
            "U_bar_K": est.U_bar_K,
            "R_hat_K": est.R_hat_K,
        },
        "saturation_curve": saturation,
        "bootstrap_R_hat_K_at_K_max": {
            "mean": ci.mean,
            "ci_low": ci.ci_low,
            "ci_high": ci.ci_high,
            "confidence": ci.confidence,
            "n_resamples": ci.n_resamples,
        },
        "log_file": str(log_file),
    }
    if args.procedure == "direct":
        payload["tau"] = args.tau
    else:
        payload["sc_n"] = args.sc_n
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
