"""H4 cell runner — relationship between rational gap and reasoning length.

Per HANDOFF.md §8 + ``methodology/hypotheses.md``: $\\hat{\\mathcal{R}}_K(L)$
as a function of the maximum reasoning length $L$. Implementation is
two-stage budget forcing (s1-style) — stage 1 caps reasoning at $L$
tokens, stage 2 forces an answer commit. The relationship may be
non-monotonic (the paper's expected shape).

Cell:
    1 model (Tulu-3-RLVR, fixed) × 2 datasets (gsm8k, math) × 7 length
    values × seed=0 = 14 cells. HumanEval is excluded.

Per-cell wall time scales roughly with $L + 64$ (stage 2 generates ~64
answer tokens). With ``num-prompts`` set conservatively (default 500)
the longest cell (gsm8k @ L=2048) is ~50 min on a single A800.

Usage:
    python -m scripts.run_h4 --dataset gsm8k --L 256 --num-prompts 500

The model is hardcoded to ``tulu3-8b-rlvr`` because H4 is a
single-model length sweep; if you ever want to vary the model, add
``--model`` and rebuild the cache for the new model.
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
from src.sampling.inference_procedures import budget_forced
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
    gt = row[ds_cfg["ground_truth_field"]]
    if "entry_point_field" in ds_cfg:
        gt = gt + f"\ncheck({row[ds_cfg['entry_point_field']]})\n"
    return gt


def _saturation_grid(K_max: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= K_max]
    if K_max not in grid:
        grid.append(K_max)
    return sorted(set(grid))


def main() -> None:
    parser = argparse.ArgumentParser(description="H4 cell runner")
    parser.add_argument("--model", default="tulu3-8b-rlvr",
                        help="H4 default is Tulu-3-RLVR; rebuild cache if changed.")
    parser.add_argument(
        "--dataset", required=True, choices=["gsm8k", "math", "matharena", "bbh"],
        help="HumanEval / LiveCodeBench excluded — budget forcing needs a "
             "short verifiable answer. Math sets (gsm8k/math/matharena) plus "
             "bbh (BIG-Bench Hard: diverse non-math reasoning with short "
             "MC/boolean/count answers) qualify.",
    )
    parser.add_argument("--L", type=int, required=True,
                        help="max_reasoning_length (0, 64, 128, 256, 512, 1024, 2048).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--answer-max-tokens", type=int, default=64)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    paths = load_paths()
    paths.ensure_dirs()
    h4_dir = paths.results_dir / "h4"
    h4_dir.mkdir(parents=True, exist_ok=True)

    experiment = f"h4_{args.model}_{args.dataset}_L{args.L}_seed{args.seed}"
    _root, log_file = setup_run_logger(paths.logs_dir, experiment)
    log = logging.getLogger(__name__)
    log.info(
        "Starting %s | L=%d, K=%d, num_prompts=%d, seed=%d",
        experiment, args.L, args.K, args.num_prompts, args.seed,
    )

    models_cfg = _load_yaml(_REPO_ROOT / "configs" / "models.yaml")
    datasets_cfg = _load_yaml(_REPO_ROOT / "configs" / "datasets.yaml")

    if args.model not in models_cfg:
        raise SystemExit(f"Unknown model alias {args.model!r}; known: {sorted(models_cfg)}")
    model_cfg = models_cfg[args.model]
    ds_cfg = datasets_cfg[args.dataset]
    if model_cfg["prompt_mode"] != "chat":
        raise SystemExit(
            f"H4 currently runs only chat-mode models (default Tulu-3-RLVR). "
            f"{args.model} has prompt_mode={model_cfg['prompt_mode']!r}."
        )

    # The cache key carries max_reasoning_length so different L values
    # do NOT collide. max_tokens here records the *total* token budget
    # (L + answer_max_tokens) so two runs with different answer-token
    # caps stay distinct too.
    total_max_tokens = args.L + args.answer_max_tokens

    # --- load + slice dataset --------------------------------------------
    if "subsets" in ds_cfg:
        # Multi-source dataset (e.g. MathArena = AIME 2025 + BRUMO 2025).
        from datasets import Value, concatenate_datasets
        parts = []
        for subset in ds_cfg["subsets"]:
            if ds_cfg.get("hf_id"):
                part = load_dataset(ds_cfg["hf_id"], subset, split=ds_cfg["split"])
            else:
                part = load_dataset(subset, split=ds_cfg["split"])
            # Cast string-typed fields to a common type so concat works
            # when subsets infer different types (e.g. AIME's int answer
            # vs BRUMO's string answer).
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
        log.info("Loading dataset: %s", ds_kwargs)
        raw_ds = load_dataset(**ds_kwargs)
    raw_ds = raw_ds.select(range(min(args.num_prompts, len(raw_ds))))
    raw_inputs = [dict(row) for row in raw_ds]
    log.info("Using %d prompts", len(raw_inputs))

    key = CacheKey(
        model=model_cfg["hf_id"],
        dataset=args.dataset,
        K=args.K,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=total_max_tokens,
        seed=args.seed,
        prompt_template_version=ds_cfg["prompt_template_version"],
        num_prompts=len(raw_inputs),
        max_reasoning_length=args.L,
    )
    log.info("Cache key fingerprint: %s (L=%d)", key.fingerprint(), args.L)

    # --- format prompts (chat-mode) --------------------------------------
    log.info("Loading tokenizer for %s", model_cfg["hf_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"])
    chat_tmpl = ds_cfg["templates"]["chat"]
    prompt_field = ds_cfg["prompt_field"]

    prompts: list[str] = []
    ground_truths: list[str] = []
    for row in raw_inputs:
        user_text = chat_tmpl["user_template"].format(**{prompt_field: row[prompt_field]})
        prompts.append(_format_chat_prompt(tokenizer, chat_tmpl["system"], user_text))
        ground_truths.append(_assemble_ground_truth(row, ds_cfg))

    # --- sample (or load from cache) -------------------------------------
    if cache_exists(paths.samples_dir, key):
        log.info("Cache HIT — reading samples")
        records = list(read_cache(paths.samples_dir, key))
        sample_seconds = 0.0
    else:
        log.info("Cache miss — running budget_forced (L=%d)", args.L)
        t0 = time.time()
        with VllmRunner(
            model_cfg["hf_id"],
            gpu_memory_utilization=args.gpu_mem,
            enforce_eager=True,
        ) as runner:
            samples_list = budget_forced(
                runner,
                prompts,
                K=args.K,
                seed=args.seed,
                max_reasoning_length=args.L,
                answer_max_tokens=args.answer_max_tokens,
            )
        sample_seconds = time.time() - t0
        log.info("Sampling done in %.1f s (%.2f GPU-hr)", sample_seconds, sample_seconds / 3600)

        records = [
            {
                "prompt_id": f"{args.dataset}_L{args.L}_{i}",
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
                "L": args.L,
                "num_prompts": len(prompts),
                "K": args.K,
                "seed": args.seed,
                "stage": "budget_forced_sampling",
            },
        )

    # --- verify ----------------------------------------------------------
    M = len(records)
    log.info("Verifying %d (prompt, sample) pairs", M * args.K)
    utility = np.zeros((M, args.K), dtype=float)
    for i, rec in enumerate(records):
        for k, sample in enumerate(rec["samples"]):
            u = verify_dispatch(args.dataset, sample, rec["ground_truth"])
            utility[i, k] = u
            log_verifier_decision(
                paths.logs_dir,
                args.dataset,
                {
                    "experiment": experiment,
                    "prompt_id": rec["prompt_id"],
                    "k": k,
                    "utility": u,
                    "L": args.L,
                    "seed": args.seed,
                },
            )

    # --- aggregate + saturation curve + bootstrap CI ---------------------
    est = compute_rational_gap(utility)
    ci = bootstrap_ci_over_prompts(
        est.per_prompt_R_hat_K, n_resamples=1000, confidence=0.95, seed=args.seed,
    )
    K_grid = _saturation_grid(args.K)
    saturation = [
        {
            "K": K_prime,
            "U_circ_K": U_circ_at_K(utility, K_prime),
            "U_bar_K": U_bar_at_K(utility, K_prime),
            "R_hat_K": R_hat_at_K(utility, K_prime),
        }
        for K_prime in K_grid
    ]

    results = {
        "experiment": experiment,
        "hypothesis": "h4",
        "model": args.model,
        "model_hf_id": model_cfg["hf_id"],
        "dataset": args.dataset,
        "L": args.L,
        "answer_max_tokens": args.answer_max_tokens,
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
    out_path = h4_dir / f"{args.model}_{args.dataset}_L{args.L}_seed{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print()
    print(f"=== H4 cell: {experiment}  (M={est.M}, K_max={est.K}, L={args.L}) ===")
    print(f"  At K=K_max:")
    print(f"    U_circ_K = {est.U_circ_K:.3f}")
    print(f"    U_bar_K  = {est.U_bar_K:.3f}")
    print(
        f"    R_hat_K  = {est.R_hat_K:.3f}  "
        f"[{ci.ci_low:.3f}, {ci.ci_high:.3f}] (95% bootstrap CI)"
    )
    if sample_seconds > 0:
        print(f"\n  sampling: {sample_seconds:.1f} s ({sample_seconds/3600:.3f} GPU-hr)")
    print(f"  results:  {out_path}")
    print()

    log.info(
        "Done. L=%d  R_hat_K=%.3f  U_circ_K=%.3f  U_bar_K=%.3f",
        args.L, est.R_hat_K, est.U_circ_K, est.U_bar_K,
    )


if __name__ == "__main__":
    main()
