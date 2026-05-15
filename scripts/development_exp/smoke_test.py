"""End-to-end smoke test: sampling and verification on a tiny model.

Per AGENT.md §7, "no change is complete until smoke test passes". This
script exercises every layer of the pipeline:

    paths config -> models/datasets configs -> HF dataset load -> chat
    template formatting -> vLLM sampling -> sample cache -> verifier
    loop -> per-decision audit log -> R_hat_K + bootstrap CI -> compute
    log

Defaults (per HANDOFF.md §4): qwen2.5-1.5b-instruct, gsm8k, 10 prompts,
K=4, seed=0. End-to-end wall time should be well under 5 minutes on a
single small GPU once the model is cached.

Usage:
    python -m scripts.smoke_test
    python -m scripts.smoke_test --model qwen2.5-0.5b-instruct --K 2 --num-prompts 5
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from src.metrics.rational_gap import bootstrap_ci_over_prompts, compute_rational_gap
from src.pipeline.cache import CacheKey, cache_exists, read_cache, write_cache
from src.pipeline.logging_utils import log_compute, log_verifier_decision, setup_run_logger
from src.pipeline.paths import load_paths
from src.sampling.vllm_runner import SamplingConfig, VllmRunner
from src.verification.interface import verify as verify_dispatch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _format_chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    """Apply the model's HF chat template to a system+user pair."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _assemble_ground_truth(row: dict[str, Any], ds_cfg: dict[str, Any]) -> str:
    """Build the ground-truth string the verifier expects.

    For HumanEval the dataset's ``test`` field is a ``def check(...)``
    block; we append the ``check(<entry_point>)`` invocation that
    actually runs the assertions. For GSM8K and MATH the ground-truth
    field is taken as-is (the verifiers extract the answer themselves).
    """
    gt = row[ds_cfg["ground_truth_field"]]
    if "entry_point_field" in ds_cfg:
        gt = gt + f"\ncheck({row[ds_cfg['entry_point_field']]})\n"
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Rational-gap smoke test")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--gpu-mem", type=float, default=0.5,
        help="vLLM gpu_memory_utilization (default 0.5)",
    )
    args = parser.parse_args()

    paths = load_paths()
    paths.ensure_dirs()
    experiment = f"smoke_{args.model}_{args.dataset}"
    _root_logger, log_file = setup_run_logger(paths.logs_dir, experiment)
    log = logging.getLogger(__name__)
    log.info(
        "Starting %s | K=%d, num_prompts=%d, seed=%d, max_tokens=%d",
        experiment, args.K, args.num_prompts, args.seed, args.max_tokens,
    )

    models_cfg = _load_yaml(_REPO_ROOT / "configs" / "models.yaml")
    datasets_cfg = _load_yaml(_REPO_ROOT / "configs" / "datasets.yaml")

    if args.model not in models_cfg:
        raise SystemExit(
            f"Unknown model alias {args.model!r}; known: {sorted(models_cfg)}"
        )
    if args.dataset not in datasets_cfg:
        raise SystemExit(
            f"Unknown dataset alias {args.dataset!r}; known: {sorted(datasets_cfg)}"
        )
    model_cfg = models_cfg[args.model]
    ds_cfg = datasets_cfg[args.dataset]

    if model_cfg["prompt_mode"] != "chat":
        raise SystemExit(
            f"Smoke test currently supports chat-mode models only; "
            f"{args.model} is {model_cfg['prompt_mode']!r} (few-shot files "
            f"land with H2 — see HANDOFF.md §6)."
        )

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
        num_prompts=args.num_prompts,
    )
    log.info("Cache key fingerprint: %s", key.fingerprint())

    # --- load + slice the HF dataset --------------------------------------
    ds_kwargs: dict[str, Any] = {
        "path": ds_cfg["hf_id"],
        "split": ds_cfg["split"],
    }
    if ds_cfg.get("hf_config"):
        ds_kwargs["name"] = ds_cfg["hf_config"]
    log.info("Loading dataset: %s", ds_kwargs)
    raw_ds = load_dataset(**ds_kwargs).select(range(args.num_prompts))
    raw_inputs = [dict(row) for row in raw_ds]

    # --- format prompts ---------------------------------------------------
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
    log.info(
        "Formatted %d prompts (preview: %s ...)",
        len(prompts), prompts[0][:120].replace("\n", " | "),
    )

    # --- sample (or load from cache) --------------------------------------
    if cache_exists(paths.samples_dir, key):
        log.info("Cache hit at %s", paths.samples_dir)
        records = list(read_cache(paths.samples_dir, key))
        sample_seconds = 0.0
    else:
        log.info("Cache miss — sampling K=%d completions per prompt", args.K)
        t0 = time.time()
        with VllmRunner(
            model_cfg["hf_id"],
            gpu_memory_utilization=args.gpu_mem,
            enforce_eager=True,
        ) as runner:
            cfg = SamplingConfig(
                temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens,
            )
            samples_list = runner.sample(prompts, K=args.K, seed=args.seed, config=cfg)
        sample_seconds = time.time() - t0
        log.info("Sampling complete in %.1f s", sample_seconds)

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
                "num_prompts": args.num_prompts,
                "K": args.K,
                "seed": args.seed,
                "stage": "sampling",
            },
        )

    # --- verify each (prompt, sample) and log audit decisions -------------
    log.info("Verifying %d (prompt, sample) pairs", len(records) * args.K)
    M = len(records)
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
                    "seed": args.seed,
                },
            )

    # --- aggregate + bootstrap CI -----------------------------------------
    est = compute_rational_gap(utility)
    ci = bootstrap_ci_over_prompts(
        est.per_prompt_R_hat_K,
        n_resamples=1000,
        confidence=0.95,
        seed=args.seed,
    )

    print()
    print(f"=== Smoke test: {experiment}  (M={est.M}, K={est.K}) ===")
    print(f"  U_circ_K = {est.U_circ_K:.3f}")
    print(f"  U_bar_K  = {est.U_bar_K:.3f}")
    print(
        f"  R_hat_K  = {est.R_hat_K:.3f}  "
        f"[{ci.ci_low:.3f}, {ci.ci_high:.3f}] (95% bootstrap CI over prompts)"
    )
    if sample_seconds > 0:
        print(f"  sampling: {sample_seconds:.1f} s")
    print(f"  log:      {log_file}")
    print(f"  cache:    {paths.samples_dir}")
    print()

    log.info(
        "Done. U_circ_K=%.3f, U_bar_K=%.3f, R_hat_K=%.3f",
        est.U_circ_K, est.U_bar_K, est.R_hat_K,
    )


if __name__ == "__main__":
    main()
