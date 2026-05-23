"""H2 cell runner — Tulu-3 trajectory (independence from value alignment).

Per HANDOFF.md §6: alignment does NOT eliminate the rational gap. Along
the SFT → DPO → RLVR trajectory $\\bar{U}_K$ should rise (alignment
working), while $U^\\circ_K$ may stagnate or decline (distributional
sharpening). H2 reports both terms separately, not just the gap.

Per ``methodology/hypotheses.md``:
- 4 models: Tulu-3-8B base (= Llama-3.1-8B), SFT, DPO, RLVR.
- 2 datasets: gsm8k, math (HumanEval excluded — Tulu trajectory is
  optimised for math/reasoning, not code; HumanEval would conflate
  alignment effects with code-generation training).
- Same K_max=64 as H1 so cells are commensurable.

Prompt mode is read from ``configs/models.yaml`` per model:
- ``chat`` — apply tokenizer.apply_chat_template (SFT/DPO/RLVR have
  Tulu-3's chat template).
- ``few_shot`` — prepend the contents of
  ``configs/few_shot/<dataset>_<N>shot.txt`` to the bare user prompt.
  Used for the base model (Llama-3.1-8B has no chat template).

Cells produce:
    ${samples_dir}/v2_<dataset>_<model>_K<K>_<hash>.jsonl.gz   — samples cache
    ${results_dir}/h2/<model_alias>_<dataset>_seed<S>.json     — per-cell results

For Tulu-3-RLVR cells (chat mode), the cache key fingerprint MATCHES
the H1 fingerprint, so the H1 sample cache is reused — no duplicate
sampling. Only base/SFT/DPO sample fresh.

Usage:
    python -m scripts.run_h2 --model tulu3-8b-base --dataset gsm8k --seed 0 --K 64
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
    """Apply HF chat template to system+user pair."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _format_few_shot_prompt(examples_text: str, user: str) -> str:
    """Concatenate few-shot examples with the test prompt.

    The examples file contains complete worked examples ending with
    a clear answer marker (e.g. "The answer is X." or "\\boxed{X}").
    The model continues from "\\nAnswer:" or "\\nSolution:" with its own
    reasoning + final answer in the same format.
    """
    return examples_text.rstrip() + "\n\n" + user


def _assemble_ground_truth(row: dict[str, Any], ds_cfg: dict[str, Any]) -> str:
    if ds_cfg.get("verifier") == "self_judge":
        chosen = row[ds_cfg["ground_truth_field"]]
        if isinstance(chosen, list):
            for msg in chosen:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""
        return str(chosen)
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


def _saturation_grid(K_max: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= K_max]
    if K_max not in grid:
        grid.append(K_max)
    return sorted(set(grid))


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 cell runner")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset", required=True,
        choices=["gsm8k", "math", "humaneval", "matharena", "livecodebench", "ultrafeedback", "alpaca_eval"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument(
        "--api-concurrency", type=int, default=20,
        help="Max in-flight API judge requests when --judge api.",
    )
    parser.add_argument(
        "--judge", choices=["self", "api"], default="self",
        help="Preference judge backend: 'self' (fixed vLLM judge) or 'api' (hosted).",
    )
    parser.add_argument(
        "--judge-model", default="deepseek-v4-flash",
        help="models.yaml alias of the API judge when --judge api.",
    )
    parser.add_argument(
        "--judge-local-model", default=None,
        help="models.yaml alias of a fixed local vLLM judge for --judge self, "
             "overriding the default fixed Tulu-3-RLVR judge (e.g. "
             "qwen2.5-14b-instruct for an independent judge-robustness re-run).",
    )
    parser.add_argument(
        "--judge-L", type=int, default=None,
        help="Override judge calls per pair (default: ds_cfg.judge_L, or 3 for API).",
    )
    args = parser.parse_args()

    paths = load_paths()
    paths.ensure_dirs()
    h2_dir = paths.results_dir / "h2"
    h2_dir.mkdir(parents=True, exist_ok=True)

    experiment = f"h2_{args.model}_{args.dataset}_seed{args.seed}"
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

    prompt_mode = model_cfg["prompt_mode"]
    if prompt_mode not in ("chat", "few_shot"):
        raise SystemExit(f"Unknown prompt_mode {prompt_mode!r} for {args.model}")
    log.info("Model %s uses prompt_mode=%s", args.model, prompt_mode)

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
    elif ds_cfg.get("hf_data_file"):
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
    # Optional date filter (LiveCodeBench: keep only post-cutoff items).
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

    # --- format prompts based on prompt_mode -----------------------------
    log.info("Loading tokenizer for %s", model_cfg["hf_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"])

    prompt_field = ds_cfg["prompt_field"]
    prompts: list[str] = []
    raw_user_prompts: list[str] = []   # raw user question, used by self_judge verifier
    ground_truths: list[str] = []

    if prompt_mode == "chat":
        chat_tmpl = ds_cfg["templates"]["chat"]
        for row in raw_inputs:
            raw_q = row[prompt_field]
            if isinstance(raw_q, (bytes, bytearray)):
                raw_q = raw_q.decode("utf-8")
            raw_user_prompts.append(str(raw_q))
            user_text = chat_tmpl["user_template"].format(**{prompt_field: raw_q})
            prompts.append(_format_chat_prompt(tokenizer, chat_tmpl["system"], user_text))
            ground_truths.append(_assemble_ground_truth(row, ds_cfg))
    else:  # few_shot
        fs_tmpl = ds_cfg["templates"]["few_shot"]
        examples_path = _REPO_ROOT / fs_tmpl["examples_path"]
        if not examples_path.exists():
            raise SystemExit(
                f"Few-shot examples file not found: {examples_path}\n"
                f"Author it before running H2 with a few-shot model."
            )
        examples_text = examples_path.read_text()
        log.info(
            "Few-shot examples loaded from %s (%d chars)",
            examples_path, len(examples_text),
        )
        user_template = fs_tmpl["user_template"]
        for row in raw_inputs:
            raw_q = row[prompt_field]
            if isinstance(raw_q, (bytes, bytearray)):
                raw_q = raw_q.decode("utf-8")
            raw_user_prompts.append(str(raw_q))
            user_text = user_template.format(**{prompt_field: raw_q})
            prompts.append(_format_few_shot_prompt(examples_text, user_text))
            ground_truths.append(_assemble_ground_truth(row, ds_cfg))

    log.info(
        "Formatted %d prompts (preview: %s ...)",
        len(prompts), prompts[0][:120].replace("\n", " | "),
    )

    # --- sample (or load from cache) -------------------------------------
    if cache_exists(paths.samples_dir, key):
        log.info("Cache HIT — reading samples (no GPU work for this cell)")
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
        log.info("Sampling done in %.1f s (%.2f GPU-hr)", sample_seconds, sample_seconds / 3600)

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
                "prompt_mode": prompt_mode,
                "stage": "sampling",
            },
        )

    # --- verify ---------------------------------------------------------
    # Two paths — preference mode (self-judge with FIXED Tulu-3-RLVR
    # judge across all trajectory stages) vs verifier-binary.
    #
    # H2 deliberately uses a FIXED judge across SFT/DPO/RLVR rather than
    # strict-self per stage: SFT models are weak at A/B parsing (refusal,
    # verbose tangents, etc.) and would produce unreliable verdicts. By
    # judging all three stages with the post-alignment Tulu-3-RLVR model
    # we measure "the same judge's opinion across the trajectory", which
    # is the scientific claim we can actually defend.
    M = len(records)
    samples_per_prompt = [rec["samples"] for rec in records]

    if ds_cfg.get("verifier") == "self_judge" and args.judge == "api":
        # --- preference mode: hosted API judge (no GPU), fixed across stages ---
        from src.sampling.api_runner import ApiModelSpec
        from src.verification.api_judge import score_matrix_api
        judge_cfg = models_cfg[args.judge_model]
        spec = ApiModelSpec.from_model_cfg(judge_cfg)
        L_judge = int(args.judge_L or ds_cfg.get("judge_L", 3))
        resume = paths.samples_dir / f"apijudge_{key.fingerprint()}_{args.judge_model}.jsonl"
        log.info(
            "Preference mode (API judge=%s, fixed across stages): %d * %d * L=%d calls",
            args.judge_model, M, args.K, L_judge,
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
                        "judge": args.judge_model,
                        "seed": args.seed,
                    },
                )
    elif ds_cfg.get("verifier") == "self_judge":
        from src.verification.self_judge import score_matrix
        L_judge = int(args.judge_L or ds_cfg.get("judge_L", 5))
        # FIXED judge — default Tulu-3-RLVR (the post-alignment checkpoint),
        # held constant across SFT/DPO/RLVR stages. --judge-local-model
        # overrides it with any models.yaml alias (e.g. an independent
        # Qwen2.5-14B for a judge-robustness re-run).
        # NOTE: 8B RLVR default; for 70B self-judge this would need the 70B
        # RLVR. Server B uses --judge api, so this path is unused there.
        if args.judge_local_model:
            judge_hf_id = models_cfg[args.judge_local_model]["hf_id"]
        else:
            judge_hf_id = "allenai/Llama-3.1-Tulu-3-8B"
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_hf_id)
        log.info(
            "Preference mode: %d * %d * L=%d = %d judge calls; "
            "FIXED judge=%s (not generator=%s)",
            M, args.K, L_judge, M * args.K * L_judge,
            judge_hf_id, model_cfg["hf_id"],
        )
        with VllmRunner(
            judge_hf_id,
            gpu_memory_utilization=args.gpu_mem,
            enforce_eager=True,
        ) as judge_runner:
            outcome = score_matrix(
                judge_runner=judge_runner,
                judge_tokenizer=judge_tokenizer,
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
                        "seed": args.seed,
                    },
                )

    else:
        # --- verifier-binary mode (existing path) ---
        # Parallelised across threads — same rationale as scripts/run_h1.py.
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
        "hypothesis": "h2",
        "model": args.model,
        "model_hf_id": model_cfg["hf_id"],
        "trajectory_stage": model_cfg.get("trajectory_stage"),
        "prompt_mode": prompt_mode,
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
    # Tag the filename with the judge when it is not the default (fixed RLVR),
    # so an independent-judge re-run sits alongside the original instead of
    # overwriting it (needed for the judge-robustness comparison).
    if ds_cfg.get("verifier") == "self_judge" and args.judge == "api":
        judge_tag = f"_judge-{args.judge_model}"
    elif ds_cfg.get("verifier") == "self_judge" and args.judge_local_model:
        judge_tag = f"_judge-{args.judge_local_model}"
    else:
        judge_tag = ""
    out_path = h2_dir / f"{args.model}_{args.dataset}{judge_tag}_seed{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2))

    # --- print summary ---------------------------------------------------
    print()
    print(
        f"=== H2 cell: {experiment}  (M={est.M}, K_max={est.K}, "
        f"stage={model_cfg.get('trajectory_stage')}, mode={prompt_mode}) ==="
    )
    print(f"  At K=K_max:")
    print(f"    U_circ_K = {est.U_circ_K:.3f}")
    print(f"    U_bar_K  = {est.U_bar_K:.3f}")
    print(
        f"    R_hat_K  = {est.R_hat_K:.3f}  "
        f"[{ci.ci_low:.3f}, {ci.ci_high:.3f}] (95% bootstrap CI)"
    )
    print(f"\n  Saturation  K -> R_hat_K:")
    for row in saturation:
        print(f"    K={row['K']:>4}: R_hat_K={row['R_hat_K']:.3f}  "
              f"(U_circ={row['U_circ_K']:.3f}, U_bar={row['U_bar_K']:.3f})")
    if sample_seconds > 0:
        print(f"\n  sampling: {sample_seconds:.1f} s ({sample_seconds/3600:.3f} GPU-hr)")
    else:
        print("\n  (cache hit — no GPU work)")
    print(f"  results:  {out_path}")
    print()

    log.info("Done. R_hat_K (at K=%d) = %.3f", args.K, est.R_hat_K)


if __name__ == "__main__":
    main()
