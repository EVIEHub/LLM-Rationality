"""Re-verify a cached H1 sample file and overwrite the result JSON.

Use case: the original cell on a server ran sampling correctly but lost the
verifier output (e.g. ``math_verify`` lib was missing at verify time, so
every utility was scored 0.0). The samples cache is intact; we just need
to re-run the verifier locally and rewrite the result file with the
correct numbers.

This is a re-verify ONLY — no model loading, no GPU. Calls the same
``src.verification.<dataset>.verify`` function and the same estimators
that ``scripts.run_h1`` would use, so the output matches the schema
``run_h1.py`` writes byte-for-byte (modulo ``sampling_seconds`` which
is preserved from the existing result file, since no new sampling
happened).

Usage:
    python -m scripts.reverify_from_cache \\
        --samples /path/to/v2_math_..._<fingerprint>.jsonl.gz \\
        --result  /path/to/qwen2.5-72b-instruct_math_seed0.json \\
        --dataset math

Writes the new result JSON in place (after backing up the old one to
``<result>.corrupt.bak``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

# Project root on the path so we can import the verifiers + estimators.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.metrics.rational_gap import (
    U_bar_at_K,
    U_circ_at_K,
    bootstrap_ci_over_prompts,
    compute_rational_gap,
)


def _saturation_grid(K_max: int) -> list[int]:
    """Same grid as scripts.run_h1._saturation_grid (powers of 2 + K_max)."""
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= K_max]
    if K_max not in grid:
        grid.append(K_max)
    return sorted(set(grid))


def _load_verify(dataset: str):
    """Import the per-dataset verify function lazily.

    Going through the dispatcher in src.verification.interface would pull in
    every verifier (humaneval needs subprocess, livecodebench needs more,
    etc.), which is overkill here.
    """
    if dataset == "math":
        from src.verification.math import verify as _verify
    elif dataset == "gsm8k":
        from src.verification.gsm8k import verify as _verify
    elif dataset == "matharena":
        from src.verification.matharena import verify as _verify
    elif dataset == "humaneval":
        from src.verification.humaneval import verify as _verify
    elif dataset == "livecodebench":
        from src.verification.livecodebench import verify as _verify
    else:
        raise ValueError(
            f"reverify_from_cache only supports deterministic verifiers; "
            f"got dataset={dataset!r}. For preference datasets re-run "
            f"scripts.run_h1 directly so the judge runs."
        )
    return _verify


def main() -> None:
    p = argparse.ArgumentParser(description="Re-verify a cached H1 sample file.")
    p.add_argument("--samples", required=True, help="Path to v2_<dataset>_<model>_K<K>_<fingerprint>.jsonl.gz")
    p.add_argument("--result", required=True, help="Path to the result JSON to overwrite.")
    p.add_argument("--dataset", required=True, help="Dataset alias (e.g. math, gsm8k).")
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-confidence", type=float, default=0.95)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    args = p.parse_args()

    verify_fn = _load_verify(args.dataset)
    samples_path = Path(args.samples)
    result_path = Path(args.result)
    if not samples_path.is_file():
        sys.exit(f"samples file not found: {samples_path}")
    if not result_path.is_file():
        sys.exit(f"result file not found (need it for metadata): {result_path}")

    # 1) Load existing result so we can preserve metadata (model_hf_id,
    #    cache_key_fingerprint, sampling_seconds, etc.).
    existing = json.loads(result_path.read_text())
    M_existing = existing.get("M")
    K_existing = existing.get("K_max")

    # 2) Stream the samples cache. Header row is metadata; the rest are
    #    per-prompt records with keys {prompt_id, prompt, ground_truth, samples}.
    records: list[dict] = []
    with gzip.open(samples_path, "rt") as f:
        header = json.loads(f.readline())  # {_cache_key, _format_version}
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    M = len(records)
    K = len(records[0]["samples"])
    print(f"loaded {M} prompts × K={K} from {samples_path.name}")
    if M_existing is not None and M != M_existing:
        print(f"  WARNING: M from samples ({M}) != M from existing result ({M_existing})")
    if K_existing is not None and K != K_existing:
        print(f"  WARNING: K from samples ({K}) != K_max from existing result ({K_existing})")

    # 3) Verify every (i, k) sample. Single-process — the verifier is
    #    pure-CPU and fast enough for M=1000, K=64.
    t0 = time.time()
    utility = np.zeros((M, K), dtype=np.float32)
    for i, rec in enumerate(records):
        gt = rec["ground_truth"]
        for k, sample in enumerate(rec["samples"]):
            utility[i, k] = verify_fn(sample, gt)
        if (i + 1) % 100 == 0:
            print(f"  verified {i + 1}/{M} prompts in {time.time() - t0:.1f}s")
    verify_seconds = time.time() - t0
    print(f"verify done in {verify_seconds:.1f}s; mean utility = {utility.mean():.4f}")

    # 4) Estimates at K_max + bootstrap CI + saturation curve.
    est = compute_rational_gap(utility)
    ci = bootstrap_ci_over_prompts(
        est.per_prompt_R_hat_K,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed,
    )
    saturation = []
    for K_prime in _saturation_grid(K):
        saturation.append(
            {
                "K": K_prime,
                "U_circ_K": U_circ_at_K(utility, K_prime),
                "U_bar_K": U_bar_at_K(utility, K_prime),
            }
        )

    # 5) Write back, preserving existing metadata + sampling_seconds.
    new_results = {
        **existing,  # keep experiment, model, model_hf_id, dataset, seed, fingerprint, judge, ...
        "M": int(est.M),
        "K_max": int(est.K),
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
        # Re-verification metadata so the rewrite is traceable.
        "reverify_seconds": verify_seconds,
        "reverify_source_samples": str(samples_path),
    }

    backup_path = result_path.with_suffix(".corrupt.bak")
    shutil.copy2(result_path, backup_path)
    print(f"backed up old result -> {backup_path}")
    result_path.write_text(json.dumps(new_results, indent=2))
    print(f"wrote re-verified result -> {result_path}")
    print()
    print(f"  U_circ_K = {est.U_circ_K:.4f}")
    print(f"  U_bar_K  = {est.U_bar_K:.4f}")
    print(f"  R_hat_K  = {est.R_hat_K:.4f}  95% CI=[{ci.ci_low:.4f}, {ci.ci_high:.4f}]  (B={ci.n_resamples})")


if __name__ == "__main__":
    main()
