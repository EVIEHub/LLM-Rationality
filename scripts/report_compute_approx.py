"""Reproducible compute-approximation report — $A_K$ and the $p_x$ spike.

For every result JSON under ``${results_dir}/<hyp>/``, derive the
compute-approximation error of Lemma 2,

    A_K = R - R_K = 1 - U_circ_K        (ground-truth R = 1),

directly from the cell's saved ``saturation_curve`` (so this needs no GPU,
no re-sampling, and no dataset download — it is pure post-processing of the
committed results). Also reports:

    floor = A_{K_max}                    estimate of Pr(p_x = 0), the
                                         unreachable mass / irreducible floor;
    spike = mean(per_prompt_p_hat == 0)  per-prompt check of the same floor.

With ``--labels FILE.json`` (a JSON mapping ``"<hyp>/<model>_<dataset>"`` ->
list of per-prompt bucket strings, prompt order matching the cell's
``per_prompt_*`` arrays) the floor/spike are additionally broken down by
difficulty bucket — the MATH-level / LiveCodeBench-difficulty analysis.

Outputs a tidy CSV (one row per (cell, K) for the curve, plus floor rows)
and prints a compact summary.

Usage:
    python -m scripts.report_compute_approx                       # all hyps
    python -m scripts.report_compute_approx --hyps h1 h5
    python -m scripts.report_compute_approx --results-dir DIR --out report.csv
    python -m scripts.report_compute_approx --labels difficulty_labels.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _cells(results_dir: Path, hyp: str, seed: int) -> list[dict[str, Any]]:
    d = results_dir / hyp
    out = []
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if rec.get("seed", 0) != seed:
            continue
        rec["_hyp"] = hyp
        rec["_stem"] = f.stem
        out.append(rec)
    return out


def _cell_id(rec: dict[str, Any]) -> str:
    parts = [str(rec.get("model", "?")), str(rec.get("dataset", "?"))]
    for k in ("trajectory_stage", "tau", "sc_n", "L"):
        if rec.get(k) is not None:
            parts.append(f"{k}={rec[k]}")
    return " | ".join(parts)


def _spike_by_bucket(p_hat: list[float], buckets: list[str] | None):
    """Return {bucket: (n, frac p_hat==0)}; '__all__' always present."""
    out: dict[str, tuple[int, float]] = {}
    n = len(p_hat)
    zeros = sum(1 for v in p_hat if v == 0.0)
    out["__all__"] = (n, zeros / n if n else 0.0)
    if buckets and len(buckets) == n:
        for b in sorted(set(buckets)):
            idx = [i for i in range(n) if buckets[i] == b]
            z = sum(1 for i in idx if p_hat[i] == 0.0)
            out[b] = (len(idx), z / len(idx) if idx else 0.0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compute-approximation (A_K) report.")
    p.add_argument("--results-dir", default=None,
                   help="Dir with h1/ h2/ ... subdirs. Defaults to load_paths().results_dir.")
    p.add_argument("--hyps", nargs="+", default=["h1", "h2", "h3", "h4", "h5"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(_REPO_ROOT / "outputs" / "compute_approx_report.csv"))
    p.add_argument("--labels", default=None,
                   help="Optional JSON: {'<hyp>/<stem>': [bucket per prompt]} for "
                        "difficulty breakdown of the floor/spike.")
    args = p.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser()
    else:
        from src.pipeline.paths import load_paths
        results_dir = load_paths().results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"results dir not found: {results_dir}")

    labels = {}
    if args.labels:
        labels = json.loads(Path(args.labels).read_text())

    rows: list[dict[str, Any]] = []
    print(f"{'cell':52}{'K_max':>6}{'A_Kmax(floor)':>15}{'spike pp':>10}")
    for hyp in args.hyps:
        # h5 reuses the h1 directory (API models live there)
        src_hyp = "h1" if hyp == "h5" else hyp
        for rec in _cells(results_dir, src_hyp, args.seed):
            cid = _cell_id(rec)
            sat = rec.get("saturation_curve", [])
            if not sat:
                continue
            K_max = max(r["K"] for r in sat)
            # A_K curve straight from U_circ_K (A_K = 1 - U_circ_K)
            for r in sat:
                A = 1.0 - r["U_circ_K"]
                rows.append({
                    "hypothesis": hyp, "cell": cid, "stem": rec["_stem"],
                    "K": r["K"], "U_circ_K": round(r["U_circ_K"], 6),
                    "A_K": round(A, 6), "bucket": "__all__",
                })
            floor = 1.0 - next(r["U_circ_K"] for r in sat if r["K"] == K_max)
            # per-prompt spike (and optional difficulty breakdown)
            pp = rec.get("per_prompt_U_bar_K")
            spike_all = float("nan")
            if pp is not None:
                key = f"{src_hyp}/{rec['_stem']}"
                by = _spike_by_bucket(pp, labels.get(key))
                spike_all = by["__all__"][1]
                for b, (nb, fr) in by.items():
                    if b == "__all__":
                        continue
                    rows.append({
                        "hypothesis": hyp, "cell": cid, "stem": rec["_stem"],
                        "K": K_max, "U_circ_K": "", "A_K": round(fr, 6),
                        "bucket": b, "n": nb,
                    })
            print(f"{cid[:52]:52}{K_max:>6}{floor:>15.3f}{spike_all:>10.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["hypothesis", "cell", "stem", "K", "U_circ_K", "A_K", "bucket", "n"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
