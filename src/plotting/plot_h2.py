"""H2 figure — Tulu-3 trajectory: U_circ_K vs U_bar_K along base→SFT→DPO→RLVR.

Per ``methodology/hypotheses.md``: H2 tests claim (b), that alignment
does NOT eliminate the gap. The figure must show $U^\\circ_K$ and
$\\bar{U}_K$ SEPARATELY across stages — not just the gap — so a reader
can see whether a shrinking gap is due to $U^\\circ_K$ stagnating /
falling (sharpening) or $\\bar{U}_K$ catching up (alignment working).

Reads ``${results_dir}/h2/<model>_<dataset>_seed<S>.json``.

Usage:
    python -m src.plotting.plot_h2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.paths import load_paths

# Post-SFT alignment trajectory only — base was excluded from H2 because
# its few-shot prompting mode confounds the chat-mode SFT/DPO/RLVR
# comparison (see AGENT/methodology/hypotheses.md, design note 2026-05-08).
_TRAJECTORY_ORDER = ["sft", "dpo", "rlvr"]
_TRAJECTORY_LABEL = {
    "sft": "SFT",
    "dpo": "DPO",
    "rlvr": "RLVR",
}


def main() -> None:
    paths = load_paths()
    h2_dir = paths.results_dir / "h2"
    figures_dir = paths.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for json_path in sorted(h2_dir.glob("*_seed*.json")):
        results.append(json.loads(json_path.read_text()))
    if not results:
        raise SystemExit(f"No results JSONs found in {h2_dir}")

    datasets = sorted({r["dataset"] for r in results})
    print(f"Found {len(results)} cells across {len(datasets)} datasets")

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(4.2 * len(datasets), 3.6),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    bar_w = 0.35
    xs = np.arange(len(_TRAJECTORY_ORDER))

    for j, ds in enumerate(datasets):
        ax = axes[j]
        # Index: stage -> agg
        per_stage: dict[str, dict] = {}
        for r in results:
            if r["dataset"] != ds:
                continue
            stage = r.get("trajectory_stage")
            if stage in _TRAJECTORY_ORDER:
                per_stage[stage] = r["aggregates_at_K_max"]

        if not per_stage:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(ds, fontsize=10)
            continue

        u_circs = [per_stage[s]["U_circ_K"] if s in per_stage else np.nan
                   for s in _TRAJECTORY_ORDER]
        u_bars = [per_stage[s]["U_bar_K"] if s in per_stage else np.nan
                  for s in _TRAJECTORY_ORDER]

        ax.bar(xs - bar_w/2, u_circs, bar_w, label=r"$U^\circ_K$",
               color="C0", edgecolor="black", lw=0.5)
        ax.bar(xs + bar_w/2, u_bars, bar_w, label=r"$\bar{U}_K$",
               color="C3", edgecolor="black", lw=0.5)

        # Annotate gap as line connecting the two bars
        for k, s in enumerate(_TRAJECTORY_ORDER):
            if s in per_stage:
                gap = per_stage[s]["R_hat_K"]
                ax.text(k, max(u_circs[k], u_bars[k]) + 0.02,
                        f"$\\hat{{R}}_K$={gap:.3f}",
                        ha="center", fontsize=7, color="black")

        ax.set_xticks(xs)
        ax.set_xticklabels([_TRAJECTORY_LABEL[s] for s in _TRAJECTORY_ORDER],
                           fontsize=9)
        ax.set_xlabel("trajectory stage")
        if j == 0:
            ax.set_ylabel(r"value at $K=K_{\max}$")
        ax.set_title(f"{ds}  ($K_{{\\max}}$={results[0]['K_max']})", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if j == len(datasets) - 1:
            ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "H2 — alignment trajectory: $U^\\circ_K$ vs $\\bar{U}_K$ "
        "along base→SFT→DPO→RLVR  (Tulu-3-8B, seed=0)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    pdf_path = figures_dir / "h2_trajectory.pdf"
    png_path = figures_dir / "h2_trajectory.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Wrote {pdf_path}\nWrote {png_path}")

    # Cross-cell summary
    print()
    print("=== H2 summary (at K_max) ===")
    print(f"{'stage':<8} {'dataset':<10} {'U_circ':>8} {'U_bar':>8} {'R_hat':>8}")
    for stage in _TRAJECTORY_ORDER:
        for ds in datasets:
            for r in results:
                if r.get("trajectory_stage") == stage and r["dataset"] == ds:
                    a = r["aggregates_at_K_max"]
                    print(f"{stage:<8} {ds:<10} {a['U_circ_K']:>8.3f} "
                          f"{a['U_bar_K']:>8.3f} {a['R_hat_K']:>8.3f}")


if __name__ == "__main__":
    main()
