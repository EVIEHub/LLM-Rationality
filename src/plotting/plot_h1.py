"""H1 figure — saturation curves of $\\hat{\\mathcal{R}}_K$ vs $K$.

Reads ``${results_dir}/h1/<model>_<dataset>_seed<S>.json`` files and
renders a (M_models × N_datasets) grid of subplots. Each subplot shows:
- the per-K curve from the cell's ``saturation_curve`` field;
- an errorbar at $K=K_{\\max}$ from the bootstrap CI in
  ``bootstrap_R_hat_K_at_K_max``.

Usage:
    python -m src.plotting.plot_h1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.pipeline.paths import load_paths


def main() -> None:
    paths = load_paths()
    h1_dir = paths.results_dir / "h1"
    figures_dir = paths.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for json_path in sorted(h1_dir.glob("*_seed*.json")):
        results.append(json.loads(json_path.read_text()))

    if not results:
        raise SystemExit(f"No results JSONs found in {h1_dir}")

    models = sorted({r["model"] for r in results})
    datasets = sorted({r["dataset"] for r in results})
    n_m, n_d = len(models), len(datasets)
    print(f"Found {len(results)} cells: {n_m} models × {n_d} datasets")

    fig, axes = plt.subplots(
        n_m, n_d,
        figsize=(3.5 * n_d, 2.6 * n_m),
        sharex=True, sharey=False,
        squeeze=False,
    )

    for r in results:
        i = models.index(r["model"])
        j = datasets.index(r["dataset"])
        ax = axes[i, j]

        Ks = [pt["K"] for pt in r["saturation_curve"]]
        R_hats = [pt["R_hat_K"] for pt in r["saturation_curve"]]
        U_circs = [pt["U_circ_K"] for pt in r["saturation_curve"]]
        U_bars = [pt["U_bar_K"] for pt in r["saturation_curve"]]

        ax.plot(Ks, R_hats, marker="o", color="black", lw=1.6, label=r"$\hat{R}_K$")
        ax.plot(Ks, U_circs, marker="^", color="C0", lw=1.0, alpha=0.7,
                label=r"$U^\circ_K$")
        ax.plot(Ks, U_bars, marker="v", color="C3", lw=1.0, alpha=0.7,
                label=r"$\bar{U}_K$")

        ci = r["bootstrap_R_hat_K_at_K_max"]
        K_max = r["K_max"]
        R_at_max = r["aggregates_at_K_max"]["R_hat_K"]
        ax.errorbar(
            [K_max], [R_at_max],
            yerr=[[R_at_max - ci["ci_low"]], [ci["ci_high"] - R_at_max]],
            fmt="none", color="black", capsize=4, lw=1.5,
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks], fontsize=7)
        if i == n_m - 1:
            ax.set_xlabel(r"$K$ (sampling budget)", fontsize=9)
        if j == 0:
            ax.set_ylabel(r"value", fontsize=9)
        ax.set_title(
            f"{r['model']}  ×  {r['dataset']}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        if i == 0 and j == n_d - 1:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        r"H1 — Saturation of $\hat{\mathcal{R}}_K$ vs $K$  (seed=0; "
        r"errorbar = 95% bootstrap CI over prompts at $K_{\max}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    pdf_path = figures_dir / "h1_saturation.pdf"
    png_path = figures_dir / "h1_saturation.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Wrote {pdf_path}\nWrote {png_path}")

    # Also dump a cross-cell summary table to stdout.
    print()
    print("=== H1 summary (at K_max) ===")
    print(f"{'model':<28} {'dataset':<12} {'U_circ':>8} {'U_bar':>8} {'R_hat':>8} {'CI low':>8} {'CI high':>8}")
    for r in results:
        agg = r["aggregates_at_K_max"]
        ci = r["bootstrap_R_hat_K_at_K_max"]
        print(f"{r['model']:<28} {r['dataset']:<12} "
              f"{agg['U_circ_K']:>8.3f} {agg['U_bar_K']:>8.3f} "
              f"{agg['R_hat_K']:>8.3f} {ci['ci_low']:>8.3f} {ci['ci_high']:>8.3f}")


if __name__ == "__main__":
    main()
