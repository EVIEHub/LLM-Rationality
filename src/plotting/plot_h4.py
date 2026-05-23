"""H4 figure — relationship between rational gap and reasoning length.

Reads ``${results_dir}/h4/<model>_<dataset>_L<L>_seed<S>.json``.
Plots $\\hat{\\mathcal{R}}_K(L)$ vs $L$ for each dataset, plus $U^\\circ_K(L)$
and $\\bar{U}_K(L)$ on the same axes for context.

Usage:
    python -m src.plotting.plot_h4
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.pipeline.paths import load_paths
from src.plotting._common import SCOPE_DATASETS, datasets_in_scope, split_cells_by_scope


def _render_scope(scope, cells, figures_dir):
    if not cells:
        print(f"  scope {scope!r}: no cells, skipping figure")
        return None
    datasets = datasets_in_scope(scope, cells)
    print(f"  scope {scope!r}: {len(cells)} cells, datasets={datasets}")

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(4.4 * len(datasets), 3.4),
        sharey=False, squeeze=False,
    )
    axes = axes[0]

    for j, ds in enumerate(datasets):
        ax = axes[j]
        ds_cells = sorted(
            (r for r in cells if r["dataset"] == ds),
            key=lambda r: r["L"],
        )
        Ls = [r["L"] for r in ds_cells]
        u_circ = [r["aggregates_at_K_max"]["U_circ_K"] for r in ds_cells]
        u_bar = [r["aggregates_at_K_max"]["U_bar_K"] for r in ds_cells]
        r_hat = [r["aggregates_at_K_max"]["R_hat_K"] for r in ds_cells]

        ci_low = [r["bootstrap_R_hat_K_at_K_max"]["ci_low"] for r in ds_cells]
        ci_high = [r["bootstrap_R_hat_K_at_K_max"]["ci_high"] for r in ds_cells]
        err_low = [v - lo for v, lo in zip(r_hat, ci_low)]
        err_high = [hi - v for v, hi in zip(r_hat, ci_high)]

        ax.plot(Ls, u_circ, marker="^", color="C0", lw=1.4, label=r"$U^\circ_K$")
        ax.plot(Ls, u_bar, marker="v", color="C3", lw=1.4, label=r"$\bar{U}_K$")
        ax.errorbar(Ls, r_hat, yerr=[err_low, err_high],
                    marker="o", color="black", lw=1.6, capsize=3,
                    label=r"$\hat{R}_K$ (95% CI)")

        ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_xticks(Ls)
        ax.set_xticklabels([str(L) for L in Ls], fontsize=8)
        ax.set_xlabel(r"$L$  (max reasoning tokens)")
        if j == 0:
            ax.set_ylabel(r"value at $K=K_{\max}$", fontsize=14)
        ax.set_title(
            f"{ds}  ($K_{{\\max}}$={ds_cells[0]['K_max']}, "
            f"M={ds_cells[0]['M']})",
            fontsize=10,
        )
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if j == len(datasets) - 1:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"H4 ({scope}) — "
        r"$\hat{\mathcal{R}}_K(L)$ vs reasoning-length budget $L$  "
        r"(Tulu-3-RLVR, seed=0; budget-forced two-stage sampling)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    pdf_path = figures_dir / f"h4_length_sweep_{scope}.pdf"
    png_path = figures_dir / f"h4_length_sweep_{scope}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Wrote {pdf_path}")
    return pdf_path


def main() -> None:
    paths = load_paths()
    h4_dir = paths.results_dir / "h4"
    figures_dir = paths.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for json_path in sorted(h4_dir.glob("*_L*_seed*.json")):
        results.append(json.loads(json_path.read_text()))
    if not results:
        raise SystemExit(f"No results JSONs found in {h4_dir}")

    by_scope = split_cells_by_scope(results)
    print(f"Loaded {len(results)} H4 cells; splitting by experiment scope:")
    for scope in SCOPE_DATASETS:
        _render_scope(scope, by_scope[scope], figures_dir)

    print()
    print("=== H4 summary (at K_max) ===")
    print(f"{'L':>6} {'dataset':<14} {'U_circ':>8} {'U_bar':>8} {'R_hat':>8} {'CI low':>8} {'CI high':>8}")
    for r in sorted(results, key=lambda r: (r["dataset"], r["L"])):
        a = r["aggregates_at_K_max"]
        ci = r["bootstrap_R_hat_K_at_K_max"]
        print(f"{r['L']:>6} {r['dataset']:<14} {a['U_circ_K']:>8.3f} "
              f"{a['U_bar_K']:>8.3f} {a['R_hat_K']:>8.3f} "
              f"{ci['ci_low']:>8.3f} {ci['ci_high']:>8.3f}")


if __name__ == "__main__":
    main()
