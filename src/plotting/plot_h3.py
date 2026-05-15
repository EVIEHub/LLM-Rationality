"""H3 figures — rational gap per inference procedure.

Two figures:

  h3_sc_saturation.{pdf,png}
      $\\hat{\\mathcal{R}}_K(\\pi_{\\text{SC}(n)})$ vs $n$ for each
      (model, dataset). One curve per model; one panel per dataset.
      Headline plot for the SC saturation finding.

  h3_procedures.{pdf,png}
      Bar chart of $\\hat{\\mathcal{R}}_K$ across all evaluated
      procedures (direct $\\tau \\in \\{0, 0.7, 1.0\\}$ and
      SC at $n \\in \\{2, 4, 8, 16, 32\\}$) for each (model, dataset).
      Supplementary plot for the procedure-comparison view.

Reads ``${results_dir}/h3/<model>_<dataset>_t<tau>_seed<S>.json`` and
``${results_dir}/h3/<model>_<dataset>_sc_n<n>_seed<S>.json``.

Usage:
    python -m src.plotting.plot_h3
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.paths import load_paths
from src.plotting._common import SCOPE_DATASETS, datasets_in_scope, split_cells_by_scope


_DIRECT_RE = re.compile(r"^(.+?)_(gsm8k|math|humaneval)_t([\d\.]+)_seed(\d+)\.json$")
_SC_RE = re.compile(r"^(.+?)_(gsm8k|math|humaneval)_sc_n(\d+)_seed(\d+)\.json$")

_MODEL_COLORS = {
    "tulu3-8b-rlvr":         "#1f77b4",
    "qwen2.5-7b-instruct":   "#2ca02c",
    "llama3.1-8b-instruct":  "#d62728",
}
_MODEL_LABELS = {
    "tulu3-8b-rlvr":         "Tülu-3-8B (RLVR)",
    "qwen2.5-7b-instruct":   "Qwen2.5-7B-Instruct",
    "llama3.1-8b-instruct":  "Llama-3.1-8B-Instruct",
}


def _load_cells(h3_dir: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for json_path in sorted(h3_dir.glob("*.json")):
        d = json.loads(json_path.read_text())
        proc = d.get("procedure", "direct" if "tau" in d else "sc")
        d["_proc"] = proc
        cells.append(d)
    return cells


def _r_hat_with_err(cell: dict[str, Any]) -> tuple[float, float, float]:
    a = cell["aggregates_at_K_max"]
    ci = cell["bootstrap_R_hat_K_at_K_max"]
    r = a["R_hat_K"]
    return r, r - ci["ci_low"], ci["ci_high"] - r


def plot_sc_saturation(scope: str, cells: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    sc_cells = [c for c in cells if c["_proc"] == "sc"]
    if not sc_cells:
        print(f"  scope {scope!r}: no SC cells, skipping saturation figure")
        return None

    datasets = datasets_in_scope(scope, sc_cells)
    models = sorted({c["model"] for c in sc_cells})

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(4.4 * len(datasets), 3.6),
        sharey=False, squeeze=False,
    )
    axes = axes[0]

    for j, ds in enumerate(datasets):
        ax = axes[j]
        for model in models:
            rows = sorted(
                (c for c in sc_cells if c["dataset"] == ds and c["model"] == model),
                key=lambda c: c["sc_n"],
            )
            if not rows:
                continue
            ns = [c["sc_n"] for c in rows]
            r_hats = [c["aggregates_at_K_max"]["R_hat_K"] for c in rows]
            err_lows = [r - c["bootstrap_R_hat_K_at_K_max"]["ci_low"]
                        for r, c in zip(r_hats, rows)]
            err_highs = [c["bootstrap_R_hat_K_at_K_max"]["ci_high"] - r
                         for r, c in zip(r_hats, rows)]
            ax.errorbar(
                ns, r_hats, yerr=[err_lows, err_highs],
                marker="o", lw=1.6, capsize=3,
                color=_MODEL_COLORS.get(model, "black"),
                label=_MODEL_LABELS.get(model, model),
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"$n$  (samples per SC draw)")
        if j == 0:
            ax.set_ylabel(r"$\hat{\mathcal{R}}_K(\pi_{\mathrm{SC}(n)})$")
        # Match the spec: M is 1319 for gsm8k, 1000 for math.
        sample_M = next(
            (c["M"] for c in sc_cells if c["dataset"] == ds), None,
        )
        ax.set_title(f"{ds}  (M={sample_M}, K=64)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        if j == len(datasets) - 1:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"H3 ({scope}) — SC procedure saturation: "
        r"$\hat{\mathcal{R}}_K(\pi_{\mathrm{SC}(n)})$ vs $n$  "
        r"(K=64 bootstrap draws, seed=0)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    pdf_path = figures_dir / f"h3_sc_saturation_{scope}.pdf"
    png_path = figures_dir / f"h3_sc_saturation_{scope}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pdf_path


def plot_procedures(scope: str, cells: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """Bar chart per (model, dataset): R_hat across all procedures."""
    if not cells:
        print(f"  scope {scope!r}: no cells, skipping procedures figure")
        return None
    models = sorted({c["model"] for c in cells})
    datasets = datasets_in_scope(scope, cells)

    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(3.7 * len(datasets), 2.6 * len(models)),
        squeeze=False, sharey=True,
    )

    # Procedure ordering and labels.
    proc_order: list[tuple[str, str]] = [
        ("direct", "0.0"), ("direct", "0.7"), ("direct", "1.0"),
        ("sc", "2"), ("sc", "4"), ("sc", "8"), ("sc", "16"), ("sc", "32"),
    ]
    proc_labels = [
        r"$\tau$=0",  r"$\tau$=0.7",  r"$\tau$=1",
        "SC n=2",  "SC n=4",  "SC n=8",  "SC n=16",  "SC n=32",
    ]
    proc_colors = [
        "#fc8d62", "#fdb863", "#fee090",
        "#91bfdb", "#74add1", "#4575b4", "#313695", "#08306b",
    ]

    for i, model in enumerate(models):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            for k, (kind, val) in enumerate(proc_order):
                if kind == "direct":
                    rows = [c for c in cells
                            if c["model"] == model and c["dataset"] == ds
                            and c["_proc"] == "direct"
                            and abs(c["tau"] - float(val)) < 1e-9]
                else:
                    rows = [c for c in cells
                            if c["model"] == model and c["dataset"] == ds
                            and c["_proc"] == "sc"
                            and c["sc_n"] == int(val)]
                if not rows:
                    continue
                r, lo, hi = _r_hat_with_err(rows[0])
                ax.bar(
                    k, r, color=proc_colors[k],
                    yerr=[[lo], [hi]], capsize=3, ecolor="black",
                )

            ax.set_xticks(range(len(proc_order)))
            ax.set_xticklabels(proc_labels, rotation=45, ha="right", fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)
            if j == 0:
                ax.set_ylabel(
                    _MODEL_LABELS.get(model, model) + "\n" + r"$\hat{\mathcal{R}}_K$",
                    fontsize=9,
                )
            if i == 0:
                ax.set_title(ds, fontsize=10)

    fig.suptitle(
        f"H3 ({scope}) — "
        r"$\hat{\mathcal{R}}_K$ across inference procedures  (K=64, seed=0)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    pdf_path = figures_dir / f"h3_procedures_{scope}.pdf"
    png_path = figures_dir / f"h3_procedures_{scope}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pdf_path


def main() -> None:
    paths = load_paths()
    h3_dir = paths.results_dir / "h3"
    figures_dir = paths.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cells = _load_cells(h3_dir)
    if not cells:
        raise SystemExit(f"No H3 results JSONs found in {h3_dir}")
    print(f"Loaded {len(cells)} H3 cells")

    by_scope = split_cells_by_scope(cells)
    for scope in SCOPE_DATASETS:
        scope_cells = by_scope[scope]
        if not scope_cells:
            print(f"  scope {scope!r}: no cells")
            continue
        print(f"  scope {scope!r}: {len(scope_cells)} cells")
        sat = plot_sc_saturation(scope, scope_cells, figures_dir)
        if sat:
            print(f"  Wrote {sat}")
        proc = plot_procedures(scope, scope_cells, figures_dir)
        if proc:
            print(f"  Wrote {proc}")


if __name__ == "__main__":
    main()
