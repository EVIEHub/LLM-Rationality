"""H3 figures — REU / AEU / RVR across inference procedures.

Two combined figures (development + deployment datasets together, never
split by scope):

  h3_sc_saturation.{pdf,png}
      Self-consistency SC($n$): REU, AEU, RVR vs $n$. Grid of
      (model x dataset) panels; one panel per cell, three lines.
      Only the answer-extractable datasets appear (GSM8K, MATH,
      MathArena) — SC needs a key to vote on.

  h3_temperature.{pdf,png}
      Direct sampling: REU, AEU, RVR vs decoding temperature
      $\\tau \\in \\{0, 0.7, 1.0\\}$. Grid of (model x dataset) panels.
      RVR is drawn as a solid line; REU and AEU as dotted lines.
      Greedy ($\\tau{=}0$) is deterministic so REU=AEU and RVR=0.

Reads ``${results_dir}/h3/*.json``.

Usage:
    python -m src.plotting.plot_h3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.pipeline.paths import load_paths

# Times New Roman look (falls back to metric-compatible clones if the
# exact font is not installed) + larger fonts, with the y-axis emphasised.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
})


# Canonical ordering for the figure grids (Conversation -> Development ->
# Deployment); only datasets actually present are shown. LiveCodeBench is
# intentionally excluded — H3 procedures (SC, temperature sweep) don't
# motivate a coding-benchmark column in the figure narrative; the LCB
# numbers stay in the headline H1/H5 tables.
_DATASET_ORDER = [
    ("ultrafeedback", "UltraFeedback"), ("alpaca_eval", "AlpacaEval"),
    ("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval"),
    ("matharena", "MathArena"),
]
_MODEL_ORDER = [
    # Display labels truncated (drop "-RLVR" / "-Instruct" tails) so the
    # row-header text doesn't overlap the tight grid columns. The
    # disambiguation lives in the figure caption, not the row label.
    ("tulu3-8b-rlvr",        "Tülu-3-8B"),
    ("qwen2.5-7b-instruct",  "Qwen2.5-7B"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B"),
]

# REU / AEU / RVR line styles — colours anchored to match the tables.
# RVR is the headline (solid line); REU and AEU are the decomposition (dotted).
_METRIC_STYLE = [
    ("U_circ_K", "REU", "#8FC4DF", "^", ":"),   # soft blue, dotted
    ("U_bar_K",  "AEU", "#D6A77E", "v", ":"),   # muted amber, dotted
    ("R_hat_K",  "RVR", "#CC7E9A", "o", "-"),   # dusty pink, solid
]

_TAUS = [0.0, 0.7, 1.0]


def _load_cells(h3_dir: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for json_path in sorted(h3_dir.glob("*.json")):
        d = json.loads(json_path.read_text())
        d["_proc"] = d.get("procedure", "direct" if "tau" in d else "sc")
        cells.append(d)
    return cells


def _present(order: list[tuple[str, str]], keys: set[str]) -> list[tuple[str, str]]:
    return [(k, lbl) for k, lbl in order if k in keys]


def _grid(models, datasets):
    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(3.6 * len(datasets), 2.6 * len(models)),
        sharex=True, sharey=True, squeeze=False,
    )
    return fig, axes


def plot_sc_saturation(cells: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """SC saturation: rows = models, columns = datasets.

    Matches the row/column convention of :func:`plot_temperature`
    so the reader doesn't have to mentally transpose between the two
    H3 figures.
    """
    sc = [c for c in cells if c["_proc"] == "sc"]
    if not sc:
        print("  no SC cells, skipping saturation figure")
        return None
    datasets = _present(_DATASET_ORDER, {c["dataset"] for c in sc})
    models = _present(_MODEL_ORDER, {c["model"] for c in sc})

    # Evenly-spaced categorical x-axis labelled with the actual n values
    # (2, 4, 8, ...) instead of a log 2^k axis.
    all_ns = sorted({c["sc_n"] for c in sc})
    pos = {n: i for i, n in enumerate(all_ns)}

    fig, axes = _grid(models, datasets)
    for i, (m, m_lbl) in enumerate(models):
        for j, (ds, ds_lbl) in enumerate(datasets):
            ax = axes[i][j]
            rows = sorted(
                (c for c in sc if c["dataset"] == ds and c["model"] == m),
                key=lambda c: c["sc_n"],
            )
            legend_panel = (i == 0 and j == len(datasets) - 1)
            if rows:
                xs = [pos[c["sc_n"]] for c in rows]
                for field, label, color, marker, ls in _METRIC_STYLE:
                    ys = [c["aggregates_at_K_max"][field] for c in rows]
                    kw: dict[str, Any] = {}
                    if field == "R_hat_K":
                        kw["yerr"] = [
                            [y - c["bootstrap_R_hat_K_at_K_max"]["ci_low"] for y, c in zip(ys, rows)],
                            [c["bootstrap_R_hat_K_at_K_max"]["ci_high"] - y for y, c in zip(ys, rows)],
                        ]
                        kw["capsize"] = 2.5
                    ax.errorbar(xs, ys, marker=marker, lw=3.0, markersize=8, ls=ls, color=color,
                                label=(label if legend_panel else None), **kw)
            ax.set_xticks(range(len(all_ns)))
            ax.set_xticklabels([str(n) for n in all_ns])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            if i == len(models) - 1:
                ax.set_xlabel(r"$n$")
            if j == 0:
                # Model label on the y-axis (left edge); dataset labels
                # are the column titles.
                ax.set_ylabel(m_lbl + "\nutility", fontsize=19)
            if i == 0:
                ax.set_title(ds_lbl, fontsize=18)
            if legend_panel:
                ax.legend(fontsize=15, loc="upper right")

    fig.tight_layout()
    return _save(fig, figures_dir, "h3_sc_saturation")


def plot_temperature(cells: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """Temperature sweep: rows = models, columns = datasets.

    Same row/column convention as :func:`plot_sc_saturation` —
    model labels on the left edge of each row, dataset labels along
    the column titles.
    """
    direct = [c for c in cells if c["_proc"] == "direct" and "tau" in c]
    if not direct:
        print("  no direct cells, skipping temperature figure")
        return None
    datasets = _present(_DATASET_ORDER, {c["dataset"] for c in direct})
    models = _present(_MODEL_ORDER, {c["model"] for c in direct})

    # index (model, dataset, tau) -> cell
    idx: dict[tuple, dict] = {}
    for c in direct:
        idx[(c["model"], c["dataset"], round(float(c["tau"]), 3))] = c

    # Rows = models (3), columns = datasets (up to 6). Wide & short.
    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(2.2 * len(datasets), 2.0 * len(models)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, (m, m_lbl) in enumerate(models):
        for j, (ds, ds_lbl) in enumerate(datasets):
            ax = axes[i][j]
            legend_panel = (i == 0 and j == len(datasets) - 1)
            present_taus = [t for t in _TAUS if (m, ds, t) in idx]
            if present_taus:
                for field, label, color, marker, ls in _METRIC_STYLE:
                    ys = [idx[(m, ds, t)]["aggregates_at_K_max"][field] for t in present_taus]
                    kw: dict[str, Any] = {}
                    if field == "R_hat_K":
                        kw["yerr"] = [
                            [idx[(m, ds, t)]["aggregates_at_K_max"][field]
                             - idx[(m, ds, t)]["bootstrap_R_hat_K_at_K_max"]["ci_low"]
                             for t in present_taus],
                            [idx[(m, ds, t)]["bootstrap_R_hat_K_at_K_max"]["ci_high"]
                             - idx[(m, ds, t)]["aggregates_at_K_max"][field]
                             for t in present_taus],
                        ]
                        kw["capsize"] = 2.5
                    ax.errorbar(present_taus, ys, marker=marker, lw=3.0, markersize=8, ls=ls,
                                color=color, label=(label if legend_panel else None), **kw)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(_TAUS)
            if i == len(models) - 1:
                ax.set_xlabel(r"$\tau$")
            if j == 0:
                ax.set_ylabel(m_lbl + "\nutility", fontsize=19)
            if i == 0:
                ax.set_title(ds_lbl, fontsize=18)
            if legend_panel:
                ax.legend(fontsize=15, loc="upper right")

    fig.tight_layout()
    return _save(fig, figures_dir, "h3_temperature")


def _save(fig, figures_dir: Path, stem: str) -> Path:
    pdf = figures_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pdf


def main() -> None:
    paths = load_paths()
    h3_dir = paths.results_dir / "h3"
    figures_dir = paths.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cells = _load_cells(h3_dir)
    if not cells:
        raise SystemExit(f"No H3 results JSONs found in {h3_dir}")
    print(f"Loaded {len(cells)} H3 cells")

    for fn in (plot_sc_saturation, plot_temperature):
        out = fn(cells, figures_dir)
        if out:
            print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
