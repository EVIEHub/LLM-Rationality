"""H4 figure — rational value risk vs forced reasoning-length budget $L$.

One combined figure (development + deployment datasets together) in the
same row/column convention as the H3 plots: rows = models, columns =
datasets, x-axis = $L$ (the budget-forced stage-1 max-tokens cap). On
each panel we draw REU / AEU as dotted lines and RVR as a solid line
with 95\\% prompt-bootstrap CI error bars — palette and style anchored
to ``plot_h3._METRIC_STYLE`` so all three H-figures look like siblings.

Reads ``${results_dir}/h4/<model>_<dataset>_L<L>_seed<S>.json``.

Usage:
    python -m src.plotting.plot_h4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.pipeline.paths import load_paths

# Match the H3 figures: same font + sizing + per-metric style anchors,
# so the three H-figures share a single visual vocabulary.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
})

# Canonical ordering: same as the H3 plots so a reader can scan across.
# BBH is intentionally excluded — only the Tülu-3-RLVR BBH cells exist,
# so the Qwen / Llama BBH panels would render empty. The Tülu BBH
# numbers stay in the result JSONs and can be quoted in prose if needed.
_DATASET_ORDER = [
    ("gsm8k",     "GSM8K"),
    ("math",      "MATH"),
    ("humaneval", "HumanEval"),
    ("matharena", "MathArena"),
]
_MODEL_ORDER = [
    # Display labels truncated to match plot_h3.py and free up the row-
    # header column; the full names appear in the table captions.
    ("tulu3-8b-rlvr",        "Tülu-3-8B"),
    ("qwen2.5-7b-instruct",  "Qwen2.5-7B"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B"),
]
# Same anchor palette as the H1 tables / H3 plots.
_METRIC_STYLE = [
    ("U_circ_K", "REU", "#8FC4DF", "^", ":"),
    ("U_bar_K",  "AEU", "#D6A77E", "v", ":"),
    ("R_hat_K",  "RVR", "#CC7E9A", "o", "-"),
]


def _present(order: list[tuple[str, str]], keys: set[str]) -> list[tuple[str, str]]:
    return [(k, lbl) for k, lbl in order if k in keys]


def _save(fig, figures_dir: Path, stem: str) -> Path:
    pdf = figures_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pdf


def plot_h4_budget(cells: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """H4 budget-forced sweep: rows = models, columns = datasets.

    Mirrors :func:`src.plotting.plot_h3.plot_sc_saturation` exactly —
    same grid shape, same x-axis treatment (evenly-spaced categorical
    ticks labelled with the actual $L$ values, not a log axis), same
    REU/AEU/RVR colour anchors. RVR carries 95\\% prompt-bootstrap
    CI bars from each cell's ``bootstrap_R_hat_K_at_K_max``.
    """
    if not cells:
        print("  no H4 cells, skipping figure")
        return None
    datasets = _present(_DATASET_ORDER, {c["dataset"] for c in cells})
    models = _present(_MODEL_ORDER, {c["model"] for c in cells})

    # Evenly-spaced categorical x-axis labelled with the actual L values.
    all_Ls = sorted({int(c["L"]) for c in cells})
    pos = {L: i for i, L in enumerate(all_Ls)}

    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(3.0 * len(datasets), 2.6 * len(models)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, (m, m_lbl) in enumerate(models):
        for j, (ds, ds_lbl) in enumerate(datasets):
            ax = axes[i][j]
            rows = sorted(
                (c for c in cells
                 if c["dataset"] == ds and c["model"] == m),
                key=lambda c: int(c["L"]),
            )
            legend_panel = (i == 0 and j == len(datasets) - 1)
            if rows:
                xs = [pos[int(c["L"])] for c in rows]
                for field, label, color, marker, ls in _METRIC_STYLE:
                    ys = [c["aggregates_at_K_max"][field] for c in rows]
                    kw: dict[str, Any] = {}
                    if field == "R_hat_K":
                        kw["yerr"] = [
                            [y - c["bootstrap_R_hat_K_at_K_max"]["ci_low"]
                             for y, c in zip(ys, rows)],
                            [c["bootstrap_R_hat_K_at_K_max"]["ci_high"] - y
                             for y, c in zip(ys, rows)],
                        ]
                        kw["capsize"] = 2.5
                    ax.errorbar(
                        xs, ys, marker=marker, lw=3.0, markersize=8,
                        ls=ls, color=color,
                        label=(label if legend_panel else None), **kw,
                    )
            ax.set_xticks(range(len(all_Ls)))
            # The T=0 cell is "no reasoning"; label as 0 not as $\log_2$.
            # Rotate so the wide adjacent labels (1024/2048) don't collide.
            ax.set_xticklabels(
                [str(L) for L in all_Ls], rotation=45, ha="right",
            )
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            if i == len(models) - 1:
                ax.set_xlabel(r"$T$")
            if j == 0:
                ax.set_ylabel(m_lbl + "\nutility", fontsize=19)
            if i == 0:
                ax.set_title(ds_lbl, fontsize=18)
            if legend_panel:
                ax.legend(fontsize=15, loc="upper right")

    fig.tight_layout()
    return _save(fig, figures_dir, "h4_budget_sweep")


def _load_cells(h4_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for json_path in sorted(h4_dir.glob("*_L*_seed*.json")):
        out.append(json.loads(json_path.read_text()))
    return out


def main() -> None:
    paths = load_paths()
    h4_dir = paths.results_dir / "h4"
    figures_dir = paths.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cells = _load_cells(h4_dir)
    if not cells:
        raise SystemExit(f"No H4 results JSONs found in {h4_dir}")
    print(f"Loaded {len(cells)} H4 cells")

    out = plot_h4_budget(cells, figures_dir)
    if out:
        print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
