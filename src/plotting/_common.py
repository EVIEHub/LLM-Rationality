"""Shared constants + helpers for the H1-H4 plotters.

All four hypothesis plotters (`plot_h1`, `plot_h2`, `plot_h3`, `plot_h4`)
emit one figure per experiment scope (`development` vs `deployment`) so
the in-distribution and contamination-resistant results are visualised
side by side, not commingled.

`SCOPE_DATASETS` maps scope -> ordered dataset list (the order drives
column order within each figure). When new datasets land, add them here
and the splitter handles routing automatically.
"""

from __future__ import annotations

from typing import Any

# Paper figure style, applied on import so every plotter shares it.
# Times New Roman (with metric-compatible fallbacks for Linux boxes that
# lack the proprietary font) + larger axis/tick fonts for readability.
import matplotlib as _mpl

_mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman", "Liberation Serif", "Nimbus Roman No9 L",
        "Times", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",   # Times-like math to match the serif text
    "axes.labelsize": 14,         # x/y axis labels
    "axes.titlesize": 13,
    "ytick.labelsize": 13,        # y-axis tick numbers (larger, as requested)
    "xtick.labelsize": 11,
    "legend.fontsize": 10,
})

SCOPE_DATASETS: dict[str, list[str]] = {
    "development": ["gsm8k", "math", "humaneval", "bbh"],
    "deployment":  ["matharena", "livecodebench"],
    "preference":  ["ultrafeedback", "alpaca_eval"],
}


def split_cells_by_scope(
    cells: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Bucket cells into scopes by their `dataset` field.

    Cells whose dataset is not in any scope are dropped with a warning
    (caller can inspect the returned dict to see which scopes are empty).
    """
    by_scope: dict[str, list[dict[str, Any]]] = {s: [] for s in SCOPE_DATASETS}
    known = {d for ds_list in SCOPE_DATASETS.values() for d in ds_list}
    unknown_datasets = set()
    for c in cells:
        ds = c["dataset"]
        if ds not in known:
            unknown_datasets.add(ds)
            continue
        for scope, ds_list in SCOPE_DATASETS.items():
            if ds in ds_list:
                by_scope[scope].append(c)
                break
    if unknown_datasets:
        print(f"WARNING: cells with unclassified dataset(s) "
              f"{sorted(unknown_datasets)} are not plotted")
    return by_scope


def datasets_in_scope(scope: str, cells: list[dict[str, Any]]) -> list[str]:
    """Spec-defined dataset order for the scope, filtered to those present."""
    return [d for d in SCOPE_DATASETS[scope]
            if any(c["dataset"] == d for c in cells)]
