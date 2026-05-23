"""Generate the paper's LaTeX results tables (H1, H2, H4, H5) directly
from the result JSONs — no hand transcription.

H3 is a figure, not a table; see :mod:`src.plotting.plot_h3`.

Symbol mapping (paper presentation names ← pipeline fields):
    REU  ← aggregates_at_K_max.U_circ_K   (rational expected utility)
    AEU  ← aggregates_at_K_max.U_bar_K     (actual expected utility)
    RVR  ← aggregates_at_K_max.R_hat_K     (rational value risk = REU - AEU)
The RVR 95% prompt-bootstrap CI half-width is
    (bootstrap_R_hat_K_at_K_max.ci_high - .ci_low) / 2.

Each table's layout matches the agreed paper design:
    H1  full-width: rows = datasets (3 task bands), columns = open models
        (Llama / Qwen / Tülu / Qwen-72B), each split REU/AEU/RVR; RVR with
        +/- CI; bold = smallest RVR per row.
    H2  single-col: rows = dataset x {REU,AEU,RVR}, columns = SFT/DPO/RLVR;
        RVR with +/- CI; bold = smallest RVR per dataset.
    H4  full-width: rows = Tülu-3-RLVR x {REU,AEU,RVR}, columns = dataset x L;
        no inline CI (width); bold = peak RVR per dataset.
    H5  single-col: rows = deployment datasets, columns = frontier APIs
        (GPT-5.2 / DeepSeek-V4-Flash), each split REU/AEU/RVR; bold =
        smallest RVR per row.

Conversation tasks (UltraFeedback, AlpacaEval) are rendered from whatever
JSONs exist. H1 uses a strict-self judge, H2 a fixed Tülu-3-RLVR judge;
the H5 frontier APIs are deployment-only, so their conversation rows stay
blank pending the API-as-judge evaluation.

Usage:
    python -m src.plotting.tables                      # -> drafts/results_tables_auto.tex
    python -m src.plotting.tables --results-dir DIR --out FILE
    python -m src.plotting.tables --no-heatmap
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

# --- display registries ------------------------------------------------
OPEN_MODELS = [
    ("llama3.1-8b-instruct", "Llama-3.1-8B"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B"),
    ("tulu3-8b-rlvr", "Tülu-3-8B-RLVR"),
    ("qwen2.5-72b-instruct", "Qwen2.5-72B"),
]
API_MODELS = [
    ("gpt-5.2-chat", "GPT-5.2"),
    ("deepseek-v4-flash", "DeepSeek-V4-Flash"),
]
H1_TASK_BANDS = [
    ("Conversation", [("ultrafeedback", "UltraFeedback"), ("alpaca_eval", "AlpacaEval")]),
    ("Development", [("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval")]),
    ("Deployment", [("matharena", "MathArena"), ("livecodebench", "LiveCodeBench")]),
]
H2_DATASETS = [
    ("ultrafeedback", "UltraFeedback"), ("alpaca_eval", "AlpacaEval"),
    ("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval"),
    ("matharena", "MathArena"), ("livecodebench", "LiveCodeBench"),
]
H2_STAGES = [("sft", "SFT"), ("dpo", "DPO"), ("rlvr", "RLVR")]
# H3 temperature table: open models x all datasets x {greedy, 0.7, 1.0}.
H3_TEMP_MODELS = [
    ("llama3.1-8b-instruct", "Llama-3.1-8B"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B"),
    ("tulu3-8b-rlvr", "Tülu-3-8B-RLVR"),
]
H3_TEMP_DATASETS = [
    ("ultrafeedback", "UltraFeedback"), ("alpaca_eval", "AlpacaEval"),
    ("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval"),
    ("matharena", "MathArena"), ("livecodebench", "LiveCodeBench"),
]
H3_TAUS = [(0.0, r"$\tau{=}0$"), (0.7, r"$\tau{=}0.7$"), (1.0, r"$\tau{=}1$")]
H4_MODEL = "tulu3-8b-rlvr"
H4_DATASETS = [("gsm8k", "GSM8K"), ("math", "MATH"), ("matharena", "MathArena")]
H4_LS = [0, 64, 128, 256, 512, 1024, 2048]
H5_DEPLOY = [("matharena", "MathArena"), ("livecodebench", "LiveCodeBench")]
H5_CONV = [("ultrafeedback", "UltraFeedback"), ("alpaca_eval", "AlpacaEval")]

_EPS = 1e-9


@dataclass(frozen=True)
class Cell:
    reu: float
    aeu: float
    rvr: float
    ci_half: float


def _load_dir(results_dir: Path, sub: str, seed: int) -> list[dict]:
    d = results_dir / sub
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
        out.append(rec)
    return out


def _to_cell(rec: dict) -> Cell:
    a = rec["aggregates_at_K_max"]
    ci = rec.get("bootstrap_R_hat_K_at_K_max", {})
    lo, hi = ci.get("ci_low"), ci.get("ci_high")
    half = (hi - lo) / 2.0 if (lo is not None and hi is not None) else 0.0
    return Cell(float(a["U_circ_K"]), float(a["U_bar_K"]), float(a["R_hat_K"]), half)


def _index(records: list[dict], key) -> dict:
    """Map key(rec) -> Cell, last-write-wins."""
    out = {}
    for rec in records:
        try:
            k = key(rec)
        except KeyError:
            continue
        if k is None:
            continue
        out[k] = _to_cell(rec)
    return out


# --- number / cell formatting -----------------------------------------
def _fmt(x: float, dp: int) -> str:
    return f"{x:.{dp}f}"


def _rvr_str(cell: Cell, *, dp: int, with_ci: bool, bold: bool,
             stacked: bool = False) -> str:
    """RVR cell. ``stacked`` puts the CI on a second line under the mean."""
    mean = _fmt(cell.rvr, dp)
    if bold:
        mean = r"\best{" + mean + "}"
    if not with_ci:
        return mean
    ci = _fmt(cell.ci_half, dp)
    if stacked:
        return r"\stk{" + mean + "}{" + ci + "}"
    return mean + r"\ci{" + ci + "}"


def _extreme_idx(values: list[Optional[float]], mode: str) -> set[int]:
    """Indices of the min/max non-None value (ties all marked)."""
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if not present:
        return set()
    ref = (min if mode == "min" else max)(v for _, v in present)
    return {i for i, v in present if abs(v - ref) < _EPS}


# --- table builders ----------------------------------------------------
def build_h1(results_dir: Path, seed: int, dp: int) -> str:
    cells = _index(_load_dir(results_dir, "h1", seed),
                   lambda r: (r["model"], r["dataset"]))
    models = OPEN_MODELS
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{H1 — existence of the rational value risk across "
        r"open models.} REU $=\mathbb{E}_{o\sim P(\cdot\mid x,y^\circ)}U(o)$; "
        r"AEU $=\mathbb{E}_{y\sim d_\theta,\,o\sim P(\cdot\mid x,y)}U(o)$; "
        r"RVR $=$ REU $-$ AEU. $K{=}64$, seed~" + str(seed) + r". RVR shows "
        r"mean $\pm$ 95\% prompt-bootstrap half-width. Conversation tasks "
        r"(UltraFeedback, AlpacaEval) use a strict-self LLM-as-judge. "
        r"\best{Bold} marks the smallest RVR per row. Blank cells (\na) are "
        r"the pending Qwen2.5-72B panel.}",
        r"\label{tab:h1}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll *{" + str(len(models)) + r"}{cc c}}",
        r"\toprule",
    ]
    head1 = " & " + "".join(
        r" & \multicolumn{3}{c}{" + disp + "}" for _, disp in models
    ) + r" \\"
    lines.append(head1)
    cmid = "".join(
        r"\cmidrule(lr){%d-%d}" % (3 + 3 * i, 5 + 3 * i) for i in range(len(models))
    )
    lines.append(cmid)
    lines.append("Task & Dataset" + " & REU & AEU & RVR" * len(models) + r" \\")
    lines.append(r"\midrule")

    n_bands = len(H1_TASK_BANDS)
    for bi, (band, datasets) in enumerate(H1_TASK_BANDS):
        for di, (ds, ds_disp) in enumerate(datasets):
            rvrs = [cells.get((m, ds)).rvr if (m, ds) in cells else None
                    for m, _ in models]
            bold_set = _extreme_idx(rvrs, "min")
            cellstrs = []
            for mi, (m, _) in enumerate(models):
                c = cells.get((m, ds))
                if c is None:
                    cellstrs.append(r"\na & \na & \na")
                else:
                    cellstrs.append(
                        f"{_fmt(c.reu, dp)} & {_fmt(c.aeu, dp)} & "
                        + _rvr_str(c, dp=dp, with_ci=True, bold=(mi in bold_set))
                    )
            prefix = (r"\multirow{%d}{*}{%s}" % (len(datasets), band)) if di == 0 else ""
            lines.append(f"{prefix}\n & {ds_disp} & " + " & ".join(cellstrs) + r" \\")
        if bi < n_bands - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
    return "\n".join(lines)


def build_h2(results_dir: Path, seed: int, dp: int) -> str:
    cells = _index(_load_dir(results_dir, "h2", seed),
                   lambda r: (r.get("trajectory_stage"), r["dataset"]))
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{H2 — alignment trajectory of the Tülu-3-8B "
        r"pipeline.} Along SFT$\to$DPO$\to$RLVR, AEU rises while REU stays "
        r"flat, so RVR shrinks but never closes. $K{=}64$, seed~" + str(seed)
        + r"; RVR shows mean $\pm$ 95\% bootstrap half-width. \best{Bold} "
        r"marks the smallest RVR per dataset. The base (few-shot) stage is "
        r"omitted; conversation tasks are scored by a fixed Tülu-3-RLVR "
        r"judge held constant across stages.}",
        r"\label{tab:h2}",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"Dataset & & SFT & DPO & RLVR \\",
        r"\midrule",
    ]
    for i, (ds, ds_disp) in enumerate(H2_DATASETS):
        reu = [cells.get((st, ds)) for st, _ in H2_STAGES]
        rvrs = [c.rvr if c else None for c in reu]
        bold_set = _extreme_idx(rvrs, "min")
        def row(metric_fn, is_rvr=False):
            out = []
            for si, c in enumerate(reu):
                if c is None:
                    out.append(r"\na")
                elif is_rvr:
                    out.append(_rvr_str(c, dp=dp, with_ci=True, bold=(si in bold_set),
                                        stacked=True))
                else:
                    out.append(_fmt(metric_fn(c), dp))
            return " & ".join(out)
        lines.append(r"\multirow{3}{*}{%s}" % ds_disp)
        lines.append(r" & REU & " + row(lambda c: c.reu) + r" \\")
        lines.append(r" & AEU & " + row(lambda c: c.aeu) + r" \\")
        lines.append(r" & RVR & " + row(None, is_rvr=True) + r" \\")
        if i < len(H2_DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_h3_temp(results_dir: Path, seed: int, dp: int) -> str:
    """H3 temperature table: rows = (Dataset, Model) x {REU,AEU,RVR},
    columns = greedy / tau=0.7 / tau=1.0."""
    direct = [r for r in _load_dir(results_dir, "h3", seed)
              if (r.get("procedure", "direct") == "direct" or "tau" in r)]
    cells = {}
    for r in direct:
        if "tau" not in r:
            continue
        cells[(r["model"], r["dataset"], round(float(r["tau"]), 3))] = _to_cell(r)

    n_metrics = 3
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{H3 — rational value risk across decoding "
        r"temperatures} ($K{=}64$, seed~" + str(seed) + r"). Greedy "
        r"($\tau{=}0$) is deterministic so REU${=}$AEU and RVR${=}0$ by "
        r"construction; sampling opens the gap. RVR shows mean $\pm$ 95\% "
        r"bootstrap half-width. Self-consistency is reported separately "
        r"(Fig.~\ref{fig:h3sc}), as it requires an extractable answer.}",
        r"\label{tab:h3temp}",
        r"\begin{tabular}{lll ccc}",
        r"\toprule",
        r"Dataset & Model & & " + " & ".join(t for _, t in H3_TAUS) + r" \\",
        r"\midrule",
    ]
    metrics = [("REU", "reu"), ("AEU", "aeu"), ("RVR", "rvr")]
    for di, (ds, ds_disp) in enumerate(H3_TEMP_DATASETS):
        ds_span = len(H3_TEMP_MODELS) * n_metrics
        first_ds = True
        for mi, (m, m_disp) in enumerate(H3_TEMP_MODELS):
            rvrs = [cells[(m, ds, tau)].rvr if (m, ds, tau) in cells else None
                    for tau, _ in H3_TAUS]
            # bold the smallest *positive* RVR (greedy's 0 is trivial)
            pos = [(j, rvrs[j]) for j in range(len(rvrs)) if rvrs[j] and rvrs[j] > 0]
            bold_set = ({min(pos, key=lambda t: t[1])[0]} if pos else set())
            for mj, (mlabel, attr) in enumerate(metrics):
                cellvals = []
                for ti, (tau, _) in enumerate(H3_TAUS):
                    c = cells.get((m, ds, tau))
                    if c is None:
                        cellvals.append(r"\na")
                    elif attr == "rvr":
                        cellvals.append(_rvr_str(c, dp=dp, with_ci=True,
                                                 bold=(ti in bold_set)))
                    else:
                        cellvals.append(_fmt(getattr(c, attr), dp))
                ds_cell = (r"\multirow{%d}{*}{%s}" % (ds_span, ds_disp)
                           if first_ds else "")
                m_cell = (r"\multirow{%d}{*}{%s}" % (n_metrics, m_disp)
                          if mj == 0 else "")
                lines.append(f"{ds_cell} & {m_cell} & {mlabel} & "
                             + " & ".join(cellvals) + r" \\")
                first_ds = False
            if mi < len(H3_TEMP_MODELS) - 1:
                lines.append(r"\cmidrule(l){2-6}")
        if di < len(H3_TEMP_DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_h4(results_dir: Path, seed: int, dp: int, heatmap: bool = True) -> str:
    cells = _index(
        [r for r in _load_dir(results_dir, "h4", seed) if r.get("model") == H4_MODEL],
        lambda r: (r["dataset"], int(r["L"])),
    )
    # Colour-scale ranges: REU/AEU share the utility range (same quantity);
    # RVR is scaled on its own range. Intensity is linear in the value.
    util_vals, rvr_vals = [], []
    for ds, _ in H4_DATASETS:
        for L in H4_LS:
            c = cells.get((ds, L))
            if c is not None:
                util_vals += [c.reu, c.aeu]
                rvr_vals.append(c.rvr)
    uvmin, uvmax = (min(util_vals), max(util_vals)) if util_vals else (0.0, 1.0)
    rvmin, rvmax = (min(rvr_vals), max(rvr_vals)) if rvr_vals else (0.0, 1.0)

    def heat(v: float, vmin: float, vmax: float, base: str, maxpct: int = 55) -> str:
        if not heatmap:
            return ""
        t = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        return r"\cellcolor{%s!%d}" % (base, round(t * maxpct))

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{H4 — rational value risk vs.\ reasoning-length "
        r"budget $L$} (Tülu-3-8B-RLVR, two-stage budget forcing, $K{=}64$, "
        r"seed~" + str(seed) + r"). As $L$ grows, REU and AEU rise and "
        r"converge; RVR peaks at an intermediate budget and then closes. RVR "
        r"shows mean $\pm$ 95\% bootstrap half-width; \best{bold} marks the "
        r"peak RVR per dataset. Cell shading is linear in the value (REU/AEU "
        r"teal, RVR orange).}",
        r"\label{tab:h4}",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"Dataset & $L$ & REU & AEU & RVR \\",
        r"\midrule",
    ]
    for di, (ds, ds_disp) in enumerate(H4_DATASETS):
        series = [cells.get((ds, L)) for L in H4_LS]
        rvrs = [c.rvr if c else None for c in series]
        bold_set = _extreme_idx(rvrs, "max")
        for li, L in enumerate(H4_LS):
            c = series[li]
            head = r"\multirow{%d}{*}{%s}" % (len(H4_LS), ds_disp) if li == 0 else ""
            if c is None:
                body = r"\na & \na & \na"
            else:
                reu_c = heat(c.reu, uvmin, uvmax, "teal")
                aeu_c = heat(c.aeu, uvmin, uvmax, "teal")
                rvr_c = heat(c.rvr, rvmin, rvmax, "orange")
                body = (f"{reu_c}{_fmt(c.reu, dp)} & {aeu_c}{_fmt(c.aeu, dp)} & "
                        f"{rvr_c}" + _rvr_str(c, dp=dp, with_ci=True, bold=(li in bold_set)))
            lines.append(f"{head} & {L} & {body} \\\\")
        if di < len(H4_DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_h5(results_dir: Path, seed: int, dp: int) -> str:
    cells = _index(_load_dir(results_dir, "h1", seed),
                   lambda r: (r["model"], r["dataset"]))
    models = API_MODELS
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{H5 — rational value risk in frontier proprietary "
        r"LLMs} on the contamination-resistant deployment datasets ($K{=}64$, "
        r"seed~" + str(seed) + r"). The gap persists even for the strongest "
        r"available models. RVR shows mean $\pm$ 95\% bootstrap half-width; "
        r"\best{bold} marks the smaller RVR per dataset. The frontier APIs are "
        r"evaluated on the deployment datasets only; conversation rows (\na) "
        r"await the API-as-judge evaluation.}",
        r"\label{tab:h5}",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"Dataset & Model & REU & AEU & RVR \\",
        r"\midrule",
    ]
    groups = H5_DEPLOY + H5_CONV
    for gi, (ds, ds_disp) in enumerate(groups):
        rvrs = [cells.get((m, ds)).rvr if (m, ds) in cells else None
                for m, _ in models]
        bold_set = _extreme_idx(rvrs, "min")
        for mi, (m, m_disp) in enumerate(models):
            c = cells.get((m, ds))
            head = r"\multirow{%d}{*}{%s}" % (len(models), ds_disp) if mi == 0 else ""
            if c is None:
                body = r"\na & \na & \na"
            else:
                body = (f"{_fmt(c.reu, dp)} & {_fmt(c.aeu, dp)} & "
                        + _rvr_str(c, dp=dp, with_ci=True, bold=(mi in bold_set),
                                   stacked=True))
            lines.append(f"{head} & {m_disp} & {body} \\\\")
        if gi < len(groups) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# A_K + RVR summary tables: rows = dataset (banded), cols = model x {A_K, RVR}.
AKRVR_TASK_BANDS = [
    ("Conversation", [("ultrafeedback", "UltraFeedback"), ("alpaca_eval", "AlpacaEval")]),
    ("Development", [("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval")]),
    ("Deployment", [("matharena", "MathArena"), ("livecodebench", "LiveCB")]),
]
# Per-dataset challenge level (qualitative tier; edit to taste).
CHALLENGE = {
    "ultrafeedback": "Open", "alpaca_eval": "Open",
    "gsm8k": "Easy", "math": "Medium", "humaneval": "Easy",
    "matharena": "Expert", "livecodebench": "Hard",
}
AKRVR_OPEN = [
    ("tulu3-8b-rlvr", "Tülu-3-8B-RLVR"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B"),
]
AKRVR_API = [
    ("gpt-5.2-chat", "GPT-5.2"),
    ("deepseek-v4-flash", "DeepSeek-V4-Flash"),
    ("gpt-5.5-chat", "GPT-5.5"),     # not run yet -> blank
]


def build_ak_rvr(results_dir: Path, seed: int, dp: int, models, *,
                 caption: str, label: str, heatmap: bool = True,
                 bands=None) -> str:
    """Rows = dataset (banded) with a challenge-level column; columns =
    model x {A_K, RVR}. A_K = 1 - REU (compute-approximation floor); RVR =
    R_hat_K. Heatmap: A_K blue, RVR orange (darker = larger). ``bands``
    selects which task bands to include (default: all)."""
    bands = bands if bands is not None else AKRVR_TASK_BANDS
    cells = _index(_load_dir(results_dir, "h1", seed),
                   lambda r: (r["model"], r["dataset"]))

    def heat(v: float, base: str) -> str:
        if not heatmap:
            return ""
        t = max(0.0, min(1.0, v))
        return r"\cellcolor{%s!%d}" % (base, round(t * 55))

    n = len(models)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll *{" + str(n) + r"}{cc}}",
        r"\toprule",
    ]
    hdr = " & "
    cmids = []
    col = 3
    for _, disp in models:
        hdr += r" & \multicolumn{2}{c}{%s}" % disp
        cmids.append(r"\cmidrule(lr){%d-%d}" % (col, col + 1))
        col += 2
    lines.append(hdr + r" \\")
    lines.append("".join(cmids))
    lines.append("Dataset & Level" + r" & $A_K$ & RVR" * n + r" \\")
    lines.append(r"\midrule")

    for bi, (band, dsets) in enumerate(bands):
        for ds, ds_disp in dsets:
            row = []
            for m, _ in models:
                c = cells.get((m, ds))
                if c is None:
                    row.append(r"\na & \na")
                else:
                    a = 1.0 - c.reu
                    row.append(heat(a, "blue") + _fmt(a, dp) + " & "
                               + heat(c.rvr, "orange") + _fmt(c.rvr, dp))
            lvl = CHALLENGE.get(ds, "")
            lines.append(f"{ds_disp} & {lvl} & " + " & ".join(row) + r" \\")
        if bi < len(bands) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


_PREAMBLE = r"""% AUTO-GENERATED by `python -m src.plotting.tables` — do not edit by hand.
% Re-run the generator to refresh from the result JSONs.
% Requires in the main preamble: booktabs, multirow, graphicx, amsmath, amssymb,
% and (for the H4 heatmap cell shading) \usepackage[table]{xcolor}.
\providecommand{\best}[1]{\textbf{#1}}
\providecommand{\na}{\textendash}
\providecommand{\ci}[1]{{\scriptsize$\,\pm#1$}}
\providecommand{\stk}[2]{\shortstack{#1 \\ {\scriptsize$\pm#2$}}}
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LaTeX results tables (H1/H2/H4/H5).")
    p.add_argument("--results-dir", default=None,
                   help="Directory containing h1/ h2/ h4/ subdirs. "
                        "Defaults to load_paths().results_dir.")
    p.add_argument("--out", default=str(_REPO_ROOT / "drafts" / "results_tables_auto.tex"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-heatmap", action="store_true",
                   help="Disable H4 value-mapped cell shading (no xcolor dependency).")
    p.add_argument("--decimals", type=int, default=3)
    args = p.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser()
    else:
        from src.pipeline.paths import load_paths
        results_dir = load_paths().results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"results dir not found: {results_dir}")

    dp = args.decimals
    blocks = [
        _PREAMBLE,
        build_h1(results_dir, args.seed, dp),
        build_h2(results_dir, args.seed, dp),
        # H3 temperature is now a figure (plot_h3.plot_temperature); the
        # build_h3_temp() table is kept available but not emitted by default.
        build_h4(results_dir, args.seed, dp, heatmap=not args.no_heatmap),
        build_h5(results_dir, args.seed, dp),
        build_ak_rvr(
            results_dir, args.seed, dp, AKRVR_OPEN,
            caption=(r"\textbf{Compute-approximation error $A_K$ vs.\ rational "
                     r"value risk RVR (7--8B models).} $A_K = 1-\text{REU}$ is the "
                     r"unreachable mass (no finite budget closes it); RVR $=$ "
                     r"REU$-$AEU is the reachable-but-unconcentrated gap. $K{=}64$ "
                     r"(conversation $K{=}32$), seed~0. Cell shading is linear in "
                     r"the value (darker $=$ larger)."),
            label="tab:akrvr_open", heatmap=not args.no_heatmap),
        build_ak_rvr(
            results_dir, args.seed, dp, AKRVR_API,
            caption=(r"\textbf{Compute-approximation error $A_K$ vs.\ RVR "
                     r"(frontier APIs).} Frontier APIs are evaluated on the "
                     r"deployment datasets only. $A_K = 1-\text{REU}$; RVR $=$ "
                     r"REU$-$AEU; $K{=}64$, seed~0. GPT-5.5 pending."),
            label="tab:akrvr_api", heatmap=not args.no_heatmap,
            bands=[AKRVR_TASK_BANDS[2]]),   # Deployment only
    ]
    text = "\n\n".join(blocks) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out}  (results_dir={results_dir}, seed={args.seed})")


if __name__ == "__main__":
    main()
