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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Parse dataset_id (incl. any judge suffix) from the result-JSON filename.
# Filename pattern: "<model_alias>_<dataset_id>_seed<N>.json", where
# dataset_id may itself be e.g. "ultrafeedback_judge-qwen2.5-14b-instruct".
_SEED_RE = re.compile(r"^(?P<base>.+)_seed\d+$")


def _parse_dataset_id(stem: str, model: str) -> Optional[str]:
    m = _SEED_RE.match(stem)
    if not m:
        return None
    base = m.group("base")
    if base.startswith(model + "_"):
        return base[len(model) + 1:]
    return None


# Model alias map for cell lookup: canonical -> fallback aliases. Used for
# the 72B humaneval cell that exists only as the AWQ-quantised variant.
MODEL_ALIASES: dict[str, list[str]] = {
    "qwen2.5-72b-instruct": ["qwen2.5-72b-instruct-awq"],
}

# Per-(model, dataset_id) dataset alias: when the canonical file doesn't
# exist for a given model, fall back to a different judge variant. Used so
# the 72B's conversation "(Qwen-14B judge)" rows are filled from the
# DeepSeek-V4-Flash-judge files (the actual judge used for the 72B).
PER_MODEL_DATASET_ALIASES: dict[tuple[str, str], str] = {
    ("qwen2.5-72b-instruct", "ultrafeedback_judge-qwen2.5-14b-instruct"):
        "ultrafeedback_judge-deepseek-v4-flash",
    ("qwen2.5-72b-instruct", "alpaca_eval_judge-qwen2.5-14b-instruct"):
        "alpaca_eval_judge-deepseek-v4-flash",
}


def _lookup(cells, model: str, dataset_id: str):
    if (model, dataset_id) in cells:
        return cells[(model, dataset_id)]
    for alias in MODEL_ALIASES.get(model, []):
        if (alias, dataset_id) in cells:
            return cells[(alias, dataset_id)]
    alt_ds = PER_MODEL_DATASET_ALIASES.get((model, dataset_id))
    if alt_ds is not None:
        if (model, alt_ds) in cells:
            return cells[(model, alt_ds)]
        for alias in MODEL_ALIASES.get(model, []):
            if (alias, alt_ds) in cells:
                return cells[(alias, alt_ds)]
    return None

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
# H1 task bands: Conversation / Math / Coding.
H1_TASK_BANDS = [
    ("Conversation", [
        ("ultrafeedback", "UltraFeedback (self-as-verifier)"),
        ("ultrafeedback_judge-qwen2.5-14b-instruct", "UltraFeedback (external verifier)"),
        ("alpaca_eval", "AlpacaEval (self-as-verifier)"),
        ("alpaca_eval_judge-qwen2.5-14b-instruct", "AlpacaEval (external verifier)"),
    ]),
    ("Math", [("gsm8k", "GSM8K"), ("math", "MATH"), ("matharena", "MathArena")]),
    ("Coding", [("humaneval", "HumanEval"), ("livecodebench", "LiveCB")]),
]
H2_DATASETS = [
    ("ultrafeedback_judge-qwen2.5-14b-instruct", "UltraFeedback (external verifier)"),
    ("alpaca_eval_judge-qwen2.5-14b-instruct", "AlpacaEval (external verifier)"),
    ("gsm8k", "GSM8K"), ("math", "MATH"), ("humaneval", "HumanEval"),
    ("livecodebench", "LiveCB"),
]
H2_STAGES = [("sft", "SFT"), ("dpo", "DPO"), ("rlvr", "RLVR")]
# Tulu-70B trajectory (server B; deployment only). Cells may still be
# landing — missing entries render as \na.
H2_70B_STAGES = [
    ("tulu3-70b-sft", "SFT"),
    ("tulu3-70b-dpo", "DPO"),
    ("tulu3-70b-rlvr", "RLVR"),
]
H2_70B_DEPLOY_DATASETS = [("matharena", "MathArena"), ("livecodebench", "LiveCB")]
# H2 MathArena trajectory table: outer rows = Model (8B / 70B);
# inner rows = Stage (SFT / DPO / RLVR).
H2_MATHARENA_MODELS = [
    ("Tülu-3-8B",  [("SFT", "tulu3-8b-sft"),
                    ("DPO", "tulu3-8b-dpo"),
                    ("RLVR", "tulu3-8b-rlvr")]),
    ("Tülu-3-70B", [("SFT", "tulu3-70b-sft"),
                    ("DPO", "tulu3-70b-dpo"),
                    ("RLVR", "tulu3-70b-rlvr")]),
]
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
H4_MODEL = "tulu3-8b-rlvr"  # legacy single-model alias (still used elsewhere)
H4_MODELS = [
    ("tulu3-8b-rlvr",       "Tülu-3-8B-RLVR"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B-Instruct"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct"),
]
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
        # Attach filename-derived dataset id (preserves judge suffixes like
        # "ultrafeedback_judge-qwen2.5-14b-instruct"). Falls back to the
        # JSON's bare 'dataset' field for files without a judge suffix.
        rec["_dataset_id"] = (_parse_dataset_id(f.stem, rec.get("model", ""))
                              or rec.get("dataset"))
        rec["_filename"] = f.name
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


def _fmt_pct(x: float, dp: int = 1) -> str:
    """Render a fraction in [0,1] as a percentage (e.g. 0.365 -> '36.5\\%')."""
    return f"{x * 100:.{dp}f}" + r"\%"


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
def build_h1(results_dir: Path, seed: int, dp: int,
             heatmap: bool = False,
             *, caption: str | None = None, label: str = "tab:h1") -> str:
    cells = _index(_load_dir(results_dir, "h1", seed),
                   lambda r: (r["model"], r["_dataset_id"]))
    models = OPEN_MODELS
    if caption is None:
        caption = (r"Rational value risk across open language models on "
                   r"conversational and development benchmarks. Compute "
                   r"budget $K{=}64$, values are reported as mean $\pm$ 95\% "
                   r"bootstrap confidence interval. Bold indicates the "
                   r"smallest RVR per dataset. 7--8B models are judged by "
                   r"Qwen2.5-14B-Instruct; Qwen2.5-72B is judged by "
                   r"DeepSeek-V4-Flash.")
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
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
            row_cells = [_lookup(cells, m, ds) for m, _ in models]
            rvrs = [c.rvr if c is not None else None for c in row_cells]
            bold_set = _extreme_idx(rvrs, "min")
            def _tint(base, v):
                return _shade(base, v, heatmap)
            cellstrs = []
            for mi, c in enumerate(row_cells):
                if c is None:
                    cellstrs.append(r"\na & \na & \na")
                else:
                    cellstrs.append(
                        _tint("colREU", c.reu) + _fmt(c.reu, dp) + " & "
                        + _tint("colAEU", c.aeu) + _fmt(c.aeu, dp) + " & "
                        + _tint("colRVR", c.rvr)
                        + _rvr_str(c, dp=dp, with_ci=True, bold=(mi in bold_set))
                    )
            prefix = (r"\multirow{%d}{*}{%s}" % (len(datasets), band)) if di == 0 else ""
            lines.append(f"{prefix}\n & {ds_disp} & " + " & ".join(cellstrs) + r" \\")
        if bi < n_bands - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
    return "\n".join(lines)


def build_h2(results_dir: Path, seed: int, dp: int, heatmap: bool = True) -> str:
    cells = _index(_load_dir(results_dir, "h2", seed),
                   lambda r: (r.get("trajectory_stage"), r["_dataset_id"]))

    # Anchor colours per metric with value-proportional intensity.
    def tint(base, v):
        return _shade(base, v, heatmap)

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Rational value risk of T\"ulu-3-8B family across SFT, DPO, "
        r"and RLVR stages. Compute budget $K{=}64$; values are reported as "
        r"mean $\pm$ 95\% bootstrap confidence interval.}",
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

        def row(attr, is_rvr=False):
            out = []
            for si, c in enumerate(reu):
                if c is None:
                    out.append(r"\na")
                elif is_rvr:
                    out.append(tint("colRVR", c.rvr)
                               + _rvr_str(c, dp=dp, with_ci=True,
                                          bold=False, stacked=True))
                else:
                    v = getattr(c, attr)
                    base = "colREU" if attr == "reu" else "colAEU"
                    out.append(tint(base, v) + _fmt(v, dp))
            return " & ".join(out)
        disp = _label_with_judge_break(ds_disp, True)
        lines.append(r"\multirow{3}{*}{%s}" % disp)
        lines.append(r" & REU & " + row("reu") + r" \\")
        lines.append(r" & AEU & " + row("aeu") + r" \\")
        lines.append(r" & RVR & " + row(None, is_rvr=True) + r" \\")
        if i < len(H2_DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_h2_matharena(results_dir: Path, seed: int, dp: int,
                       heatmap: bool = True) -> str:
    """H2 trajectory on MathArena — one row per (stage, model), with both
    the 8B and 70B Tulu checkpoints. Columns: Stage, Model, 1-REU, RVR,
    %RVR. 70B rows render blank until the trajectory completes."""
    cells = _index(_load_dir(results_dir, "h2", seed),
                   lambda r: (r["model"], r["_dataset_id"]))

    def tint(base, v):
        return _shade(base, v, heatmap)

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Compute-approximation error $1{-}\text{REU}$, rational "
        r"value risk RVR, and recoverable fraction $\%\text{RVR} = "
        r"\dfrac{\text{RVR}}{(1-\text{REU})+\text{RVR}}$ along the "
        r"T\"ulu trajectory on MathArena (8B and 70B per stage). 70B rows "
        r"are pending.}",
        r"\label{tab:h2_matharena}",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"Model & Stage & $1{-}\text{REU}$ & RVR & \%RVR \\",
        r"\midrule",
    ]
    n_stages = max(len(stages) for _, stages in H2_MATHARENA_MODELS)
    for mi, (model_disp, stages) in enumerate(H2_MATHARENA_MODELS):
        for si, (stage_disp, model_id) in enumerate(stages):
            c = _lookup(cells, model_id, "matharena")
            model_cell = (r"\multirow{%d}{*}{%s}" % (n_stages, model_disp)) if si == 0 else ""
            if c is None:
                body = " & ".join([r"\na"] * 3)
            else:
                shortfall = 1.0 - c.reu
                rvr = c.rvr
                denom = shortfall + rvr
                pct = rvr / denom if denom > 0 else 0.0
                body = (tint("colAK", shortfall) + _fmt(shortfall, dp) + " & "
                        + tint("colRVR", rvr) + _fmt(rvr, dp) + " & "
                        + tint("colPCTRVR", pct) + _fmt_pct(pct))
            lines.append(f"{model_cell} & {stage_disp} & {body} \\\\")
        if mi < len(H2_MATHARENA_MODELS) - 1:
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
    """H4 budget-forced table with model as the outer column dimension.

    Rows: (dataset, $T$) where $T \\in$ ``H4_LS`` is the forced
    reasoning-length budget. Columns: 3 models $\\times$ \\{REU, AEU,
    RVR\\}. Cells are auto-coloured (``colREU`` / ``colAEU`` /
    ``colRVR``) with intensity proportional to value; the per-dataset
    max-RVR per model is marked in \\textbf{bold}.
    """
    cells = _index(
        [r for r in _load_dir(results_dir, "h4", seed)
         if any(r.get("model") == m for m, _ in H4_MODELS)],
        lambda r: (r["model"], r["dataset"], int(r["L"])),
    )

    def tint(base: str, v: float) -> str:
        return _shade(base, v, heatmap)

    n_models = len(H4_MODELS)
    col_spec = "ll " + " ".join(["ccc"] * n_models)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Rational value risk under varying reasoning-length "
        r"budgets $T$, across three open-weight 8B-scale models. "
        r"Columns are grouped by model; each group reports REU, AEU, "
        r"and RVR (mean $\pm$ 95\% bootstrap half-width). Cells are "
        r"auto-coloured by value (\texttt{colREU} / \texttt{colAEU} / "
        r"\texttt{colRVR}; intensity $\propto$ value); \textbf{bold} "
        r"marks the largest RVR per (dataset, model).}",
        r"\label{tab:h4}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{$T$} & "
        + " & ".join(r"\multicolumn{3}{c}{%s}" % disp
                     for _, disp in H4_MODELS) + r" \\",
        " ".join(
            r"\cmidrule(lr){%d-%d}" % (3 + 3 * j, 5 + 3 * j)
            for j in range(n_models)
        ),
        r" & & " + " & ".join(["REU & AEU & RVR"] * n_models) + r" \\",
        r"\midrule",
    ]
    for di, (ds, ds_disp) in enumerate(H4_DATASETS):
        # bold the max-RVR L per (model, dataset) so each model gets one bold row
        bold_sets = {}
        for m, _ in H4_MODELS:
            rvrs = [cells.get((m, ds, L)).rvr if (m, ds, L) in cells else None
                    for L in H4_LS]
            bold_sets[m] = _extreme_idx(rvrs, "max")
        for li, L in enumerate(H4_LS):
            head = (r"\multirow{%d}{*}{%s}" % (len(H4_LS), ds_disp)
                    if li == 0 else "")
            parts = [head, str(L)]
            for m, _ in H4_MODELS:
                c = cells.get((m, ds, L))
                if c is None:
                    parts.extend([r"\na", r"\na", r"\na"])
                else:
                    parts.append(tint("colREU", c.reu) + _fmt(c.reu, dp))
                    parts.append(tint("colAEU", c.aeu) + _fmt(c.aeu, dp))
                    rvr_cell = (tint("colRVR", c.rvr)
                                + _rvr_str(c, dp=dp, with_ci=True,
                                           bold=(li in bold_sets[m])))
                    parts.append(rvr_cell)
            lines.append(" & ".join(parts) + r" \\")
        if di < len(H4_DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
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
# Per-dataset challenge level (qualitative tier; edit to taste). Keyed on
# the base dataset; judge-suffixed ids fall back via _challenge().
CHALLENGE = {
    "ultrafeedback": "Open", "alpaca_eval": "Open",
    "gsm8k": "Easy", "math": "Medium", "humaneval": "Easy",
    "matharena": "Expert", "livecodebench": "Hard",
}


def _challenge(ds_id: str) -> str:
    base = ds_id.split("_judge-")[0]
    return CHALLENGE.get(base, "")


def _shade(base: str, value: float, heatmap: bool = True, maxpct: int = 60) -> str:
    """Anchor-coloured cell with intensity proportional to ``value`` in [0,1].
    Returns the empty string when ``heatmap`` is False (fully uncoloured)."""
    if not heatmap:
        return ""
    t = max(0.0, min(1.0, float(value)))
    return r"\cellcolor{%s!%d}" % (base, round(t * maxpct))


def _label_with_judge_break(disp: str, two_line: bool) -> str:
    """Render '<X> (external verifier)' on two lines via \\shortstack so the
    column doesn't blow up width-wise. Pass two_line=False to keep it
    single-line (used for H1)."""
    tag = "(external verifier)"
    if two_line and tag in disp:
        base = disp.replace(" " + tag, "").strip()
        return r"\shortstack[l]{" + base + r"\\" + tag + "}"
    return disp


# Bands for the akrvr_open table: include a Conversation band with only the
# Qwen-14B-judge UF/AlpacaEval variants (the headline conversation results).
AKRVR_OPEN_BANDS = [
    ("Conversation", [
        ("ultrafeedback_judge-qwen2.5-14b-instruct", "UltraFeedback (external verifier)"),
        ("alpaca_eval_judge-qwen2.5-14b-instruct",   "AlpacaEval (external verifier)"),
    ]),
    AKRVR_TASK_BANDS[1],  # Development
    AKRVR_TASK_BANDS[2],  # Deployment
]
AKRVR_OPEN = [
    ("tulu3-8b-rlvr", "Tülu-3-8B-RLVR"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B"),
    ("qwen2.5-72b-instruct", "Qwen2.5-72B"),
]
AKRVR_API = [
    ("gpt-5.2-chat", "GPT-5.2"),
    ("deepseek-v4-flash", "DeepSeek-V4-Flash"),
]
# Banded by model-size class for the %RVR deployment table.
AKRVR_PCT_BANDS = [
    ("7--8B", [
        ("qwen2.5-7b-instruct", "Qwen2.5-7B"),
        ("tulu3-8b-rlvr", "Tülu-3-8B-RLVR"),
        ("llama3.1-8b-instruct", "Llama-3.1-8B"),
    ]),
    ("70--72B", [
        ("qwen2.5-72b-instruct", "Qwen2.5-72B"),
        ("tulu3-70b-rlvr", "Tülu-3-70B-RLVR"),
    ]),
    ("APIs", [
        ("deepseek-v4-flash", "DeepSeek-V4-Flash"),
        ("gpt-5.2-chat", "GPT-5.2"),
        ("gpt-5.5", "GPT-5.5"),
    ]),
]
AKRVR_PCT_DATASETS = [("matharena", "MathArena")]


def build_ak_rvr_pct(results_dir: Path, seed: int, dp: int, *,
                     caption: str, label: str, heatmap: bool = True) -> str:
    """Rows = (deployment dataset, model); columns = A_K, 1-REU, RVR, %RVR.
    %RVR = RVR / ((1-REU) + RVR) — fraction of the total gap to the
    ground-truth optimum that is reachable-but-unconcentrated (recoverable
    by better sampling) rather than unreachable (compute-irreducible).
    Reads h1/ (open models + APIs) AND h2/ so the 70B-RLVR trajectory
    endpoint is picked up."""
    recs = (_load_dir(results_dir, "h1", seed)
            + _load_dir(results_dir, "h2", seed))
    cells = _index(recs, lambda r: (r["model"], r["_dataset_id"]))

    def tint(base: str, v: float) -> str:
        return _shade(base, v, heatmap)

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"Size & Model & $1{-}\text{REU}$ & RVR & \%RVR \\",
        r"\midrule",
    ]
    # Only MathArena is used; iterate bands of (Size, [models]).
    ds, ds_disp = AKRVR_PCT_DATASETS[0]
    for bi, (band_disp, models) in enumerate(AKRVR_PCT_BANDS):
        for mi, (m, m_disp) in enumerate(models):
            c = _lookup(cells, m, ds)
            size_cell = (r"\multirow{%d}{*}{%s}" % (len(models), band_disp)) if mi == 0 else ""
            if c is None:
                cells_str = " & ".join([r"\na"] * 3)
            else:
                shortfall = 1.0 - c.reu               # = A_K, the unreachable mass
                rvr = c.rvr
                denom = shortfall + c.rvr
                pct = rvr / denom if denom > 0 else 0.0
                cells_str = (
                    tint("colAK", shortfall) + _fmt(shortfall, dp) + " & "
                    + tint("colRVR", rvr) + _fmt(rvr, dp) + " & "
                    + tint("colPCTRVR", pct) + _fmt_pct(pct)
                )
            lines.append(f"{size_cell} & {m_disp} & {cells_str} \\\\")
        if bi < len(AKRVR_PCT_BANDS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def build_ak_rvr(results_dir: Path, seed: int, dp: int, models, *,
                 caption: str, label: str, heatmap: bool = True,
                 bands=None, source_dir: str = "h1",
                 two_line_judge_label: bool = True) -> str:
    """Rows = dataset (banded) with a challenge-level column; columns =
    model x {A_K, RVR}. A_K = 1 - REU (compute-approximation floor); RVR =
    R_hat_K. Heatmap: A_K blue, RVR orange (darker = larger). ``bands``
    selects which task bands to include; ``source_dir`` chooses h1 (default)
    or h2 (for the trajectory variant)."""
    bands = bands if bands is not None else AKRVR_TASK_BANDS
    cells = _index(_load_dir(results_dir, source_dir, seed),
                   lambda r: (r["model"], r["_dataset_id"]))

    def tint(base: str, v: float) -> str:
        return _shade(base, v, heatmap)

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
                c = _lookup(cells, m, ds)
                if c is None:
                    row.append(r"\na & \na")
                else:
                    a = 1.0 - c.reu
                    row.append(tint("colAK", a) + _fmt(a, dp) + " & "
                               + tint("colRVR", c.rvr) + _fmt(c.rvr, dp))
            lvl = _challenge(ds)
            disp = _label_with_judge_break(ds_disp, two_line_judge_label)
            lines.append(f"{disp} & {lvl} & " + " & ".join(row) + r" \\")
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
% Anchor colours per term (used as fixed cell backgrounds; same colour
% across all tables and matched in the matplotlib figures).
\definecolor{colREU}{HTML}{8FC4DF}      % soft blue
\definecolor{colAEU}{HTML}{D6A77E}      % muted amber
\definecolor{colRVR}{HTML}{CC7E9A}      % dusty pink
\definecolor{colAK}{HTML}{A998DD}       % soft lavender
\definecolor{colPCTRVR}{HTML}{6F8FE8}
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
        build_h1(results_dir, args.seed, dp,
                 heatmap=not args.no_heatmap, label="tab:h1_coloured",
                 caption=(r"Rational value risk across open language models on "
                          r"conversational and development benchmarks "
                          r"(autocoloured by value: REU, AEU, and RVR each "
                          r"use a fixed hue with intensity proportional to "
                          r"the cell value). Compute budget $K{=}64$; values "
                          r"are reported as mean $\pm$ 95\% bootstrap "
                          r"confidence interval. Bold indicates the smallest "
                          r"RVR per dataset. 7--8B models are judged by "
                          r"Qwen2.5-14B-Instruct; Qwen2.5-72B is judged by "
                          r"DeepSeek-V4-Flash.")),
        # H2 main table is intentionally uncoloured; H2-MathArena (below)
        # keeps the value-shaded anchor heatmap.
        build_h2(results_dir, args.seed, dp, heatmap=False),
        build_h2_matharena(results_dir, args.seed, dp,
                          heatmap=not args.no_heatmap),
        # H3 temperature is now a figure (plot_h3.plot_temperature); the
        # build_h3_temp() table is kept available but not emitted by default.
        build_h4(results_dir, args.seed, dp, heatmap=not args.no_heatmap),
        # H5 is now the three A_K/RVR tables below (open / 72B traj / API);
        # the legacy build_h5() function is kept available but not emitted.
        build_ak_rvr(
            results_dir, args.seed, dp, AKRVR_OPEN,
            caption=(r"Compute-approximation error and rational value risk "
                     r"for open-weight models across conversational, "
                     r"development, and deployment benchmarks. "
                     r"7--8B models are judged by Qwen2.5-14B-Instruct; "
                     r"Qwen2.5-72B is judged by DeepSeek-V4-Flash."),
            label="tab:akrvr_open", heatmap=not args.no_heatmap,
            # Conversation (14B-judge variants) + Development + Deployment.
            bands=AKRVR_OPEN_BANDS),
        build_ak_rvr(
            results_dir, args.seed, dp, H2_70B_STAGES,
            caption=(r"Compute-approximation error and rational value risk "
                     r"for 72B Tulu models (SFT, DPO, RLVR) on deployment "
                     r"benchmarks."),
            label="tab:akrvr_70b_traj", heatmap=not args.no_heatmap,
            bands=[("Deployment", H2_70B_DEPLOY_DATASETS)],
            source_dir="h2"),
        build_ak_rvr(
            results_dir, args.seed, dp, AKRVR_API,
            caption=(r"Compute-approximation error and rational value risk "
                     r"for API models on deployment benchmarks."),
            label="tab:akrvr_api", heatmap=not args.no_heatmap,
            bands=[AKRVR_TASK_BANDS[2]]),   # Deployment only
        build_ak_rvr_pct(
            results_dir, args.seed, dp,
            caption=(r"Decomposition of total utility discrepancy between "
                     r"true answer and actual reasoning across all evaluated "
                     r"models on MathArena benchmark."),
            label="tab:akrvr_pct", heatmap=not args.no_heatmap),
    ]
    text = "\n\n".join(blocks) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out}  (results_dir={results_dir}, seed={args.seed})")


if __name__ == "__main__":
    main()
