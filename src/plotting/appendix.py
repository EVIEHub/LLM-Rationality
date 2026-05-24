"""Build LaTeX fragments for the paper appendix from the result JSONs.

Each ``build_*`` function returns a LaTeX string (no preamble), ready to be
``\\input``-ed from ``drafts/appendix.tex``. The orchestrator
:func:`scripts.build_appendix.main` calls them and writes per-section
fragments under ``drafts/appendix_inputs/``.

What's auto-filled here (no extra experiments needed):
  * B.4 — GPU-hour breakdown (sums ``sampling_seconds`` from each cell).
  * C.1 — Saturation-curve tables for the headline H1 cells.
  * C.3 — Bootstrap CI half-width vs $M$ (sub-sample the per-prompt array).
  * C.4 — Theorem-1 epsilon summary derived from the three terms above.
  * E.2 — Per-difficulty RVR breakdown (MATH level, LCB difficulty,
          HumanEval canonical-solution-length tercile). Reads the datasets
          via ``datasets`` (HF cache OK; mirror endpoint supported).

What still needs an extra run / re-verification (flagged here, the main
:mod:`scripts.build_appendix` records them in ``appendix_inputs/PENDING.md``):
  * C.2 — L sensitivity needs the raw L=5 verdicts stored in the audit log.
  * D.1 — MATH failure-mode rates need re-verification with an instrumented
          ``math_verify`` wrapper that records the failure reason.
  * D.2 — Position bias / inter-rater agreement need the raw verdict array
          from the self-judge audit log.
  * D.3 — GSM8K extractor pattern-firing rates need a re-extract pass that
          tags which regex fired.
  * G  — Failure-case traces are qualitative; :func:`build_G_candidates`
          dumps a list of (prompt_id, model, dataset) tuples that pass the
          spike-at-0 filter for hand-selection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.plotting.tables import _load_dir, _fmt


# ---- headline cells used by the saturation + epsilon tables -----------
HEADLINE_MODELS = [
    ("tulu3-8b-rlvr", "Tülu-3-8B-RLVR"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B-Instruct"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct"),
]
HEADLINE_DATASETS = [
    ("gsm8k", "GSM8K"),
    ("math", "MATH"),
    ("humaneval", "HumanEval"),
]


def _cell_index(results_dir: Path, seed: int = 0):
    """Index ALL h1 result JSONs by (model, dataset_id)."""
    out = {}
    for rec in _load_dir(results_dir, "h1", seed):
        out[(rec.get("model"), rec.get("_dataset_id"))] = rec
    return out


# ======================================================================
# B.4  GPU-hour breakdown
# ======================================================================
def build_B4_gpu_hours(results_dir: Path, seed: int = 0) -> str:
    """Sum ``sampling_seconds`` across cells, grouped by hypothesis dir."""
    groups: dict[str, list[float]] = {}
    for sub in ("h1", "h2", "h3", "h4"):
        for rec in _load_dir(results_dir, sub, seed):
            secs = rec.get("sampling_seconds") or 0.0
            groups.setdefault(sub, []).append(secs)

    rows = []
    total = 0.0
    for sub, secs_list in groups.items():
        n = len(secs_list)
        gh = sum(secs_list) / 3600.0
        total += gh
        rows.append((sub.upper(), n, gh))

    lines = [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{GPU-hour breakdown summed from each cell's "
        r"\texttt{sampling\_seconds}. Verification (CPU) and bootstrap "
        r"(CPU) wall-time are not included.}",
        r"\label{tab:appendix_gpu_hours}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Hypothesis & Cells & GPU-hr (sampling) \\", r"\midrule",
    ]
    for hyp, n, gh in rows:
        lines.append(f"{hyp} & {n} & {gh:.1f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total billable}} & {sum(n for _,n,_ in rows)} "
                 f"& {total:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# C.1  Saturation curve tables for the 3 headline H1 cells
# ======================================================================
def build_C1_saturation(results_dir: Path, seed: int = 0,
                        models=HEADLINE_MODELS,
                        datasets=HEADLINE_DATASETS) -> str:
    """Per-(model, dataset) saturation-curve table: 3 metrics x 7 $K$ values
    with the 95% prompt-bootstrap CI on RVR at $K_{\\max}$."""
    idx = _cell_index(results_dir, seed)

    lines = [
        r"\begin{table*}[h]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Saturation curves for the headline H1 cells. Each entry "
        r"is the prompt-mean at the given budget $K'$; the last column's "
        r"$\hat{\mathcal R}_K$ row carries the 95\% prompt-bootstrap CI "
        r"(B$=$1000 resamples).}",
        r"\label{tab:appendix_saturation}",
        r"\begin{tabular}{lll *{7}{r}}", r"\toprule",
    ]
    # Determine column grid from first available cell.
    K_grid = None
    for m, _ in models:
        for ds, _ in datasets:
            rec = idx.get((m, ds))
            if rec is not None:
                K_grid = [r["K"] for r in rec["saturation_curve"]]
                break
        if K_grid is not None: break
    if K_grid is None:
        K_grid = [1, 2, 4, 8, 16, 32, 64]
    lines.append(r"Model & Dataset & Metric & "
                 + " & ".join(f"$K{{=}}{k}$" for k in K_grid) + r" \\")
    lines.append(r"\midrule")

    for m, m_disp in models:
        for di, (ds, ds_disp) in enumerate(datasets):
            rec = idx.get((m, ds))
            if rec is None:
                continue
            curve = {r["K"]: r for r in rec["saturation_curve"]}
            ci = rec.get("bootstrap_R_hat_K_at_K_max", {})
            ci_str = ""
            if ci.get("ci_low") is not None and ci.get("ci_high") is not None:
                half = (ci["ci_high"] - ci["ci_low"]) / 2
                ci_str = f" $\\pm$ {half:.3f}"
            m_cell = (r"\multirow{%d}{*}{%s}" % (len(datasets) * 3, m_disp)) \
                     if di == 0 else ""
            ds_cell = r"\multirow{3}{*}{%s}" % ds_disp
            # 3 rows per (model, dataset)
            for ri, (label, key, with_ci) in enumerate([
                (r"$U^\circ_K$", "U_circ_K", False),
                (r"$\bar U_K$", "U_bar_K", False),
                (r"$\hat{\mathcal R}_K$", "R_hat_K", True),
            ]):
                vals = " & ".join(
                    _fmt(curve.get(k, {}).get(key, 0.0), 3) if k in curve else r"\na"
                    for k in K_grid
                )
                # only the RVR row at K_max gets the CI annotation
                if with_ci and ci_str:
                    # tack the CI onto the rightmost (K_max) column
                    vals = " & ".join(
                        (_fmt(curve.get(k, {}).get(key, 0.0), 3) +
                         (ci_str if (k == K_grid[-1]) else ""))
                        if k in curve else r"\na"
                        for k in K_grid
                    )
                head_m = m_cell if ri == 0 and di == 0 else ""
                head_ds = ds_cell if ri == 0 else ""
                lines.append(f"{head_m} & {head_ds} & {label} & {vals} \\\\")
            if di < len(datasets) - 1:
                lines.append(r"\cmidrule(l){2-%d}" % (3 + len(K_grid)))
        lines.append(r"\midrule")
    # strip trailing \midrule
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ======================================================================
# C.3  Bootstrap CI half-width vs M  (sub-sample per-prompt array)
# ======================================================================
def build_C3_M_convergence(results_dir: Path, seed: int = 0,
                           model: str = "tulu3-8b-rlvr",
                           dataset: str = "gsm8k",
                           m_grid=(50, 100, 200, 500, 1000),
                           B: int = 1000) -> str:
    rec = _cell_index(results_dir, seed).get((model, dataset))
    if rec is None or "per_prompt_R_hat_K" not in rec:
        return r"% C3: per-prompt array missing for chosen cell"
    arr = np.array(rec["per_prompt_R_hat_K"], dtype=float)
    M_full = len(arr)
    if M_full not in m_grid:
        m_grid = tuple(list(m_grid) + [M_full])

    rng = np.random.default_rng(seed)
    rows = []
    for M_p in m_grid:
        if M_p > M_full:
            continue
        idx_sub = rng.choice(M_full, size=M_p, replace=False)
        sub = arr[idx_sub]
        # B prompt-bootstrap resamples (size = M_p) -> bootstrap of mean
        boot = sub[rng.integers(0, M_p, size=(B, M_p))].mean(axis=1)
        lo, hi = np.quantile(boot, 0.025), np.quantile(boot, 0.975)
        half = (hi - lo) / 2
        rows.append((M_p, half))

    lines = [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{Prompt-bootstrap CI half-width on $\hat{\mathcal R}_K$ "
        r"shrinks as $1/\sqrt{M'}$ as the prompt sub-sample grows "
        r"(headline cell: " + model + r" / " + dataset
        + r", $K{=}64$, $B{=}1000$). Last column is the half-width "
        r"$\times\sqrt{M'}$; a near-constant ratio confirms the rate.}",
        r"\label{tab:appendix_M_convergence}",
        r"\begin{tabular}{rrr}", r"\toprule",
        r"$M'$ & CI half-width & $\sqrt{M'}\cdot$half-width \\",
        r"\midrule",
    ]
    for M_p, h in rows:
        lines.append(f"{M_p} & {h:.4f} & {h * (M_p ** 0.5):.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# C.4  Theorem-1 epsilon summary
# ======================================================================
def build_C4_epsilon(results_dir: Path, seed: int = 0) -> str:
    """The three Thm-1 error terms on the GSM8K Tülu-3-RLVR cell."""
    rec = _cell_index(results_dir, seed).get(("tulu3-8b-rlvr", "gsm8k"))
    if rec is None:
        return r"% C4: headline cell missing"
    curve = rec["saturation_curve"]
    K_max = max(r["K"] for r in curve)
    rvr_full = next(r["R_hat_K"] for r in curve if r["K"] == K_max)
    rvr_half = next((r["R_hat_K"] for r in curve if r["K"] == K_max // 2), rvr_full)
    truncation = abs(rvr_full - rvr_half)
    ci = rec.get("bootstrap_R_hat_K_at_K_max", {})
    if ci.get("ci_low") is not None and ci.get("ci_high") is not None:
        prompt_var = (ci["ci_high"] - ci["ci_low"]) / 2
    else:
        prompt_var = float("nan")
    eps = max(truncation, prompt_var)
    return (
        r"\paragraph{Theorem-1 error budget (GSM8K / Tülu-3-RLVR, $K{=}"
        + str(K_max) + r"$).} Truncation residual "
        r"$|\hat{\mathcal R}_K - \hat{\mathcal R}_{K/2}| = "
        + f"{truncation:.3f}" + r"$; prompt-bootstrap half-width "
        r"$= " + f"{prompt_var:.3f}" + r"$. Verifier-call variance is zero "
        r"(deterministic verifier). Operating accuracy "
        r"$\epsilon = \max{} = " + f"{eps:.3f}" + r"$.")


# ======================================================================
# E.2  Per-difficulty RVR breakdown
# ======================================================================
def _maybe_load_math_levels():
    """Returns list of levels (int 1..5) for the first 1000 algebra-test
    problems, or None on failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra",
                          split="test").select(range(1000))
        import re
        out = []
        for r in ds:
            m = re.search(r"(\d)", str(r.get("level", "")))
            out.append(int(m.group(1)) if m else None)
        return out
    except Exception as e:
        print(f"  (E.2 MATH: {type(e).__name__}: {e}; using fallback)")
        return None


def _maybe_load_humaneval_terciles():
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        lens = np.array([len(r["canonical_solution"]) for r in ds])
        q1, q2 = np.quantile(lens, [1/3, 2/3])
        return np.where(lens <= q1, "short",
                        np.where(lens <= q2, "medium", "long")).tolist()
    except Exception as e:
        print(f"  (E.2 HumanEval: {type(e).__name__}: {e})")
        return None


def build_E2_per_difficulty(results_dir: Path, seed: int = 0) -> str:
    """RVR by difficulty bucket — MATH (level 1-5), HumanEval (length tercile),
    for the three headline models. Falls back to fewer columns if datasets
    can't be loaded."""
    idx = _cell_index(results_dir, seed)
    levels = _maybe_load_math_levels()
    tier = _maybe_load_humaneval_terciles()

    lines = [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Per-difficulty $\hat{\mathcal R}_K$ at $K{=}64$. MATH is "
        r"bucketed by the official \texttt{level} field (1=easiest, "
        r"5=hardest); HumanEval is bucketed by canonical-solution-length "
        r"tercile.}",
        r"\label{tab:appendix_per_difficulty}",
        r"\begin{tabular}{lll rrr}", r"\toprule",
        r"Dataset & Bucket & $n$ & "
        + " & ".join(disp for _, disp in HEADLINE_MODELS) + r" \\",
        r"\midrule",
    ]

    def _row(ds_disp, bucket, mask):
        n = int(mask.sum())
        vals = []
        for m, _ in HEADLINE_MODELS:
            rec = idx.get((m, ds_id))
            if rec is None or "per_prompt_R_hat_K" not in rec:
                vals.append(r"\na")
                continue
            arr = np.array(rec["per_prompt_R_hat_K"])
            if len(arr) != len(mask):
                vals.append(r"\na"); continue
            vals.append(_fmt(float(arr[mask].mean()), 3) if mask.any() else r"\na")
        lines.append(f"{ds_disp} & {bucket} & {n} & " + " & ".join(vals) + r" \\")

    if levels is not None:
        ds_id = "math"; lvl_arr = np.array(levels)
        ds_disp = "MATH"
        for L in range(1, 6):
            mask = (lvl_arr == L)
            _row(ds_disp if L == 1 else "", f"L{L}", mask)
        lines.append(r"\midrule")
    if tier is not None:
        ds_id = "humaneval"; tier_arr = np.array(tier)
        for k, label in enumerate(["short", "medium", "long"]):
            mask = (tier_arr == label)
            _row("HumanEval" if k == 0 else "", label, mask)
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# G — Failure-case candidate filter (qualitative)
# ======================================================================
def build_G_candidates(results_dir: Path, seed: int = 0,
                        max_per_dataset: int = 20) -> str:
    """List prompts that pass the spike-at-0 filter — i.e. $U^\\circ_K(x)=1$
    and $\\bar U_K(x) \\le 0.1$ on the three headline cells. The selection
    of which traces to put in the paper is editorial; this fragment just
    seeds the candidate set."""
    idx = _cell_index(results_dir, seed)
    lines = [
        r"\paragraph{Spike-at-0 candidate prompts for the failure-case "
        r"appendix.} The full per-prompt arrays in the result JSONs let any "
        r"reader filter the same candidate set themselves; the rule is:",
        r"$U^\circ_K(x_i)=1$ \emph{and} $\bar U_K(x_i) \le 0.1$, "
        r"i.e.\ the model can reach the correct answer in 64 samples but "
        r"only on a small fraction of them. The candidate counts below "
        r"come straight from the per-prompt arrays:",
        r"\begin{itemize}",
    ]
    for m, m_disp in HEADLINE_MODELS:
        for ds, ds_disp in HEADLINE_DATASETS:
            rec = idx.get((m, ds))
            if rec is None:
                continue
            uc = np.array(rec.get("per_prompt_U_circ_K", []))
            ub = np.array(rec.get("per_prompt_U_bar_K", []))
            if len(uc) == 0 or len(ub) == 0:
                continue
            mask = (uc == 1.0) & (ub <= 0.1)
            n = int(mask.sum())
            lines.append(f"  \\item {m_disp} / {ds_disp}: {n} candidates "
                         f"out of {len(uc)} prompts.")
    lines.append(r"\end{itemize}")
    return "\n".join(lines)
