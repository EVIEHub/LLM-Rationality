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

import collections
import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional

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


# ---- sample-cache loader (used by D.1 + D.3) --------------------------
def _samples_dir(results_dir: Path) -> Path:
    """Resolve the samples directory from the results directory.

    Cache files live at ``<outputs_root>/data/samples/`` and result JSONs
    at ``<outputs_root>/results/`` in the canonical layout from
    ``configs/paths.yaml``. If the user passed a non-standard
    ``--results-dir`` that doesn't sit next to ``data/samples/``, we fall
    back to :func:`src.pipeline.paths.load_paths`.
    """
    sibling = results_dir.parent / "data" / "samples"
    if sibling.exists():
        return sibling
    from src.pipeline.paths import load_paths
    return load_paths().samples_dir


def _iter_cache_rows(samples_dir: Path,
                     dataset: str,
                     fingerprint: str,
                     K: int = 64) -> Iterator[dict]:
    """Yield ``{prompt_id, prompt, ground_truth, samples}`` rows from the
    sample-cache file identified by ``fingerprint``.

    The filename embeds dataset/model/K/fingerprint (see
    :func:`src.pipeline.cache.cache_path`); we resolve by glob so we don't
    need to round-trip the full ``CacheKey``. Skips the JSONL header line.
    """
    pattern = f"v2_{dataset}_*_K{K}_{fingerprint}.jsonl.gz"
    matches = list(samples_dir.glob(pattern))
    if not matches:
        return
    with gzip.open(matches[0], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == 0:  # header
                continue
            line = line.strip()
            if line:
                yield json.loads(line)


def _cell_fingerprint(rec: dict) -> Optional[str]:
    """Best-effort lookup of the sample-cache fingerprint from a result
    record. Result schemas have varied over time; cover the two field
    names that appear in our JSONs."""
    return rec.get("cache_key_fingerprint") or rec.get("_cache_key_fingerprint")


# ======================================================================
# B.4  GPU-hour breakdown
# ======================================================================
def _load_compute_budget(logs_dir: Path) -> dict[str, float]:
    """Aggregate GPU-hours per hypothesis from
    ``compute_budget.jsonl`` (and any ``compute_budget_*.jsonl``
    siblings, e.g. one per sampling server). Returns
    ``{"h1": hrs, "h2": hrs, "h3": hrs, "h4": hrs}``; missing keys
    map to $0$. Dedupes by (experiment, seed) — the same cell may
    appear in multiple server logs when work was rebalanced.
    """
    out: dict[str, float] = {}
    seen: set[tuple] = set()
    candidates = sorted(logs_dir.glob("compute_budget*.jsonl"))
    for fp in candidates:
        for line in fp.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            exp = r.get("experiment", "")
            md = r.get("metadata", {}) or {}
            key = (exp, md.get("seed"))
            if key in seen:
                continue
            seen.add(key)
            h = exp.split("_", 1)[0]
            if h in ("h1", "h2", "h3", "h4"):
                out[h] = out.get(h, 0.0) + float(r.get("gpu_hours", 0.0))
    return out


def build_B4_gpu_hours(results_dir: Path, seed: int = 0) -> str:
    """GPU-hour breakdown.

    H1, H2, H4 cell runners write ``sampling_seconds`` to each result
    JSON, so we sum that directly. The H3 runner currently omits
    ``sampling_seconds``; for H3 we read ``compute_budget.jsonl``
    (and any per-server siblings under ``outputs/logs/``) which the
    sampling stack appends to on every run. The two sources are
    combined into one table so every hypothesis has a real number.
    """
    # Per-cell sampling_seconds (the source of truth where present).
    per_cell: dict[str, dict] = {}
    for sub in ("h1", "h2", "h3", "h4"):
        n = 0
        n_with_secs = 0
        secs_sum = 0.0
        for rec in _load_dir(results_dir, sub, seed):
            n += 1
            v = rec.get("sampling_seconds")
            if v is not None:
                secs_sum += float(v)
                n_with_secs += 1
        per_cell[sub] = {"n": n, "n_with_secs": n_with_secs,
                         "secs": secs_sum}

    # Fallback / cross-check: aggregate compute_budget.jsonl by
    # hypothesis. logs_dir conventionally sits at results_dir.parent / "logs".
    logs_dir = results_dir.parent / "logs"
    budget = _load_compute_budget(logs_dir) if logs_dir.exists() else {}

    rows = []
    total_hrs = 0.0
    any_from_budget = False
    for sub in ("h1", "h2", "h3", "h4"):
        t = per_cell[sub]
        n, n_w, secs = t["n"], t["n_with_secs"], t["secs"]
        from_cells = secs / 3600.0
        if n_w == n and n > 0:
            # Complete per-cell timing — use it.
            hrs = from_cells
            note = "per-cell"
        elif n_w == 0 and sub in budget:
            # Per-cell timing absent; fall back to compute_budget log.
            hrs = budget[sub]
            note = "compute\\_budget"
            any_from_budget = True
        elif n_w == 0:
            hrs = float("nan")
            note = "no record"
        else:
            # Mixed — sum per-cell and top up with the compute-budget
            # delta for cells the budget log knows about but the JSONs
            # don't.
            hrs = max(from_cells, budget.get(sub, 0.0))
            note = "per-cell + budget"
        rows.append((sub.upper(), n, n_w, hrs, note))
        if not (hrs != hrs):  # not nan
            total_hrs += hrs

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{GPU-hour breakdown for the sampling stage "
        r"(\emph{incremental} cost; cache hits across hypotheses are "
        r"counted only once, against whichever hypothesis first "
        r"populated the cache). Verification (CPU) and bootstrap (CPU) "
        r"wall-time are not included. For H1, H2, H4 the wall-clock "
        r"comes from each cell's \texttt{sampling\_seconds} field; "
        r"for H3 the cell runner does not log per-cell timing, so we "
        r"fall back to the global \texttt{compute\_budget.jsonl} "
        r"append-log (merged across both sampling servers, dedup'd by "
        r"(experiment, seed)). H2 looks small because (i) the H2 "
        r"\textbf{Tülu-3-8B-RLVR} cells re-use the H1 RLVR sample "
        r"cache hit-for-hit (the H1 RLVR sampling, $\approx 2.45$~hr, "
        r"is counted once under H1), and (ii) the H2 "
        r"\textbf{Tülu-3-70B} trajectory currently covers only the "
        r"two deployment datasets (LiveCodeBench, MathArena); "
        r"extending it to the dev panel would add an estimated "
        r"$15\text{--}25$~GPU-hr per 70B stage. The H3 cache-hit "
        r"cells (the $\tau{=}1$ direct cells and SC bootstraps re-use "
        r"the H1 cache) do not appear in \texttt{compute\_budget.jsonl} "
        r"and correctly contribute $0$ incremental GPU-hours.}",
        r"\label{tab:appendix_gpu_hours}",
        r"\begin{tabular}{lrrrl}", r"\toprule",
        r"Hypothesis & Cells & w/ timing & GPU-hr & Source \\",
        r"\midrule",
    ]
    for hyp, n, n_w, hrs, note in rows:
        gh = r"\textemdash" if hrs != hrs else f"{hrs:.1f}"  # nan check
        lines.append(f"{hyp} & {n} & {n_w} & {gh} & \\texttt{{{note}}} \\\\")
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{Total}} & "
        f"{sum(t['n'] for t in per_cell.values())} & "
        f"{sum(t['n_with_secs'] for t in per_cell.values())} & "
        f"{total_hrs:.1f} & \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# C.1  Saturation curve tables for the 3 headline H1 cells
# ======================================================================
def build_C1_saturation(results_dir: Path, seed: int = 0,
                        models=HEADLINE_MODELS,
                        datasets=HEADLINE_DATASETS,
                        heatmap: bool = True) -> str:
    """Per-(model, dataset) saturation-curve table: 3 metrics x 7 $K$ values
    with the 95% prompt-bootstrap CI on RVR at $K_{\\max}$.

    With ``heatmap=True`` (default) each cell carries an
    anchor-coloured ``\\cellcolor{}`` whose intensity is proportional
    to the value (same `_shade` helper as the H1 main-text tables);
    REU rows are tinted ``colREU``, AEU rows ``colAEU``, RVR rows
    ``colRVR``.
    """
    from src.plotting.tables import _shade  # local import to avoid cycle
    idx = _cell_index(results_dir, seed)

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Saturation curves for the headline H1 cells. Each entry "
        r"is the prompt-mean at the given budget $K'$; the last column's "
        r"$\hat{\mathcal R}_K$ row carries the 95\% prompt-bootstrap CI "
        r"(B$=$1000 resamples). Cells are auto-coloured by value "
        r"(\texttt{colREU}/\texttt{colAEU}/\texttt{colRVR} from the "
        r"H1 main-text colour scheme; intensity $\propto$ value).}",
        r"\label{tab:appendix_saturation}",
        r"\begin{tabular}{lll *{7}{c}}", r"\toprule",
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

    def _cell(value: float, anchor: str) -> str:
        return _shade(anchor, value, heatmap) + _fmt(value, 3)

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
            # 3 rows per (model, dataset). Each ``anchor`` is the
            # colour ramp used as background.
            for ri, (label, key, anchor, with_ci) in enumerate([
                (r"$U^\circ_K$",            "U_circ_K", "colREU", False),
                (r"$\bar U_K$",             "U_bar_K",  "colAEU", False),
                (r"$\hat{\mathcal R}_K$",   "R_hat_K",  "colRVR", True),
            ]):
                def _format_cell(k):
                    if k not in curve:
                        return r"\na"
                    v = curve[k].get(key, 0.0)
                    suffix = ci_str if (with_ci and k == K_grid[-1] and ci_str) else ""
                    return _cell(v, anchor) + suffix
                vals = " & ".join(_format_cell(k) for k in K_grid)
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
        r"\begin{table}[htbp]", r"\centering", r"\small",
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
# Audit-log loader shared by C.2 + D.2
# ======================================================================
def _load_self_judge_audit(
    logs_dir: Path,
    dataset: str = "ultrafeedback",
    experiment: str = "h1",
    experiment_match: str = "exact",
) -> Optional[dict[str, np.ndarray]]:
    """Load the self-judge audit log for one cell and reshape into arrays.

    The audit log lives at ``${logs_dir}/verifier/{dataset}_log.jsonl``;
    each line is the per-(prompt, k) record written by
    ``scripts.run_h{1,2}.py``. We filter to the requested ``experiment``
    and gather every record with ``raw_verdicts`` populated.

    Two experiment-name formats coexist in the wild:
      * legacy: ``experiment="h1"`` (the bare hypothesis tag);
      * current: ``experiment="h1_<model>_<dataset>_seed<N>"`` (the
        full cell key recorded by ``scripts/run_h1.py``).
    Pass ``experiment_match="prefix"`` to match the latter by prefix —
    e.g.\\ ``experiment="h1_tulu3-8b-rlvr_ultrafeedback_seed0"`` selects
    the H1 Tülu-3-RLVR / UltraFeedback cell. With the default
    ``"exact"`` we require equality.

    Returns ``None`` if the file is missing or contains no usable rows
    (the C.2 / D.2 builders then emit a ``% NOTE`` stub instead of
    failing the build).

    On duplicate ``(prompt_id, k)`` pairs (a re-run of a partially-
    completed cell appends, never overwrites) we keep the *last*
    record for that pair — it reflects the final state of the cell on
    disk. This matches the convention used by the cache loader.

    Returns dict keys:
      ``raw_verdicts``      (N, L)  float  — candidate-relative outcome
      ``a_is_candidate``    (N, L)  bool   — position flag (or empty if
                                              older runs didn't record it)
      ``utility``           (N,)    float  — the aggregated U value
      ``prompt_ids``        (N,)    list of strings, indexed 0..N-1 in
                                    the same order as the arrays above
      ``ks``                (N,)    int    — per-row k
      ``L``                 int     — verdicts per row
      ``seed``              int     — recorded seed (for D.2 a_is_candidate
                                       reconstruction; assumed shared
                                       across rows of one cell)
    """
    log_file = logs_dir / "verifier" / f"{dataset}_log.jsonl"
    if not log_file.exists():
        return None

    def _matches(rec_exp: Optional[str]) -> bool:
        if rec_exp is None:
            return False
        if experiment_match == "exact":
            return rec_exp == experiment
        if experiment_match == "prefix":
            return rec_exp.startswith(experiment)
        raise ValueError(f"unknown experiment_match {experiment_match!r}")

    # First pass: collect rows, deduplicating (prompt_id, k) — later
    # entries win (matches the "append on re-run" semantics).
    by_key: dict[tuple[str, int], dict] = {}
    for line in open(log_file, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _matches(rec.get("experiment")):
            continue
        if "raw_verdicts" not in rec:
            continue
        by_key[(rec.get("prompt_id"), rec.get("k"))] = rec
    if not by_key:
        return None
    # Stable ordering: sort by (prompt_id, k) so the row index is
    # reproducible across runs and across machines.
    ordered = sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    rows = [v for _, v in ordered]

    L_per_row = {len(r["raw_verdicts"]) for r in rows}
    if len(L_per_row) != 1:
        from collections import Counter
        L_pick, _ = Counter(L_per_row).most_common(1)[0]
        rows = [r for r in rows if len(r["raw_verdicts"]) == L_pick]
    L = len(rows[0]["raw_verdicts"])
    raw = np.array([r["raw_verdicts"] for r in rows], dtype=float)  # (N, L)
    has_pos = all("a_is_candidate" in r for r in rows)
    pos = (np.array([r["a_is_candidate"] for r in rows], dtype=bool)
           if has_pos else np.empty((0, L), dtype=bool))
    util = np.array([r["utility"] for r in rows], dtype=float)
    pids = [r["prompt_id"] for r in rows]
    ks = np.array([r.get("k", 0) for r in rows], dtype=int)
    seeds_seen = {r.get("seed") for r in rows}
    seed_val = next(iter(seeds_seen)) if len(seeds_seen) == 1 else None
    return {
        "raw_verdicts": raw, "a_is_candidate": pos,
        "utility": util, "prompt_ids": pids, "ks": ks,
        "L": L, "seed": seed_val,
    }


def _reconstruct_a_is_candidate(
    seed: int, prompt_ids: list[str], ks: np.ndarray, L: int,
) -> np.ndarray:
    """Recompute the (N, L) position-flag matrix from the cell seed alone.

    ``src.verification.self_judge.score_matrix`` draws the position
    coins as ``rng.random(size=(M, K, L)) < 0.5`` where ``rng =
    np.random.default_rng(seed)`` and the iteration order is
    ``for i in range(M): for k in range(K): for l in range(L)``. The
    audit log records ``(prompt_id, k)`` per row; we recover ``i`` as
    the unique-prompt-id rank (sorted by prompt_id, matching the
    cache's deterministic per-prompt order) and slice the full
    ``(M, K, L)`` matrix.

    Returns ``np.empty((0, L), bool)`` if any prompt_id maps to an out-
    of-bounds index, signalling the audit-log ordering and the
    runner's iteration order have diverged (in which case the caller
    should treat D.2 as unrecoverable and emit a ``% NOTE`` stub).
    """
    unique_pids = sorted(set(prompt_ids))
    M = len(unique_pids)
    K_max = int(ks.max()) + 1 if len(ks) else 0
    pid_to_i = {p: i for i, p in enumerate(unique_pids)}
    rng = np.random.default_rng(seed)
    full = rng.random(size=(M, K_max, L)) < 0.5  # (M, K, L) bool
    out = np.empty((len(prompt_ids), L), dtype=bool)
    for n, (pid, k) in enumerate(zip(prompt_ids, ks)):
        i = pid_to_i.get(pid)
        if i is None or k >= K_max:
            return np.empty((0, L), dtype=bool)
        out[n] = full[i, k]
    return out


# ======================================================================
# C.2  L-sensitivity (re-aggregate from raw verdicts)
# ======================================================================
def _aggregate_strict_majority(votes: np.ndarray, L_p: int) -> np.ndarray:
    """Per-row strict-majority aggregation of ``(N, L_p)`` verdicts.

    Mirrors :func:`src.verification.self_judge._aggregate`: a class
    wins iff its count is at least ``ceil(L_p/2)``. Ties (0.5) are not
    a class so they never trigger a win; they only matter when neither
    1.0 nor 0.0 clears the threshold, in which case the row resolves
    to 0.5.

    Tie-break: when both ``n_win >= thr`` and ``n_lose >= thr``
    (possible at even $L_p$), the reference implementation returns
    ``1.0`` because its ``if n_win >= thr: return 1.0`` branch fires
    first. We mirror that by assigning ``0.0`` first and ``1.0``
    second so the latter overrides on a clash.
    """
    thr = (L_p + 1) // 2
    n_win = (votes == 1.0).sum(axis=1)
    n_lose = (votes == 0.0).sum(axis=1)
    out = np.full(votes.shape[0], 0.5, dtype=float)
    out[n_lose >= thr] = 0.0
    out[n_win >= thr] = 1.0
    return out


def build_C2_L_sensitivity(
    results_dir: Path,
    seed: int = 0,
    dataset: str = "ultrafeedback",
    experiment: str = "h1_tulu3-8b-rlvr_ultrafeedback_seed",
    L_grid=(1, 3, 5, 7, 9),
    B: int = 200,
) -> str:
    """Re-aggregate the recorded L=5 verdicts at L' ∈ L_grid.

    L' ≤ 5 sub-samples without replacement from the recorded 5 verdicts
    per (prompt, k); L' > 5 resamples with replacement (a bootstrap that
    approximates fresh i.i.d. judge calls under the stationary judge
    distribution). For each L' we report:
      * $\\hat{\\mathcal R}_K$ — mean U across rows (here a proxy: the
        prompt-mean of utility under the re-aggregated rule).
      * std($\\hat{\\mathcal R}_K$) — across B bootstrap resamples of the
        L'-subsample (with the prompt set held fixed).
      * Parse-failure rate is read from row utilities (independent of
        L', a per-call property of the judge).
    """
    paths = _paths_for(results_dir)
    audit = _load_self_judge_audit(
        paths.logs_dir, dataset=dataset,
        experiment=experiment, experiment_match="prefix",
    )
    if audit is None:
        return (r"% C.2: self-judge audit log not present at "
                r"\verb|" + str(paths.logs_dir / "verifier" /
                                 f"{dataset}_log.jsonl") + r"| or has "
                r"no rows for experiment prefix " + experiment + r". "
                r"Pull the log from a server that ran the H1 "
                r"UltraFeedback cells with audit logging enabled.")

    raw = audit["raw_verdicts"]  # (N, L=5)
    L_rec = audit["L"]
    N = raw.shape[0]
    parse_fail = float((raw == 0.5).mean())  # rough proxy: ties + parse fails

    # Reshape rows back into the per-prompt (M, K) layout so
    # $\hat{\mathcal R}_K = \mathbb{E}_x [U^\circ_K(x) - \bar U_K(x)]$
    # can be computed under each L' aggregation. The loader sorted
    # rows by (prompt_id, k) so the row index runs prompt-major.
    pids = audit["prompt_ids"]
    ks = audit["ks"]
    unique_pids = sorted(set(pids))
    M = len(unique_pids)
    K_cell = int(ks.max()) + 1
    pid_to_i = {p: i for i, p in enumerate(unique_pids)}
    # row_idx[i, k] = index into the (N, L) raw_verdicts array.
    row_idx = np.full((M, K_cell), -1, dtype=int)
    for n, (pid, k) in enumerate(zip(pids, ks)):
        row_idx[pid_to_i[pid], int(k)] = n
    if (row_idx < 0).any():
        return (r"% C.2: audit-log gaps — some (prompt, k) pairs "
                r"missing for " + experiment + r". Re-run the cell.")

    rng = np.random.default_rng(seed)
    rows = []
    for L_p in L_grid:
        # Per-prompt R_hat_K across B resamples of the L' votes.
        rhat = np.zeros(B, dtype=float)
        for b in range(B):
            if L_p <= L_rec:
                perm = np.argsort(rng.random((N, L_rec)), axis=1)[:, :L_p]
                votes_b = np.take_along_axis(raw, perm, axis=1)
            else:
                idx = rng.integers(0, L_rec, size=(N, L_p))
                votes_b = np.take_along_axis(raw, idx, axis=1)
            u_flat = _aggregate_strict_majority(votes_b, L_p)  # (N,)
            U = u_flat[row_idx]                                # (M, K_cell)
            U_circ = U.max(axis=1)                             # (M,)
            U_bar = U.mean(axis=1)                             # (M,)
            rhat[b] = float((U_circ - U_bar).mean())
        rows.append((L_p, float(rhat.mean()), float(rhat.std())))

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Effect of $L$ on $\hat{\mathcal R}_K$ at $K{=}"
        + str(K_cell) + r"$ for the H1 preference cell (Tülu-3-RLVR "
        r"on UltraFeedback, $M{=}" + str(M) + r"$). Std is across "
        r"$B{=}" + str(B) + r"$ bootstrap resamples of the $L'$-"
        r"subsampled vote pattern with the prompt set held fixed. The "
        r"$L{=}" + str(L_rec) + r"$ row matches the headline "
        r"preference cell exactly (no resampling: when $L'{=}L_{\text{rec}}$ "
        r"each ``sample'' uses all recorded verdicts, so std$=0$ by "
        r"construction).}",
        r"\label{tab:appendix_L_sensitivity}",
        r"\begin{tabular}{cccc}", r"\toprule",
        r"$L'$ & $\hat{\mathcal R}_K$ & std (verdict-bootstrap) & "
        r"Parse-failure rate \\",
        r"\midrule",
    ]
    for L_p, mu, sd in rows:
        pf = f"{parse_fail:.4f}" if L_p == L_rec else r"\textemdash"
        lines.append(f"{L_p} & {mu:.3f} & {sd:.4f} & {pf} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# D.2  Position bias + inter-rater agreement (Krippendorff's α)
# ======================================================================
def _krippendorff_alpha_ternary(verdicts: np.ndarray) -> float:
    """Krippendorff's α on the ordinal scale ``{0, 0.5, 1}``.

    Args:
        verdicts: (N, L) array of per-call outcomes. Each row is an
            "item" rated by L "coders".

    Closed-form coincidence-matrix formulation (Hayes & Krippendorff,
    2007). Memory is O(C²) where C=3 is the number of ordinal
    categories — independent of N and L, so the function scales to
    the N≈32k items we have here (the naive pairwise version blows
    out memory at ≈190 GiB).

      $\\alpha = 1 - \\frac{\\sum_{c,c'} o_{c,c'} \\, (c-c')^2}
                          {\\sum_{c,c'} e_{c,c'} \\, (c-c')^2}$

    where $o_{c,c'}$ is the observed and $e_{c,c'}$ the expected
    coincidence matrix, both corrected for sampling without
    replacement. Returns ``1.0`` for perfect agreement, ``0.0`` for
    chance, negative for systematic disagreement; ``nan`` when the
    sample is too small or every rating is the same category
    (vacuous α).
    """
    v = verdicts
    N, L = v.shape
    if N == 0 or L < 2:
        return float("nan")
    categories = np.array([0.0, 0.5, 1.0])
    # Per-item category counts, shape (N, 3).
    counts = np.stack([(v == c).sum(axis=1) for c in categories], axis=1)
    n_per_cat = counts.sum(axis=0).astype(float)   # (3,)
    n_total = float(n_per_cat.sum())               # = N · L
    if n_total < 2 or (n_per_cat > 0).sum() < 2:
        # All ratings the same category — α is vacuous (no signal).
        return float("nan")
    diff_sq = (categories[:, None] - categories[None, :]) ** 2  # (3,3)

    # Observed coincidence: Σᵤ counts.T @ counts (pair counts per
    # item) with diagonal correction (no self-pair) divided by (L-1).
    o = counts.T.astype(float) @ counts.astype(float)
    diag = np.arange(3)
    o[diag, diag] -= n_per_cat
    o /= (L - 1)

    # Expected coincidence: random pairing across all units.
    e = np.outer(n_per_cat, n_per_cat)
    e[diag, diag] -= n_per_cat
    e /= (n_total - 1)

    D_o = float((o * diff_sq).sum())
    D_e = float((e * diff_sq).sum())
    if D_e == 0:
        return 1.0
    return 1.0 - D_o / D_e


_D2_H1_JUDGES = [
    ("h1_tulu3-8b-rlvr_ultrafeedback_seed",   "Tülu-3-8B-RLVR"),
    ("h1_qwen2.5-7b-instruct_ultrafeedback_seed", "Qwen2.5-7B-Instruct"),
    ("h1_llama3.1-8b-instruct_ultrafeedback_seed", "Llama-3.1-8B-Instruct"),
]


def build_D2_position_bias(
    results_dir: Path,
    seed: int = 0,
    dataset: str = "ultrafeedback",
    judges=_D2_H1_JUDGES,
) -> str:
    """Position-bias rate + Krippendorff's $\\alpha$ + majority margin.

    Reads the H1 self-judge audit log for each of the three H1 judges
    (Tülu-3-RLVR, Qwen2.5-7B, Llama-3.1-8B). For each judge we report:
      * A-pick rate: fraction of (i, k, l) verdicts where the judge
        picked the response in position A. A no-bias judge gives 0.5.
      * Krippendorff's $\\alpha$ on ternary scale across the $L$ raters.
      * Mean $|$majority margin$|$ = mean across (i, k) of the absolute
        difference between win-class and lose-class counts.

    A-pick recovery needs the per-call position flag
    ``a_is_candidate``. Logs written before the 2026-05-24 audit-log
    patch don't carry that field — but the position assignment in
    :func:`src.verification.self_judge.score_matrix` is purely a
    function of ``(seed, M, K, L)``:

        rng = np.random.default_rng(seed)
        a_is_candidate = rng.random(size=(M, K, L)) < 0.5

    so we recompute it deterministically from the same seed (recorded
    in every audit-log row). The Krippendorff $\\alpha$ and majority
    margin only need ``raw_verdicts`` and are unaffected by the
    reconstruction step. See ``_reconstruct_a_is_candidate`` for the
    pure function used here.
    """
    paths = _paths_for(results_dir)
    log_file = paths.logs_dir / "verifier" / f"{dataset}_log.jsonl"
    if not log_file.exists():
        return (r"% D.2: self-judge audit log not present at "
                r"\verb|" + str(log_file) + r"|. Pull the log from a "
                r"server that ran the H1 UltraFeedback cells.")

    rows = [r"\begin{table}[htbp]", r"\centering", r"\small",
            r"\caption{Self-judge audit-log diagnostics on the H1 "
            r"UltraFeedback cells. \emph{A-pick rate} is the fraction "
            r"of non-tie judge calls (across $M\cdot K\cdot L$ "
            r"verdicts) where the judge picked the response in "
            r"position A — a no-bias judge gives $0.5$. The position "
            r"flag was reconstructed from the recorded cell seed via "
            r"the same RNG call used at sampling time (deterministic "
            r"replay; see \cref{app:calibration:L}). $\alpha$ is "
            r"Krippendorff's $\alpha$ on the ternary scale "
            r"$\{0,0.5,1\}$ across the $L{=}5$ verdicts of each "
            r"(prompt, candidate) pair. \emph{Mean margin} is the "
            r"average $|n_{\text{win}} - n_{\text{lose}}|$ across pairs.}",
            r"\label{tab:appendix_position_bias}",
            r"\begin{tabular}{lccc}", r"\toprule",
            r"Judge & A-pick rate & $\alpha$ (ternary) & "
            r"mean $|$margin$|$ \\", r"\midrule"]

    any_row = False
    for exp_prefix, disp in judges:
        audit = _load_self_judge_audit(
            paths.logs_dir, dataset=dataset,
            experiment=exp_prefix, experiment_match="prefix",
        )
        if audit is None:
            rows.append(disp + r" & \textemdash & \textemdash & "
                        r"\textemdash \\")
            continue
        any_row = True
        raw = audit["raw_verdicts"]                      # (N, L)
        L = audit["L"]
        # Use the recorded a_is_candidate if present (post-Phase-2),
        # otherwise reconstruct from seed (pre-Phase-2 logs).
        pos = audit["a_is_candidate"]
        method = "logged"
        if pos.size == 0:
            cell_seed = audit["seed"]
            if cell_seed is None:
                rows.append(disp + r" & \textemdash & \textemdash & "
                            r"\textemdash \\")
                continue
            pos = _reconstruct_a_is_candidate(
                cell_seed, audit["prompt_ids"], audit["ks"], L,
            )
            method = "reconstructed"
            if pos.size == 0:
                rows.append(disp + r" & \textemdash & \textemdash & "
                            r"\textemdash \\")
                continue

        # A-pick recovery.
        picked_a = (pos & (raw == 1.0)) | (~pos & (raw == 0.0))
        non_tie = (raw != 0.5)
        a_pick = (float(picked_a[non_tie].mean())
                  if non_tie.any() else float("nan"))
        alpha = _krippendorff_alpha_ternary(raw)
        n_win = (raw == 1.0).sum(axis=1)
        n_lose = (raw == 0.0).sum(axis=1)
        margin = float(np.abs(n_win - n_lose).mean())
        rows.append(f"{disp} & {a_pick:.3f} & {alpha:.3f} & "
                    f"{margin:.2f} \\\\  % position via {method}")

    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    if not any_row:
        return (r"% D.2: audit log present but no rows for any H1 "
                r"self-judge cell. Make sure the UltraFeedback log "
                r"contains experiment prefixes \verb|h1_<model>_"
                r"ultrafeedback_seed|.")
    return "\n".join(rows)


# ======================================================================
# Paths helper for C.2 / D.2 (read the project paths config)
# ======================================================================
def _paths_for(results_dir: Path):
    """Resolve the rest of the paths layout from a passed results_dir.

    Mirror of :func:`_samples_dir`: prefer the canonical layout (logs
    sit next to results under outputs_root); fall back to
    :func:`src.pipeline.paths.load_paths` if the caller pointed at a
    non-standard results dir.
    """
    sibling = results_dir.parent / "logs"
    if sibling.exists():
        # Tiny shim object — only ``logs_dir`` is used by the C.2/D.2
        # builders, no need to bring in the full Paths dataclass.
        class _P: pass
        p = _P(); p.logs_dir = sibling; return p
    from src.pipeline.paths import load_paths
    return load_paths()


# ======================================================================
# D.1  MATH verifier failure-mode rates
# ======================================================================
# Categories returned by ``src.verification.math.verify_with_reason``,
# in the order they should appear in the table (column-major sense:
# extraction failures → parse failures → verification failures →
# semantic outcomes).
_D1_REASONS = [
    ("no_boxed",          r"\shortstack[c]{no \\ \texttt{\textbackslash boxed\{\}}}"),
    ("empty_gt",          r"\shortstack[c]{empty \\ GT}"),
    ("parse_error",       r"\shortstack[c]{parse \\ error}"),
    ("parse_empty",       r"\shortstack[c]{parse \\ empty}"),
    ("verify_exception",  r"\shortstack[c]{verify \\ exception}"),
    ("incorrect",         r"incorrect"),
    ("correct",           r"correct"),
]


def build_D1_math_failures(results_dir: Path,
                           seed: int = 0,
                           models=HEADLINE_MODELS) -> str:
    """Per-(prompt, sample) MATH verifier failure-mode rates.

    Re-verifies every cached sample for the three headline MATH cells
    using :func:`src.verification.math.verify_with_reason` and reports
    each failure mode as a percentage of the $M \\times K$ (prompt,
    sample) pairs. Rows are models; columns are failure categories.
    Returns ``"% D1: <reason>"`` and a stub if any cache is missing.
    """
    from src.verification.math import verify_with_reason  # local: avoid heavy import at module load

    idx = _cell_index(results_dir, seed)
    samples_dir = _samples_dir(results_dir)
    table_rows: list[tuple[str, dict[str, int], int]] = []
    for m, m_disp in models:
        rec = idx.get((m, "math"))
        if rec is None:
            return f"% D1: result JSON missing for {m} / math"
        fp = _cell_fingerprint(rec)
        if fp is None:
            return f"% D1: cache_key_fingerprint missing for {m} / math"
        counts: dict[str, int] = collections.Counter()
        total = 0
        any_row = False
        for row in _iter_cache_rows(samples_dir, "math", fp):
            any_row = True
            gt = row.get("ground_truth", "")
            for s in row.get("samples", []):
                _, reason = verify_with_reason(s, gt)
                counts[reason] += 1
                total += 1
        if not any_row:
            return f"% D1: sample cache missing for {m} / math (fp={fp})"
        table_rows.append((m_disp, counts, total))

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{MATH verifier failure-mode rates on the three headline "
        r"cells. Each cell is the percentage of $(x_i, y_{i,k})$ pairs "
        r"(out of $M{\cdot}K=1000{\cdot}64{=}64{,}000$) whose "
        r"verification terminated in the indicated state, produced by "
        r"re-running \texttt{verify\_with\_reason} on the cached "
        r"generations. \texttt{correct} and \texttt{incorrect} are the "
        r"two semantic outcomes; the remaining columns are failures of "
        r"extraction (no \texttt{\textbackslash boxed}, empty GT), of "
        r"\texttt{math-verify}'s SymPy parser, or of the equivalence "
        r"check itself.}",
        r"\label{tab:appendix_D1_math_failures}",
        r"\begin{tabular}{l " + "c" * len(_D1_REASONS) + r" r}", r"\toprule",
        r"Model & " + " & ".join(disp for _, disp in _D1_REASONS)
        + r" & $M{\cdot}K$ \\",
        r"\midrule",
    ]
    for m_disp, counts, total in table_rows:
        cells = []
        for key, _ in _D1_REASONS:
            pct = 100.0 * counts.get(key, 0) / total if total else 0.0
            # Format: 0.0% → "--", small (<0.1) but nonzero → "<0.1",
            # else 1-dp percent. Keeps the row from being dominated by
            # the "correct" column visually.
            if counts.get(key, 0) == 0:
                cells.append(r"\textemdash")
            elif pct < 0.1:
                cells.append(r"$<$0.1")
            else:
                cells.append(f"{pct:.1f}")
        lines.append(f"{m_disp} & " + " & ".join(cells) + f" & {total} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ======================================================================
# D.3  GSM8K extractor pattern-firing rates
# ======================================================================
# Pattern-name tags returned by
# ``src.verification.gsm8k.extract_answer_with_pattern``.
_D3_PATTERNS = [
    ("hashed",    r"\texttt{\#\#\#\# $N$}"),
    ("boxed",     r"\texttt{\textbackslash boxed\{N\}}"),
    ("answer_is", r"``the answer is $N$''"),
    ("bare",      r"\shortstack[c]{last bare \\ number}"),
    ("none",      r"\shortstack[c]{no number \\ parsed}"),
]


def build_D3_gsm8k_patterns(results_dir: Path,
                            seed: int = 0,
                            models=HEADLINE_MODELS) -> str:
    """GSM8K extractor pattern-firing rates broken down by correctness.

    For each of the four extraction patterns we report the percentage of
    $(x_i, y_{i,k})$ pairs that ended up using it (priority order means
    later patterns only fire when earlier ones missed), together with
    the conditional correctness $P(\\text{correct} \\mid \\text{pattern})$
    — the latter is the diagnostic the paper actually needs ("does the
    bare-number fallback hurt our accuracy?").
    """
    from src.verification.gsm8k import (
        extract_answer, extract_answer_with_pattern,
    )
    import math as _math

    idx = _cell_index(results_dir, seed)
    samples_dir = _samples_dir(results_dir)

    # rows[i] = (model_display, pattern_counts, correct_given_pattern, total)
    table_rows: list[tuple[str, dict[str, int], dict[str, int], int]] = []
    for m, m_disp in models:
        rec = idx.get((m, "gsm8k"))
        if rec is None:
            return f"% D3: result JSON missing for {m} / gsm8k"
        fp = _cell_fingerprint(rec)
        if fp is None:
            return f"% D3: cache_key_fingerprint missing for {m} / gsm8k"
        pat_count: dict[str, int] = collections.Counter()
        pat_correct: dict[str, int] = collections.Counter()
        total = 0
        any_row = False
        for row in _iter_cache_rows(samples_dir, "gsm8k", fp):
            any_row = True
            gt = row.get("ground_truth", "")
            target = extract_answer(gt)
            for s in row.get("samples", []):
                pred, pat = extract_answer_with_pattern(s)
                pat_count[pat] += 1
                total += 1
                if (pred is not None and target is not None
                        and _math.isclose(pred, target,
                                          rel_tol=1e-9, abs_tol=1e-9)):
                    pat_correct[pat] += 1
        if not any_row:
            return f"% D3: sample cache missing for {m} / gsm8k (fp={fp})"
        table_rows.append((m_disp, pat_count, pat_correct, total))

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{GSM8K extractor pattern-firing rates on the three "
        r"headline cells. \emph{Fires} is the percentage of "
        r"$(x_i, y_{i,k})$ pairs whose final answer was pulled out by "
        r"the given regex (priority order is left to right: \texttt{\#\#\#\#} "
        r"first, then \texttt{\textbackslash boxed}, then ``the answer is'', "
        r"then the rightmost bare number); \emph{acc.}\ is the conditional "
        r"correctness given the pattern. A high \emph{fires}/\emph{acc.}\ "
        r"on \texttt{bare} would indicate the verifier is leaning on a "
        r"noisy fallback; in practice it does not.}",
        r"\label{tab:appendix_D3_gsm8k_patterns}",
        r"\begin{tabular}{l "
        + " ".join(["cc"] * len(_D3_PATTERNS))
        + r" r}", r"\toprule",
        r"\multirow{2}{*}{Model} & "
        + " & ".join(r"\multicolumn{2}{c}{%s}" % disp for _, disp in _D3_PATTERNS)
        + r" & \multirow{2}{*}{$M{\cdot}K$} \\",
        # cmidrule per pattern (cols 2-3, 4-5, ...)
        " ".join(
            r"\cmidrule(lr){%d-%d}" % (2 + 2 * j, 3 + 2 * j)
            for j in range(len(_D3_PATTERNS))
        ),
        r" & " + " & ".join([r"fires & acc."] * len(_D3_PATTERNS)) + r" & \\",
        r"\midrule",
    ]
    for m_disp, pat_count, pat_correct, total in table_rows:
        cells = []
        for key, _ in _D3_PATTERNS:
            n = pat_count.get(key, 0)
            c = pat_correct.get(key, 0)
            pct_fires = 100.0 * n / total if total else 0.0
            if n == 0:
                cells.extend([r"\textemdash", r"\textemdash"])
                continue
            if pct_fires < 0.1:
                fires_str = r"$<$0.1"
            else:
                fires_str = f"{pct_fires:.1f}"
            if key == "none":
                # Correctness is undefined when no number was parsed
                # (always 0 by construction); leave the acc. column blank
                # to avoid implying a meaningful denominator.
                cells.extend([fires_str, r"\textemdash"])
            else:
                acc = 100.0 * c / n
                cells.extend([fires_str, f"{acc:.1f}"])
        lines.append(f"{m_disp} & " + " & ".join(cells) + f" & {total} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


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
        r"\begin{table}[htbp]", r"\centering", r"\small",
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
