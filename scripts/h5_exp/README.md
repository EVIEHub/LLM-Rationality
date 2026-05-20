# H5 — contamination-resistant deployment battery

**H5 is not a new analysis.** It is the H1–H4 battery re-run on
contamination-resistant *deployment* benchmarks released after the
evaluated models' training cutoffs. The unifying claim:

> **H5 (contamination-robust replication):** the H1–H4 signatures — the
> saturation of `U_circ_K`, the `R_hat_K` gap, the alignment-stage
> ordering, the length relationship — replicate on post-cutoff
> deployment data. The rational gap is therefore a genuine
> capability-elicitation phenomenon, not a pretraining-memorization
> artifact.

## Datasets (both contamination-resistant by construction)

| dataset | content | cutoff note |
|---|---|---|
| `matharena` | 60 competition-math problems (AIME 2025 + BRUMO 2025) | post-cutoff for all evaluated models |
| `livecodebench` | 75 competitive-programming problems, `contest_date >= 2024-01-01` | post-cutoff for Llama-3.1; borderline for Tülu-3 / Qwen2.5 |

## Panels

The four panels here invoke the **same** `scripts/run_h{1,2,3,4}.py`
cell runners as the development battery — only the dataset set differs.
Results land in `results/h{1,2,3,4}/` keyed by dataset (the `matharena`
/ `livecodebench` JSONs are the H5 slice); the plotters isolate them via
`SCOPE_DATASETS["deployment"]` in `src/plotting/_common.py`.

| panel | analysis | cells |
|---|---|---|
| `run_h1_panel.sh` | saturation curve | 3 models × 2 datasets |
| `run_h2_panel.sh` | alignment trajectory (SFT/DPO/RLVR) | 3 stages × 2 datasets |
| `run_h3_panel.sh` | inference-procedure utilities | 3 models × 2 datasets × {τ, SC} |
| `run_h4_panel.sh` | reasoning-length relationship | Tülu-3-RLVR × matharena × 7 lengths |

> Note: the per-panel filenames keep their `h1`–`h4` names because they
> drive the corresponding *analysis* runner. The directory as a whole is
> the paper's **H5 experiment**.

## Usage

```bash
bash scripts/h5_exp/run_h1_panel.sh [--num-gpus N] [model_alias ...]
bash scripts/h5_exp/run_h2_panel.sh [--num-gpus N]
bash scripts/h5_exp/run_h3_panel.sh [--num-gpus N]
bash scripts/h5_exp/run_h4_panel.sh [--num-gpus N]
```
