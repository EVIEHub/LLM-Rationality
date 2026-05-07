# In LLM Reasoning, there is a Rational Gap on top of Value Misalignment

Significant progress have been made to align LLMs with a value function. We argue that, even when an LLM has been well aligned in (post-)training, it may still fail to take actions that best maximize the aligned value - on top of the value misalignment error, a rational gap exists, echoing recent rationality theory for reinforcement learning. Extensive experiments well support our argument.

## Overview

The rational gap of a policy $\pi_\theta$ is defined as the discrepancy between the utility achievable within the model's reachable distribution and the utility actually realised under sampling-based inference:

$$
\mathcal{R}(\pi_\theta) = \mathbb{E}_{x \sim \rho}\Big[ U(x, y_\theta^\circ) - \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} U(x, y) \Big].
$$

Empirically, we estimate it by sampling $K$ trajectories per prompt and computing

$$
\widehat{\mathcal{R}}_K(\pi_\theta) = \frac{1}{N}\sum_{i=1}^{N} \max_{k \in [K]} U(x_i, y_{i,k}) - \frac{1}{NK}\sum_{i=1}^{N}\sum_{k=1}^{K} U(x_i, y_{i,k}).
$$

This repository computes $\widehat{\mathcal{R}}_K$ across models, datasets, alignment stages, inference procedures, and prompt difficulty levels, in support of the paper's four hypotheses.

## Repository Structure

```
rational_gap/                       # this repository (code only)
├── configs/                        # experiment and path configuration
│   ├── paths.template.yaml         # template for local output paths
│   ├── models.yaml                 # model registry
│   ├── datasets.yaml               # dataset registry and prompt templates
│   └── experiments/
│       ├── h1.yaml                 # existence
│       ├── h2.yaml                 # independency of value alignment
│       ├── h3.yaml                 # benefits of reasoning mechanisms
│       └── h4.yaml                 # relationship with context length
├── src/
│   ├── sampling/                   # vLLM-based sampling layer
│   ├── verification/               # GSM8K / MATH / HumanEval verifiers
│   ├── metrics/                    # rational gap and decomposition
│   ├── pipeline/                   # measurement loop, cache, logging
│   └── plotting/                   # figures and tables
├── scripts/                        # entry-point shell scripts per H
├── tests/                          # unit tests for verifiers
└── requirements.txt
```

All runtime outputs are written **outside** this repository, to a directory configured in `configs/paths.yaml`:

```
rational_gap_outputs/               # outputs (NOT in git)
├── data/
│   ├── raw/                        # downloaded datasets
│   └── samples/                    # cached trajectories (gzipped JSONL)
├── results/
│   ├── h1/ h2/ h3/ h4/             # computed metrics (JSON)
│   └── figures/                    # rendered figures (PDF/PNG)
└── logs/
    ├── runs/                       # per-run human-readable logs
    ├── errors/                     # exception traces
    ├── verifier/                   # per-sample verifier audit logs
    └── compute_budget.jsonl        # cumulative GPU-hour log
```

## Installation

### Requirements

- Python 3.10+
- CUDA 12.1+ (for vLLM)
- 80GB GPU memory (tested on NVIDIA A800 80GB; smaller GPUs supported for ≤7B models)

### Setup

```bash
git clone https://github.com/<user>/rational_gap.git
cd rational_gap

# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure local paths
cp configs/paths.template.yaml configs/paths.yaml
# Edit configs/paths.yaml to point to your outputs directory
```

### Configuring `paths.yaml`

The file `configs/paths.yaml` (gitignored) tells the code where to write samples, results, and logs. A typical setting:

```yaml
outputs_root: "/data/your_username/rational_gap_outputs"
samples_dir: "${outputs_root}/data/samples"
results_dir: "${outputs_root}/results"
raw_data_dir: "${outputs_root}/data/raw"
logs_dir: "${outputs_root}/logs"
```

The output directory should have at least 20 GB of free space for the full set of experiments.

## Quick Start

A minimal end-to-end run on a small model and dataset to verify the pipeline:

```bash
python -m scripts.smoke_test --model qwen-1.5b --dataset gsm8k --num-prompts 50 --K 4
```

This samples 4 trajectories per prompt for 50 GSM8K prompts and prints the empirical rational gap. Total runtime should be under 5 minutes.

## Running Experiments

Each hypothesis has a dedicated entry script. Outputs are deterministic given the same seed and configuration.

### Calibration of $K$

Before running the main experiments, determine the appropriate sampling budget:

```bash
bash scripts/run_calibration.sh
```

This evaluates Tülu-3-8B on a 200-prompt GSM8K subset at $K \in \{1, 4, 16, 64, 256, 1024\}$ and reports $\hat{\mathcal{R}}_K$ alongside GPU-hour cost. The smallest $K$ achieving sufficient saturation is then used as the default in subsequent experiments (default `K=64`).

### H1: Existence

```bash
bash scripts/run_h1.sh
```

Measures $\hat{\mathcal{R}}_K$ across (Tülu-3-8B, Qwen-2.5-7B-Instruct, Llama-3.1-8B-Instruct) × (GSM8K, MATH, HumanEval), with three random seeds.

### H2: Independency of value alignment

```bash
bash scripts/run_h2.sh
```

Compares base model of Llama-3.1-8B and Tülu-3-8B at three post-training stages (SFT, DPO, RLVR) on GSM8K and MATH.

### H3: Benefits of reasoning mechanisms

```bash
bash scripts/run_h3.sh
```

Holds $\pi_\theta$ fixed at Tülu-3-8B and evaluates five inference mechanisms: direct sampling at $\tau \in \{0, 0.7, 1.0\}$, CoT prompting, self-consistency over 8 CoT samples, MCTS with budget-matched search, and oracle Best-of-$N$ ($N=64$).

### H4: Relationship with context length

```bash
bash scripts/run_h4.sh
```

Holds $\pi_\theta$ fixed at Tülu-3-8B and varies the maximum reasoning length $L \in \{0, 64, 128, 256, 512, 1024, 2048\}$ tokens on MATH and GSM8K via two-stage budget forcing, characterising how $\hat{\mathcal{R}}_K(L)$ evolves with the reasoning budget allocated at inference.

### Generating Figures

After all experiments complete:

```bash
python -m src.plotting.generate_all
```

Figures and tables are written to `${outputs_root}/results/figures/`.

## Reproducing Paper Results

To reproduce all main results from scratch:

```bash
bash scripts/run_calibration.sh   # ~10 GPU-hours
bash scripts/run_h1.sh            # ~25 GPU-hours
bash scripts/run_h2.sh            # ~15 GPU-hours
bash scripts/run_h3.sh            # ~15 GPU-hours
bash scripts/run_h4.sh            # ~20 GPU-hours
python -m src.plotting.generate_all
```

Total: approximately 85 GPU-hours on a single A800 80GB, plus a few minutes for figure generation.

The cumulative GPU-hour budget is logged to `${outputs_root}/logs/compute_budget.jsonl`.

## Caching

Sampling is the expensive step (~95% of runtime). Each sampling configuration—identified by `(model, dataset, K, temperature, seed)`—is cached as a gzipped JSONL file in `${outputs_root}/data/samples/`. Subsequent runs with the same configuration reuse the cache, so iterating on metrics or verifiers does not require resampling.

To force resampling, pass `--no-cache` to any script, or delete the relevant cache file.

## Verification

Three task-specific verifiers are implemented:

- **GSM8K** (`src/verification/gsm8k.py`): extracts the final numeric answer using a chain of fallback regex patterns (`####`, `\boxed{}`, last number) and performs exact match.
- **MATH** (`src/verification/math.py`): extracts `\boxed{}` content and checks symbolic equivalence with the ground truth via `math-verify` (a SymPy-based equivalence checker robust to common LaTeX variants).
- **HumanEval** (`src/verification/humaneval.py`): executes the model's completion against the provided unit tests in a sandboxed subprocess with a 5-second timeout.

Every verification decision is logged to `${outputs_root}/logs/verifier/` for post-hoc auditing. The unit test suite in `tests/` covers approximately 60 manually verified cases per dataset.

## Logging

Three categories of logs are written:

- **Run logs** (`logs/runs/`): human-readable timeline of each run, including configuration, cache status, sampling progress, and final metrics.
- **Verifier logs** (`logs/verifier/`): JSONL audit trail of every verifier decision, supporting the manual inspection of false negatives.
- **Compute budget** (`logs/compute_budget.jsonl`): cumulative GPU-hour usage per experiment.

## Limitations

The experimental scope is intentionally bounded:

- **Model scale**: 1.5B–14B parameters. Whether the patterns persist at the 70B+ scale is not addressed by these experiments.
- **Alignment trajectory**: H2 uses the Tülu-3 trajectory only. Conclusions are illustrative of one well-documented post-training pipeline rather than universal across all alignment recipes.
- **Utility**: all experiments use binary correctness on tasks with verifiable ground truth. Extension to continuous reward functions (e.g., reward models) is left for future work.

These limitations are discussed in the paper.

## Citation

```bibtex
@inproceedings{<key>,
  title  = {Rationality of LLM Reasoning: A Gap Beyond Value Alignment},
  author = {<authors>},
  year   = {<year>},
  booktitle = {<venue>},
}
```

## License

The code is released under the MIT License. The datasets used (GSM8K, MATH, HumanEval) retain their original licenses.

## Contact

For questions about the implementation, please open an issue or contact `<email>`.
