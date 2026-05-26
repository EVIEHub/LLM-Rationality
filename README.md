# In LLM Reasoning, there is Irrationality on top of Value Misalignment

This repository contains the code for measuring **rational value risk (RVR)** in large language model reasoning.

## Paper hypotheses

The paper evaluates four hypotheses:

- **H1.** Rational value risk is widespread. Across models and benchmarks, LLMs constantly generate high-utility candidates but fail to consistently deploy them.
- **H2.** Value alignment methods can reduce, but cannot eliminate, rational value risk. Rationality is not fully solved by value alignment alone.
- **H3.** Rational value risk is highly sensitive to the inference-time reasoning strategy, including sampling temperature and self-consistency. 
- **H4.** Longer reasoning can improve rationality, but its benefits diminish beyond a certain reasoning budget.


The repository also includes a **proprietary model** covering GPT-5.2, GPT-5.5, and DeepSeek-V4-Flash on MathArena benchmark.

## Quick start

```bash
git clone <repo-url> rational-value-risk
cd rational-value-risk

# 1. Install dependencies.
# This creates the conda environment, installs PyTorch + vLLM, and runs tests.
# Optional: set OUTPUTS_ROOT=/path/with/20GB before setup.
bash scripts/setup.sh

# 2. Configure local output path if OUTPUTS_ROOT was not set.
${EDITOR:-vim} configs/paths.yaml

# 3. Run a smoke test.
python -m scripts.smoke_test

# 4. Run the full reproduction.
bash scripts/run_all.sh --num-gpus $(nvidia-smi -L | wc -l)
```

## Experiment scripts

All scripts accept `--num-gpus N`, except the proprietary model evaluation, which is API-based. Result JSONs are written under `${outputs_root}/results/`.

| Experiment | Script | Coverage |
|---|---|---|
| H1: RVR is widespread | `bash scripts/run_h1_panel.sh` | Open-weight models across preference, math, and code benchmarks |
| H2: value alignment pipeline | `bash scripts/run_h2_panel.sh` | Tülu-3 SFT/DPO/RLVR stages at 8B and 70B |
| H3: inference-time reasoning strategies | `bash scripts/run_h3_panel.sh` | Temperature sweep and self-consistency |
| H4: reasoning-length budget | `bash scripts/run_h4_panel.sh` | Reasoning lengths `T = {0, 64, 128, 256, 512, 1024, 2048}` |
| Proprietary deployment evaluation | `bash scripts/run_api_panel.sh` or `RUN_API=1 bash scripts/run_all.sh` | GPT-5.2, GPT-5.5, and DeepSeek-V4-Flash on deployment-style benchmarks |

The proprietary deployment evaluation requires API credentials in `~/.config/rg-gap.env`:

```bash
TP_BASE_URL=...
TP_API_KEY=...

DS_BASE_URL=...
DS_API_KEY=...

HF_TOKEN=...
```

## Outputs

All runtime data is written outside the repository to the `outputs_root` path configured in `configs/paths.yaml`.

```text
${outputs_root}/
├── data/samples/                # cached sampled answers
├── data/samples/apiresume_*.jsonl
├── results/h1/                  # H1 metrics
├── results/h2/                  # H2 metrics
├── results/h3/                  # H3 metrics
├── results/h4/                  # H4 metrics
└── logs/                        # run logs and verifier audit logs
```

Sampling is idempotent: if a cell has already generated samples under the same configuration, later runs reuse the cached candidates and recompute verification or metrics as needed.

## Configuration files

### `configs/paths.yaml`

Local output paths. This file is gitignored. See `paths.template.yaml`.

### `configs/models.yaml`

Model registry. Each entry specifies the model identifier, model family, prompt mode, and API settings when applicable.

### `configs/datasets.yaml`

Dataset registry. Each entry specifies the dataset, prompt format, verifier type, and answer extraction configuration.



## Tests

Run the verifier and metric tests with:

```bash
pytest tests/ -q
```

The tests cover answer extraction, verifier behaviour, metric computation, and selected pipeline utilities.

## Environment

The default tested environment is:

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1 |
| CUDA | 12.4 |
| vLLM | 0.6.3 |
| Transformers | 4.45.2 |

The main open-weight experiments were run on NVIDIA A800 80GB GPUs.

## Repository layout

```text
src/
├── sampling/        # local and API sampling
├── verification/    # dataset-specific verifiers
├── metrics/         # REU, AEU, RVR, bootstrap CI
├── pipeline/        # paths, cache, logging
└── plotting/        # figures, tables, appendix outputs

scripts/
├── run_h1_panel.sh
├── run_h2_panel.sh
├── run_h3_panel.sh
├── run_h4_panel.sh
├── run_all.sh
├── smoke_test.py
├── build_tables.py
├── build_figures.py
└── build_appendix.py

configs/
├── paths.yaml
├── models.yaml
└── datasets.yaml

tests/
└── ...
```

## Troubleshooting

### vLLM hangs at 0% prompts

Remove overly restrictive batching settings such as very small `RG_MAX_NUM_SEQS`.

### CUDA error at a specific prompt

Try switching the attention backend:

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

### Out-of-memory on long reasoning-budget cells

Try reducing the KV-cache memory footprint:

```bash
export RG_KV_DTYPE=fp8_e4m3
```

If this is unsupported in the local vLLM build, increase CPU swap space:

```bash
export RG_SWAP_SPACE_GB=32
```

### Need to re-run verification without re-sampling

Use cached generations and run the verifier again:

```bash
python -m scripts.reverify_from_cache \
  --samples <cache.jsonl.gz> \
  --result <result.json> \
  --dataset <dataset_name>
```

## License

MIT License for the code. Datasets retain their original licenses.