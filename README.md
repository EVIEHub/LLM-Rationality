# Rational Gap of LLM Reasoning

Reproducibility code for the paper *"In LLM Reasoning, there is a Rational Gap on top of Value Misalignment"*.

We measure how much of an aligned LLM's apparent value-shortfall is **reachable** by its current policy but **unrealised** under inference. Formally, with $K$ samples per prompt:

$$
\widehat{\mathcal{R}}_K(\pi_\theta) \;=\; \tfrac{1}{N}\!\sum_{i=1}^N \max_{k\in[K]} U(x_i,y_{i,k}) \;-\; \tfrac{1}{NK}\!\sum_{i=1}^{N}\!\sum_{k=1}^{K} U(x_i,y_{i,k}).
$$

Hypotheses:
- **H1** — saturation curve $\widehat{\mathcal{R}}_K$ vs $K$ across open models on math/code/preference tasks
- **H2** — Tülu-3 trajectory (SFT → DPO → RLVR) at 8B and 70B
- **H3** — inference procedures (direct $\tau$-sweep, self-consistency)
- **H4** — reasoning-budget sweep $L \in \{0,64,128,256,512,1024,2048\}$
- **H5** — closed-weight frontier APIs (gpt-5.2-chat, gpt-5.5, deepseek-v4-flash) on deployment datasets

## Quick start

```bash
git clone <repo-url> rational-gap && cd rational-gap

# 1. one-shot install (creates conda env "rg-gap", installs PyTorch + vLLM, runs tests)
#    Optional: set OUTPUTS_ROOT=/path/with/20GB before setup.
bash scripts/setup.sh

# 2. configure local outputs path (~20 GB free), if you did not set OUTPUTS_ROOT
${EDITOR:-vim} configs/paths.yaml    # set outputs_root

# 3. smoke test (~3 min on a single small GPU)
python -m scripts.smoke_test

# 4. full reproduction (~85 GPU-hours on 1×A800-80GB; run inside tmux)
bash scripts/run_all.sh --num-gpus $(nvidia-smi -L | wc -l)
```

## Per-hypothesis scripts

All accept `--num-gpus N` (H5 ignores it because it is API-only). Result JSONs land under `${outputs_root}/results/`; H5 writes its H1/H3 cells into `results/h1/` and `results/h3/`.

| Hypothesis | Script | Coverage | ~Time on 1×A800 |
|---|---|---|---|
| H1 saturation | `bash scripts/run_h1_panel.sh` | 3 open 8B models × 7 datasets | ~25 h |
| H2 trajectory | `bash scripts/run_h2_panel.sh` | Tülu-3 SFT/DPO/RLVR × {8B, 70B} × dev+deployment+preference | ~15 h (8B) + ~15 h (70B if `STAGES="tulu3-70b-sft tulu3-70b-dpo tulu3-70b-rlvr"`) |
| H3 procedures | `bash scripts/run_h3_panel.sh` | 3 models × 7 datasets × {direct τ=0/0.7/1.0, sc n=2…32} | ~15 h |
| H4 reasoning budget | `bash scripts/run_h4_panel.sh` | 1 model × 4 datasets × 7 L values (default `MODEL=tulu3-8b-rlvr`) | ~20 h |
| H5 API panel | `bash scripts/run_h5_panel.sh` | 3 hosted models × {matharena, livecodebench} × H1+H3 | depends on quota |

H5 is opt-in (`RUN_H5=1 bash scripts/run_all.sh`) — needs API credentials in `~/.config/rg-gap.env`:

```
TP_BASE_URL=…   TP_API_KEY=…        # chivier proxy (gpt-5.x)
DS_BASE_URL=…   DS_API_KEY=…        # DeepSeek official metered API
HF_TOKEN=…                          # for gated meta-llama models
```

## Outputs

All runtime data is written **outside** the repo, to `outputs_root` in `configs/paths.yaml`:

```
${outputs_root}/
├── data/samples/           # cached trajectories (v2_<ds>_<model>_K<K>_<fp>.jsonl.gz)
├── data/samples/apiresume_*.jsonl    # per-(prompt, k) API resume cache
├── results/h{1,2,3,4}/     # computed metrics (JSON); H5 API cells reuse h1/h3
└── logs/{runs,verifier}/   # per-run logs + per-decision audit trail
```

Sampling is idempotent: cache-hit cells skip generation and only re-verify.

## Configuration

### `configs/paths.yaml` (gitignored)
Single source of truth for output paths. See `paths.template.yaml`.

### `configs/models.yaml`
Model registry. Each entry has `hf_id`, `family`, `prompt_mode`, and (for API models) `api_base_url_env` + `api_key_env`. Add a model by appending one entry.

### `configs/datasets.yaml`
Dataset registry. Includes the verifier choice (`gsm8k`, `math`, `humaneval`, `livecodebench`, `matharena`, `bbh`, `self_judge`).

### Env vars (read by `_common.sh` / `vllm_runner.py`)

| Var | Effect | When to use |
|---|---|---|
| `RG_TP=N` | tensor-parallel size | 70B-scale models |
| `RG_PP=N` | pipeline-parallel size | PCIe-only multi-GPU box |
| `RG_DISABLE_CUSTOM_AR=1` | NCCL fallback for all-reduce | TP all-reduce CUDA bug (72B HumanEval) |
| `RG_MAX_NUM_SEQS=N` | smaller vLLM batch | **Use sparingly** — small values can hang vLLM scheduling |
| `RG_KV_DTYPE=fp8_e4m3` | half-precision KV cache | long-budget cells OOMing (e.g. 8B + L=2048 + K=64). May fail on some vllm-dev builds with a Triton `CompilationError`; fall back to `RG_SWAP_SPACE_GB`. |
| `RG_SWAP_SPACE_GB=32` | grow vLLM CPU swap pool from the 4 GiB default | long-budget cells when fp8 KV is unavailable — GPU spills to CPU instead of OOMing |
| `VLLM_ATTENTION_BACKEND=XFORMERS` | switch from FlashAttention | dodging FlashAttn kernel bugs (deterministic crashes at specific prompts) |
| `HF_ENDPOINT=https://hf-mirror.com` | HF mirror | clusters without direct huggingface.co access |

## Tests

```bash
pytest tests/ -q          # full baseline after setup.sh installs vLLM/transformers
```

Three verifiers: `src/verification/{gsm8k,math,humaneval,livecodebench,matharena,bbh,self_judge,api_judge}.py`. Each call is appended to `${outputs_root}/logs/verifier/` for post-hoc audit.

### Tested environments

| Component | `setup.sh` default (cu124) | Original-run rg-gap env (cu121) |
|---|---|---|
| Python | 3.11 | 3.11 |
| torch | 2.5.1 + cu124 wheels | 2.4.0 + cu121 wheels |
| vllm | 0.6.3 | 0.6.3 |
| transformers | 4.45.2 | 4.45.2 |

Both are confirmed working. For exact reproduction of the original run, use the cu121 toolchain:
```bash
PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup.sh
```

## Repository layout

```
src/
├── sampling/        # vllm_runner (local), api_runner (hosted), retry policy
├── verification/    # one verifier per dataset + self_judge + api_judge
├── metrics/         # rational_gap (U_circ, U_bar, R_hat) + bootstrap CI
├── pipeline/        # paths, cache, logging, audit
└── plotting/        # plot_h{1,2,3,4}, tables, appendix
scripts/             # run_h{1,2,3,4,5}_panel.sh + run_all.sh + setup.sh + reverify_from_cache.py
configs/             # paths.yaml, models.yaml, datasets.yaml
tests/               # verifier unit tests
archive/             # operational wrappers from the original run (study-specific)
```

The `archive/server_b_operational/` directory keeps wrapper scripts used during the original run (70B trajectory orchestration, splice intercepts, API retries, etc.) — useful as worked examples but not part of the reproduction path.

## Troubleshooting

- **vLLM hangs at 0 % prompts** — `RG_MAX_NUM_SEQS=N` with small N can deadlock the scheduler; remove the env var.
- **CUDA illegal memory at a specific prompt** — try `VLLM_ATTENTION_BACKEND=XFORMERS`.
- **OOM on long-L cells (8B + L=2048 + K=64)** — first try `RG_KV_DTYPE=fp8_e4m3` (halves KV memory). If your vllm build raises a Triton `CompilationError` on the fp8 path, fall back to `RG_SWAP_SPACE_GB=32` (or higher) to give vLLM enough CPU swap to spill KV blocks.
- **Disk full mid-trajectory (H2-70B)** — the `run_h2_panel.sh` rotation deletes a stage's weights before the next stage downloads. Don't keep stale large models in `${HF_HOME}/hub`.
- **Lost a deterministic-verifier result (zeroes everywhere)** — `python -m scripts.reverify_from_cache --samples <cache.jsonl.gz> --result <result.json> --dataset <name>` re-runs verification on the cached trajectories without re-sampling.

## Citation

```bibtex
@inproceedings{rational-gap-2026,
  title     = {In LLM Reasoning, there is a Rational Gap on top of Value Misalignment},
  author    = {<authors>},
  year      = {2026},
  booktitle = {<venue>},
}
```

## License

MIT (this code). Datasets retain their original licenses (GSM8K, MATH, HumanEval, MathArena, LiveCodeBench, UltraFeedback, AlpacaEval, BBH).
