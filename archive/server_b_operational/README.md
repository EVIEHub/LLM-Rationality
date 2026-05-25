# server_b_operational/

Operational shell wrappers used during the original paper run on AutoDL server B (4 × A800-80GB, then 1-GPU instance). **Not** part of the canonical reproduction path — for that, use `scripts/run_h{1,2,3,4,5}_panel.sh` directly. These are kept as **worked examples** of recurring patterns: multi-stage rotation, GPU/disk pressure, vllm runtime bugs, splice intercepts, API retries, etc.

## Canonical examples (top-level here)

| Script | Pattern demonstrated |
|---|---|
| `h2_70b_robust.sh` | Multi-stage trajectory with pre-download retry per stage + automatic weight rotation. Solves the "vllm auto-download is not retry-safe" issue. |
| `h4_2models_panel.sh` | Sequential multi-model H4 panel with predownload-then-rotate. |
| `h4_70b_rlvr_matharena_splice.sh` | **Splice intercept**: kill a running panel just before its rotation step so you can reuse the loaded weights for an additional sub-experiment without redownload. |
| `h4_llama8b_after_qwen.sh` | "Wait-for-flag-then-run" pattern for queueing dependent jobs. |
| `h4_2cells_1gpu_v4.sh` | Single-GPU retry of failed H4 cells with `VLLM_ATTENTION_BACKEND=XFORMERS` to dodge FlashAttention kernel bugs. |
| `h4_llama_only_v6.sh` | Single-GPU long-budget cell with `RG_SWAP_SPACE_GB=32` workaround for KV-cache OOM when FP8 KV isn't available. |
| `h1_72b_pref.sh` | 72B preference cell with DeepSeek-V4-Flash as the API judge (`--judge api`). |
| `h1_72b_pref_self.sh` | 72B preference cell with strict-self judge (`--judge self`, default fingerprint). |
| `h2_rlvr_uf_14b.sh` | Single-cell H2 retry with independent 14B-judge (`--judge-local-model qwen2.5-14b-instruct`). |
| `humaneval_72b_awq.sh` | Single-GPU AWQ-INT4 fallback for 72B HumanEval (the bf16 TP=4 path hits an all-reduce kernel bug). |
| `recover3_72b.sh` | Recovery wrapper that re-launches failed cells with GPU-drain guard between TP runs. |
| `redo_alpaca_deepseek_v2.sh` | API-judge retry with the patched `_RETRYABLE_STATUS` (handles HTTP 544). |
| `h5_gpt55_lcb_retry_B.sh` | Cross-server migration of an API panel + resume from cache. |

## iterations/

The intermediate versions (`*_v1.sh`, `*_v2.sh`, `*_v3.sh`, …) leading up to the canonical examples above. Kept for full provenance of what was tried; not useful as templates. Each iteration's failure mode is what motivated the next iteration — read top→bottom of an iteration chain if you want to see the debugging path.

## Why these scripts exist outside `scripts/`

The canonical `scripts/run_*_panel.sh` are the *declarative* reproduction path: run the panel, get the results. The wrappers here are *operational* — they handle the awkward bits of real cluster runs (disk-rotation timing, gateway timeouts, single-host vs cross-host orchestration, partial-failure recovery). They reference the canonical panels but wrap them with environment-specific glue.

If you're running the project fresh, start with `scripts/run_all.sh`. If you hit a specific failure mode that matches one of the patterns above, the wrapper here is a worked example you can adapt.
