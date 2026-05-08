#!/bin/bash
# H2 panel orchestrator — Tülu-3 alignment trajectory.
#
# Cells: 3 stages (SFT, DPO, RLVR) × 2 datasets (gsm8k, math) × seed=0 = 6.
# All cells run in chat mode using the same chat template; see
# AGENT/methodology/hypotheses.md for why we drop the Tülu-3 base/raw model
# from this trajectory (avoids confounding chat vs few-shot prompt mode).
#
# Strategy: sequential per stage with disk rotation. The 50 GB autodl-tmp
# volume cannot hold all the 8B models concurrently, so we delete the
# previous stage's weights before downloading the next.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/run_h2_panel.sh [--num-gpus N]
#
# Examples:
#   bash scripts/run_h2_panel.sh                # auto-detect GPUs
#   bash scripts/run_h2_panel.sh --num-gpus 2   # force 2 GPUs (one per dataset)
#   bash scripts/run_h2_panel.sh --num-gpus 1   # serial
#
# Run inside tmux:
#   tmux new-session -d -s h2_panel \
#       "bash scripts/run_h2_panel.sh --num-gpus 2 \
#        > outputs/logs/h2_panel.log 2>&1"
# -----------------------------------------------------------------------------
#
# H2 has 2 datasets per stage, so additional GPUs beyond 2 are idle within a
# stage; specify --num-gpus 2 (or 1 for serial) for the lowest GPU-hour cost.
#
# Stages:
#   Stage 0: RLVR — usually a cache HIT from H1 (chat-mode samples reused).
#                   Re-runs verifier + aggregates only; no GPU work.
#   Stage 1: SFT  — delete H1 leftover instruct models, download SFT, run.
#   Stage 2: DPO  — delete SFT, download DPO, run.

set -uo pipefail
# NOTE: no `-e` so a cell failure logs and we continue; the cache makes
# restarts cheap.

# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_parse_num_gpus "$@"
set -- "${RG_POSITIONAL[@]}"
rg_activate_env
rg_setup_hf

OUTPUTS_ROOT="$(rg_outputs_root)"
LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h2"

declare -A NUM_PROMPTS=(
    [gsm8k]=1319
    [math]=1000
    [humaneval]=164
)
SEED="${SEED:-0}"
K="${K:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM="${GPU_MEM:-0.85}"

# H2 covers the same dataset suite as H1, so the trajectory comparison
# (alignment vs reachability) is anchored in identical evaluation surface.
# RLVR cells are typically cache hits from H1 if H1 was run first with
# the same K, num_prompts, max_tokens, and seed.
DATASETS=(gsm8k math humaneval)
N_DATASETS=${#DATASETS[@]}
# Cap GPUs per stage at N_DATASETS — extra GPUs would be idle inside a
# stage. The user-specified NUM_GPUS still controls how many we are
# willing to use; we never exceed it.
GPUS_PER_STAGE=$(( NUM_GPUS < N_DATASETS ? NUM_GPUS : N_DATASETS ))

PANEL_T0=$(date +%s)
rg_log "=== H2 panel ==="
rg_log "Datasets:       ${DATASETS[*]}"
rg_log "Seed:           ${SEED}"
rg_log "GPUs available: ${NUM_GPUS}"
rg_log "GPUs per stage: ${GPUS_PER_STAGE}"
rg_log "K=${K}, max_tokens=${MAX_TOKENS}, gpu_mem=${GPU_MEM}"

# -----------------------------------------------------------------------------
# Run a stage: NUM_GPUS_PER_STAGE datasets in parallel; if the user has
# only 1 GPU, datasets run serially.
# -----------------------------------------------------------------------------
run_stage() {
    local model="$1"   # alias e.g. tulu3-8b-sft
    local label="$2"   # human-readable for logs
    echo
    rg_log "--- Stage: ${label} (${model}) ---"
    local stage_t0=$(date +%s)

    local i=0
    while [ "$i" -lt "$N_DATASETS" ]; do
        local pids=()
        local j=0
        while [ "$j" -lt "$GPUS_PER_STAGE" ] && [ $((i + j)) -lt "$N_DATASETS" ]; do
            local ds="${DATASETS[$((i + j))]}"
            local gpu_idx="$j"
            local np="${NUM_PROMPTS[$ds]}"
            local logfile="${LOG_DIR}/h2_${model}_${ds}.log"
            echo "  [GPU ${gpu_idx}] $(date +%H:%M:%S) ${model} × ${ds}  N=${np}"
            CUDA_VISIBLE_DEVICES="${gpu_idx}" python -m scripts.run_h2 \
                --model "${model}" \
                --dataset "${ds}" \
                --seed "${SEED}" \
                --K "${K}" \
                --num-prompts "${np}" \
                --max-tokens "${MAX_TOKENS}" \
                --gpu-mem "${GPU_MEM}" \
                > "${logfile}" 2>&1 &
            pids+=($!)
            j=$((j + 1))
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "  (cell PID $pid exited non-zero — check log)"
        done
        i=$((i + GPUS_PER_STAGE))
    done

    local stage_dur=$(( $(date +%s) - stage_t0 ))
    rg_log "--- ${label} done in ${stage_dur}s ---"
    df -h "${HF_HOME}" 2>/dev/null | head -3 || true
}

# Remove a downloaded HF model to free disk before the next stage.
# HF cache uses '--' (DOUBLE hyphen) to separate org/name in dir names,
# so 'org/name' -> 'models--org--name'.
remove_model_cache() {
    local hf_id="$1"
    local cache_path="${HF_HOME}/hub/models--$(echo "$hf_id" | sed 's|/|--|g')"
    if [ -d "$cache_path" ]; then
        echo "  removing ${cache_path}"
        rm -rf "${cache_path}"
    fi
}

# -----------------------------------------------------------------------------
# Stage 0: RLVR via cache hit
# -----------------------------------------------------------------------------
echo
rg_log "=== Stage 0: RLVR via H1 cache hits (no GPU work) ==="
for ds in "${DATASETS[@]}"; do
    np="${NUM_PROMPTS[$ds]}"
    python -m scripts.run_h2 \
        --model tulu3-8b-rlvr \
        --dataset "${ds}" \
        --seed "${SEED}" \
        --K "${K}" \
        --num-prompts "${np}" \
        --max-tokens "${MAX_TOKENS}" \
        > "${LOG_DIR}/h2_tulu3-8b-rlvr_${ds}.log" 2>&1 \
        || echo "  RLVR ${ds} cell failed — check log (likely no H1 cache)"
done
rg_log "RLVR stage done"

# -----------------------------------------------------------------------------
# Disk prep: delete H1-only models we don't need for H2
# -----------------------------------------------------------------------------
echo
rg_log "=== freeing disk: removing H1-only models ==="
remove_model_cache "Qwen/Qwen2.5-7B-Instruct"
remove_model_cache "meta-llama/Llama-3.1-8B-Instruct"
df -h "${HF_HOME}" 2>/dev/null | head -3 || true

# -----------------------------------------------------------------------------
# Stage 1: SFT
# -----------------------------------------------------------------------------
echo
rg_log "=== downloading SFT ==="
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B-SFT 2>&1 | tail -3 || \
    echo "(download warning — may already be cached)"
run_stage "tulu3-8b-sft" "Stage 1: SFT"

echo
rg_log "=== freeing SFT cache before DPO download ==="
remove_model_cache "allenai/Llama-3.1-Tulu-3-8B-SFT"

# -----------------------------------------------------------------------------
# Stage 2: DPO
# -----------------------------------------------------------------------------
echo
rg_log "=== downloading DPO ==="
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B-DPO 2>&1 | tail -3 || \
    echo "(download warning — may already be cached)"
run_stage "tulu3-8b-dpo" "Stage 2: DPO"

# DPO weights are not deleted here — H4 uses tulu3-8b-rlvr (different model)
# and the H4 panel script handles its own disk prep.

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
echo
rg_log "=== H2 PANEL DONE in ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr) ==="
echo
rg_log "Per-cell results: ${OUTPUTS_ROOT}/results/h2/"
ls -la "${OUTPUTS_ROOT}/results/h2/" 2>/dev/null || echo "(no results dir)"
echo
df -h "${HF_HOME}" 2>/dev/null | head -3 || true
