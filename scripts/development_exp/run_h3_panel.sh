#!/bin/bash
# H3 panel orchestrator — rational gap of each inference procedure.
#
# Two phases:
#
#   Phase 1 (GPU): direct sampling at tau in {0, 0.7, 1.0}
#       - tau=1.0 cells reuse H1 sample cache (no GPU work)
#       - tau=0.0 cells use K=1 (greedy is deterministic)
#       - tau=0.7 cells require fresh K=64 sampling
#     Cells: 3 models x 3 datasets x 3 tau = 27 cells.
#     Iterates over models with disk rotation since 3x 8B-class
#     models do not co-fit on a typical 50-100 GB autodl-tmp volume.
#
#   Phase 2 (CPU): self-consistency at n in {2, 4, 8, 16, 32}
#     Cells: 3 models x 2 datasets (humaneval skipped — no answer-key
#     extractor for code) x 5 n = 30 cells.
#     Reuses the H1 K=64 cache via bootstrap-resampled SC draws; no GPU.
#
# Requires: H1 panel results already on disk (Phase 1's tau=1.0 cache
# hits and Phase 2's bootstrap source both depend on it).
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/development_exp/run_h3_panel.sh [--num-gpus N]
#
# Examples:
#   bash scripts/development_exp/run_h3_panel.sh                # auto-detect
#   bash scripts/development_exp/run_h3_panel.sh --num-gpus 2
#
# Run inside tmux:
#   tmux new-session -d -s h3_panel \
#       "bash scripts/development_exp/run_h3_panel.sh --num-gpus 2 \
#        > outputs/logs/h3_panel.log 2>&1"
# -----------------------------------------------------------------------------
#
# Wall-time on 8B-class models (2 GPUs, ~50 GB disk):
#   Phase 1 (direct): ~3 hr  (math tau=0.7 K=64 cell at ~40-75 min is
#                              the long pole per model phase;
#                              3 model phases * (download + cells)
#                              ~ 1 hr/phase)
#   Phase 2 (SC):     ~5 min CPU
#   Total:            ~3 hr

set -uo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/../_common.sh"
rg_parse_num_gpus "$@"
set -- "${RG_POSITIONAL[@]}"
rg_activate_env
rg_setup_hf

OUTPUTS_ROOT="$(rg_outputs_root)"
LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h3"

# -----------------------------------------------------------------------------
# Config.
# -----------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct)
fi
DATASETS=(gsm8k math humaneval)
TAUS=(1.0 0.0 0.7)
SC_DATASETS=(gsm8k math)   # humaneval skipped (no answer-key extractor)
SC_NS=(2 4 8 16 32)
SEED="${SEED:-0}"
K="${K:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM="${GPU_MEM:-0.7}"   # H4-style conservative default for long contexts

declare -A NUM_PROMPTS=(
    [gsm8k]=1319
    [math]=1000
    [humaneval]=164
)
declare -A HF_IDS=(
    [tulu3-8b-rlvr]="allenai/Llama-3.1-Tulu-3-8B"
    [qwen2.5-7b-instruct]="Qwen/Qwen2.5-7B-Instruct"
    [llama3.1-8b-instruct]="meta-llama/Llama-3.1-8B-Instruct"
)

rg_log "=== H3 panel ==="
rg_log "Models:    ${MODELS[*]}"
rg_log "Datasets:  ${DATASETS[*]}"
rg_log "Taus:      ${TAUS[*]}"
rg_log "SC n:      ${SC_NS[*]}"
rg_log "GPUs:      ${NUM_GPUS}"

remove_model_cache() {
    local hf_id="$1"
    local p="${HF_HOME}/hub/models--$(echo "$hf_id" | sed 's|/|--|g')"
    if [ -d "$p" ]; then
        rg_log "  removing $p"
        rm -rf "$p"
    fi
}

run_direct_cell() {
    local gpu="$1"
    local model="$2"
    local dataset="$3"
    local tau="$4"
    local np="${NUM_PROMPTS[$dataset]}"
    local logfile="${LOG_DIR}/h3_${model}_${dataset}_t${tau}.log"
    echo "[GPU $gpu] $(date +%H:%M:%S) START direct ${model} x ${dataset} tau=${tau} N=${np}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu python -m scripts.run_h3 \
        --model "${model}" --dataset "${dataset}" --seed "${SEED}" --K "${K}" \
        --procedure direct --tau "${tau}" \
        --num-prompts "${np}" --max-tokens "${MAX_TOKENS}" --gpu-mem "${GPU_MEM}" \
        > "$logfile" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        echo "[GPU $gpu] $(date +%H:%M:%S) DONE  direct ${model} x ${dataset} tau=${tau} (${dur}s)"
    else
        echo "[GPU $gpu] $(date +%H:%M:%S) FAIL  direct ${model} x ${dataset} tau=${tau} (rc=$rc, log: $logfile)"
    fi
}

run_sc_cell() {
    # CPU only — uses CUDA_VISIBLE_DEVICES= to mask GPUs entirely.
    local model="$1"
    local dataset="$2"
    local n="$3"
    local np="${NUM_PROMPTS[$dataset]}"
    local logfile="${LOG_DIR}/h3_${model}_${dataset}_sc_n${n}.log"
    echo "[CPU] $(date +%H:%M:%S) START SC ${model} x ${dataset} n=${n}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES= python -m scripts.run_h3 \
        --model "${model}" --dataset "${dataset}" --seed "${SEED}" --K "${K}" \
        --procedure sc --sc-n "${n}" \
        --num-prompts "${np}" --max-tokens "${MAX_TOKENS}" \
        > "$logfile" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        echo "[CPU] $(date +%H:%M:%S) DONE  SC ${model} x ${dataset} n=${n} (${dur}s)"
    else
        echo "[CPU] $(date +%H:%M:%S) FAIL  SC ${model} x ${dataset} n=${n} (rc=$rc, log: $logfile)"
    fi
}

PANEL_T0=$(date +%s)

# ===========================================================================
# Phase 1: GPU work — direct sampling.
# Model-outer with disk rotation. Within each model phase, run
# 3 datasets x 3 taus = 9 cells in batches of NUM_GPUS.
# ===========================================================================
rg_log "============================================================"
rg_log "Phase 1: direct sampling (GPU, ~3 hr)"
rg_log "============================================================"

for model in "${MODELS[@]}"; do
    rg_log "--- direct phase for ${model} ---"
    rg_log "downloading ${HF_IDS[$model]}"
    huggingface-cli download "${HF_IDS[$model]}" 2>&1 | tail -3 || true
    df -h "$HF_HOME" 2>/dev/null | tail -1

    # Build cell list (dataset, tau).
    CELLS=()
    for tau in "${TAUS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            CELLS+=("${ds}|${tau}")
        done
    done

    # Batch loop.
    for ((i=0; i<${#CELLS[@]}; i+=NUM_GPUS)); do
        pids=()
        for ((j=0; j<NUM_GPUS; j++)); do
            idx=$((i + j))
            if [ "$idx" -lt "${#CELLS[@]}" ]; then
                IFS='|' read -r ds tau <<< "${CELLS[$idx]}"
                run_direct_cell "$j" "$model" "$ds" "$tau" &
                pids+=($!)
            fi
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
    done

    rg_log "freeing ${HF_IDS[$model]}"
    remove_model_cache "${HF_IDS[$model]}"
done

# ===========================================================================
# Phase 2: CPU work — self-consistency bootstrap.
# Sequential across cells (each is ~30 s; not worth parallelising).
# ===========================================================================
rg_log "============================================================"
rg_log "Phase 2: self-consistency bootstrap (CPU, ~5 min)"
rg_log "============================================================"

for model in "${MODELS[@]}"; do
    for ds in "${SC_DATASETS[@]}"; do
        for n in "${SC_NS[@]}"; do
            run_sc_cell "$model" "$ds" "$n"
        done
    done
done

# ===========================================================================
# Summary.
# ===========================================================================
PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
GPU_HRS=$(awk "BEGIN{printf \"%.2f\", ${NUM_GPUS} * ${PANEL_DUR} / 3600}")
echo
rg_log "=== H3 PANEL DONE ==="
rg_log "Wall time:        ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr)"
rg_log "GPU-hr billable:  ${GPU_HRS} (${NUM_GPUS} GPUs * wall hr)"
rg_log "Per-cell results: ${OUTPUTS_ROOT}/results/h3/"
ls -la "${OUTPUTS_ROOT}/results/h3/" 2>/dev/null | grep -E "\.json$" | wc -l | xargs echo "  total cells:"
