#!/bin/bash
# H4 panel orchestrator — Tülu-3-RLVR × {gsm8k, math} × 7 reasoning-length
# values (L) × seed=0.
#
# 14 cells total. Each cell uses the two-stage budget-forced procedure
# from src/sampling/inference_procedures.py (s1-style), where L is the
# fixed reasoning budget before the model is forced to emit "Final answer:".
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/run_h4_panel.sh [--num-gpus N]
#
# Examples:
#   bash scripts/run_h4_panel.sh                # auto-detect GPUs
#   bash scripts/run_h4_panel.sh --num-gpus 3   # 3 GPUs (5 batches of ~3)
#   bash scripts/run_h4_panel.sh --num-gpus 1   # serial, all 14 cells
#
# Run inside tmux:
#   tmux new-session -d -s h4_panel \
#       "bash scripts/run_h4_panel.sh --num-gpus 3 \
#        > outputs/logs/h4_panel.log 2>&1"
# -----------------------------------------------------------------------------
#
# Wall-time estimate (Tülu-3-8B, num_prompts=500, K=64):
#   - L sum across 7 values: 4032 reasoning + 7×64 answer ≈ 4480 tokens/sample.
#   - 500 prompts × 64 K × ~640 avg tokens ≈ 20M tokens/cell average.
#   - 5K tok/s on Tülu-3-8B → ~67 min/cell average for gsm8k.
#   - 14 cells × 67 min / 3 GPUs ≈ 5 hr wall.
#
# To shave time further: drop num_prompts to 250 (~2.5 hr) or trim L values.
# Override via env:  NUM_PROMPTS=250 LS="0 256 1024" bash scripts/run_h4_panel.sh

set -uo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_parse_num_gpus "$@"
set -- "${RG_POSITIONAL[@]}"
rg_activate_env
rg_setup_hf

OUTPUTS_ROOT="$(rg_outputs_root)"
LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h4"

MODEL="${MODEL:-tulu3-8b-rlvr}"
read -ra DATASETS <<< "${DATASETS:-gsm8k math}"
read -ra LS <<< "${LS:-0 64 128 256 512 1024 2048}"
SEED="${SEED:-0}"
K="${K:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
GPU_MEM="${GPU_MEM:-0.7}"  # safer default for long L cells

# Build cell list as (dataset, L). L-major order: each batch of NUM_GPUS
# cells uses similar reasoning length, so per-batch wall time is balanced
# even when NUM_GPUS does not divide len(L) × len(DATASETS).
CELLS=()
for L in "${LS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        CELLS+=("${ds}|${L}")
    done
done

TOTAL_BATCHES=$(( (${#CELLS[@]} + NUM_GPUS - 1) / NUM_GPUS ))
rg_log "=== H4 panel ==="
rg_log "Model:       ${MODEL}"
rg_log "Datasets:    ${DATASETS[*]}"
rg_log "L values:    ${LS[*]}"
rg_log "Cells:       ${#CELLS[@]}"
rg_log "GPUs:        ${NUM_GPUS}"
rg_log "Batches:     ${TOTAL_BATCHES}"
rg_log "K=${K}, num_prompts=${NUM_PROMPTS}, seed=${SEED}, gpu_mem=${GPU_MEM}"

run_one_cell() {
    local gpu_idx="$1"
    local cell_spec="$2"
    IFS='|' read -r ds L <<< "$cell_spec"
    local logfile="${LOG_DIR}/h4_gpu${gpu_idx}_${ds}_L${L}.log"
    echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${ds} L=${L}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu_idx" python -m scripts.run_h4 \
        --model "${MODEL}" \
        --dataset "${ds}" \
        --L "${L}" \
        --seed "${SEED}" \
        --K "${K}" \
        --num-prompts "${NUM_PROMPTS}" \
        --gpu-mem "${GPU_MEM}" \
        > "$logfile" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) DONE  ${ds} L=${L}  (${dur}s)"
    else
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) FAIL  ${ds} L=${L}  (rc=$rc, log: $logfile)"
    fi
}

PANEL_T0=$(date +%s)
batch_idx=0
for ((i=0; i<${#CELLS[@]}; i+=NUM_GPUS)); do
    batch_idx=$((batch_idx + 1))
    echo
    rg_log "=== Batch ${batch_idx}/${TOTAL_BATCHES} ==="
    pids=()
    for ((j=0; j<NUM_GPUS; j++)); do
        cell_idx=$((i + j))
        if [ "$cell_idx" -lt "${#CELLS[@]}" ]; then
            run_one_cell "$j" "${CELLS[$cell_idx]}" &
            pids+=($!)
        fi
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
    rg_log "=== Batch ${batch_idx}/${TOTAL_BATCHES} done ==="
done

PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
GPU_HRS=$(awk "BEGIN{printf \"%.2f\", ${NUM_GPUS} * ${PANEL_DUR} / 3600}")
echo
rg_log "=== H4 PANEL DONE ==="
rg_log "Wall time:        ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr)"
rg_log "GPU-hr billable:  ${GPU_HRS} (${NUM_GPUS} GPUs × wall hr)"
echo
rg_log "Per-cell results: ${OUTPUTS_ROOT}/results/h4/"
ls -la "${OUTPUTS_ROOT}/results/h4/" 2>/dev/null || echo "(no results dir)"
