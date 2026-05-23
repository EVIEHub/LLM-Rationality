#!/bin/bash
# Deployment H4 panel — Tülu-3-RLVR × matharena × 7 length values
# × seed=0 = 7 cells. Same budget-forced two-stage procedure as
# development_exp/run_h4_panel.sh.
#
# LiveCodeBench is intentionally excluded: the "Final answer:" forcing
# pattern only makes methodological sense on math-style benchmarks
# (numeric / boxed answer at the end of reasoning); for code, the model
# would be cut off mid-function. Same rationale used to exclude
# HumanEval from development H4.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/h5_exp/run_h4_panel.sh [--num-gpus N]
# -----------------------------------------------------------------------------
#
# Wall-time on 8B (2 GPUs):
#   ~10-15 min per cell on the small deployment splits (60 / 75 prompts)
#   14 cells / 2 GPUs = 7 batches ≈ ~70-105 min wall.

set -uo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/../_common.sh"
rg_parse_num_gpus "$@"
set -- "${RG_POSITIONAL[@]}"
rg_activate_env
rg_setup_hf

OUTPUTS_ROOT="$(rg_outputs_root)"
LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h4"

MODEL="${MODEL:-tulu3-8b-rlvr}"
read -ra DATASETS <<< "${DATASETS:-matharena}"
read -ra LS <<< "${LS:-0 64 128 256 512 1024 2048}"
SEED="${SEED:-0}"
K="${K:-64}"
GPU_MEM="${GPU_MEM:-0.7}"

declare -A NUM_PROMPTS=(
    [matharena]=60
)

CELLS=()
for L in "${LS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        CELLS+=("${ds}|${L}")
    done
done

TOTAL_BATCHES=$(( (${#CELLS[@]} + NUM_GPUS - 1) / NUM_GPUS ))
rg_log "=== H5 (deployment) — H4-analysis panel ==="
rg_log "Model:    ${MODEL}"
rg_log "Datasets: ${DATASETS[*]}"
rg_log "L values: ${LS[*]}"
rg_log "Cells:    ${#CELLS[@]}"
rg_log "GPUs:     ${NUM_GPUS}"
rg_log "Batches:  ${TOTAL_BATCHES}"

run_one_cell() {
    local gpu_idx="$1"
    local cell_spec="$2"
    IFS='|' read -r ds L <<< "$cell_spec"
    local np="${NUM_PROMPTS[$ds]}"
    local logfile="${LOG_DIR}/dep_h4_gpu${gpu_idx}_${ds}_L${L}.log"
    echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${ds} L=${L} N=${np}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu_idx" python -m scripts.run_h4 \
        --model "${MODEL}" --dataset "${ds}" --L "${L}" \
        --seed "${SEED}" --K "${K}" \
        --num-prompts "${np}" --gpu-mem "${GPU_MEM}" \
        > "$logfile" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) DONE  ${ds} L=${L} (${dur}s)"
    else
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) FAIL  ${ds} L=${L} (rc=$rc, log: $logfile)"
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
        idx=$((i + j))
        if [ "$idx" -lt "${#CELLS[@]}" ]; then
            run_one_cell "$j" "${CELLS[$idx]}" &
            pids+=($!)
        fi
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
done

PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
echo
rg_log "=== H5 (deployment) H4 PANEL DONE in ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr) ==="
ls "${OUTPUTS_ROOT}/results/h4/" 2>/dev/null | grep -E "matharena|livecodebench" || true
