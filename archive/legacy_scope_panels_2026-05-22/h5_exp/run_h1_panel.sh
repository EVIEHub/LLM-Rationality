#!/bin/bash
# Deployment-experiment H1 panel — does the rational gap persist on
# benchmarks the development-set models likely never trained on?
#
# Datasets (both contamination-resistant by construction):
#   - matharena:     60 competition-math problems from AIME 2025 + BRUMO 2025
#                    (post-cutoff for all evaluated models)
#   - livecodebench: 75 competitive-programming problems with
#                    contest_date >= 2024-01-01 (clearly post-cutoff
#                    for Llama-3.1; borderline for Tulu-3 and Qwen2.5)
#
# Cells: 3 models x 2 datasets x seed=0 = 6 cells. Cells run as
# independent vLLM processes pinned to a GPU via CUDA_VISIBLE_DEVICES;
# cells are scheduled in batches of NUM_GPUS, the orchestrator waits
# for each batch before starting the next.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/h5_exp/run_h1_panel.sh [--num-gpus N] [model_alias ...]
#
# Examples:
#   # Auto-detect GPUs, full deployment panel (3 models)
#   bash scripts/h5_exp/run_h1_panel.sh
#
#   # Force 2 GPUs
#   bash scripts/h5_exp/run_h1_panel.sh --num-gpus 2
#
#   # Single-model sub-panel
#   bash scripts/h5_exp/run_h1_panel.sh tulu3-8b-rlvr
#
# Run inside tmux:
#   tmux new-session -d -s dep_h1 \
#       "bash scripts/h5_exp/run_h1_panel.sh --num-gpus 2 \
#        > outputs/logs/dep_h1_panel.log 2>&1"
# -----------------------------------------------------------------------------
#
# Wall-time on 8B-class models (K=64, max_tokens=1024, 2 GPUs):
#   matharena     (60 prompts):  ~5 min/cell
#   livecodebench (75 prompts):  ~10 min/cell (per-test subprocess kernel
#                                              adds CPU overhead on verify)
#   6 cells / 2 GPUs = 3 batches => ~30-40 min wall total.

set -uo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/../_common.sh"
rg_parse_num_gpus "$@"
set -- "${RG_POSITIONAL[@]}"
rg_activate_env
rg_setup_hf

OUTPUTS_ROOT="$(rg_outputs_root)"
LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h1"

# -----------------------------------------------------------------------------
# Cell grid — deployment datasets.
# -----------------------------------------------------------------------------
declare -A NUM_PROMPTS=(
    [matharena]=60
    [livecodebench]=75
)

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct)
fi
DATASETS=(matharena livecodebench)
read -ra SEEDS <<< "${SEEDS:-0}"
K="${K:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM="${GPU_MEM:-0.85}"

# DATASET-OUTER ordering, same logic as development_exp/run_h1_panel.sh:
# each batch packs cells of similar wall time.
CELLS=()
for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        for s in "${SEEDS[@]}"; do
            CELLS+=("${m}|${d}|${s}")
        done
    done
done

TOTAL_BATCHES=$(( (${#CELLS[@]} + NUM_GPUS - 1) / NUM_GPUS ))
rg_log "=== H5 (deployment) — H1-analysis panel ==="
rg_log "Models:   ${MODELS[*]}"
rg_log "Datasets: ${DATASETS[*]}"
rg_log "Seeds:    ${SEEDS[*]}"
rg_log "Cells:    ${#CELLS[@]}"
rg_log "GPUs:     ${NUM_GPUS}"
rg_log "Batches:  ${TOTAL_BATCHES}"
rg_log "K=${K}, max_tokens=${MAX_TOKENS}, gpu_mem=${GPU_MEM}"

run_one_cell() {
    local gpu_idx="$1"
    local cell_spec="$2"
    IFS='|' read -r model dataset seed <<< "$cell_spec"
    local np="${NUM_PROMPTS[$dataset]}"
    local logfile="${LOG_DIR}/dep_h1_gpu${gpu_idx}_${model}_${dataset}_seed${seed}.log"

    # Both deployment-experiment datasets OOM at gpu-mem 0.85 in
    # smoke testing — long prompts (livecodebench: full problem
    # statements; matharena: olympiad-style preamble) cause vLLM's
    # scheduler to over-admit. 0.7 matches the H4 long-L cells.
    local cell_gpu_mem="0.7"

    echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${model} × ${dataset} × seed=${seed} N=${np}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu_idx" python -m scripts.run_h1 \
        --model "${model}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --K "${K}" \
        --num-prompts "${np}" \
        --max-tokens "${MAX_TOKENS}" \
        --gpu-mem "${cell_gpu_mem}" \
        > "$logfile" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) DONE  ${model} × ${dataset} × seed=${seed} (${dur}s)"
    else
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) FAIL  ${model} × ${dataset} × seed=${seed} (rc=$rc, log: $logfile)"
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
    rg_log "=== Batch ${batch_idx}/${TOTAL_BATCHES} done (${#pids[@]} cells) ==="
done

PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
GPU_HRS=$(awk "BEGIN{printf \"%.2f\", ${NUM_GPUS} * ${PANEL_DUR} / 3600}")
echo
rg_log "=== H5 (deployment) H1 PANEL DONE ==="
rg_log "Wall time:        ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr)"
rg_log "GPU-hr billable:  ${GPU_HRS} (${NUM_GPUS} GPUs × wall hr)"
echo
rg_log "Per-cell results: ${OUTPUTS_ROOT}/results/h1/  (filenames include the deployment dataset suffix)"
ls -la "${OUTPUTS_ROOT}/results/h1/" 2>/dev/null | grep -E "deepmind|bigcode" || true
