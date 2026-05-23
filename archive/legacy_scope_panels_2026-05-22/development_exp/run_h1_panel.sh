#!/bin/bash
# H1 panel orchestrator — does the rational gap exist on standard math/code
# benchmarks at K=64?  Cross-model panel: 3 models × 3 datasets × seed=0.
#
# 9 cells total. Cells run as independent vLLM processes pinned to a GPU
# via CUDA_VISIBLE_DEVICES; cells are scheduled in batches of NUM_GPUS, the
# orchestrator waits for each batch before starting the next.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/run_h1_panel.sh [--num-gpus N] [model_alias ...]
#
# Examples:
#   # Auto-detect GPUs, full panel (3 models)
#   bash scripts/run_h1_panel.sh
#
#   # Force 2 GPUs, full panel
#   bash scripts/run_h1_panel.sh --num-gpus 2
#
#   # Single-model sub-panel on whatever GPUs are visible
#   bash scripts/run_h1_panel.sh tulu3-8b-rlvr
#
#   # Re-run all seeds {0, 1, 2} in cross-seed mode
#   SEEDS="0 1 2" bash scripts/run_h1_panel.sh --num-gpus 3
#
# Run inside tmux so SSH disconnects don't kill the panel:
#   tmux new-session -d -s h1_panel \
#       "bash scripts/run_h1_panel.sh --num-gpus 3 \
#        > outputs/logs/h1_panel.log 2>&1"
# -----------------------------------------------------------------------------
#
# Wall-time on 8B-class models (K=64, num_prompts as below):
#   gsm8k (1319 prompts, 1024 reasoning tokens):  ~70 min/cell
#   math  (1000 prompts, 1024 reasoning tokens):  ~50 min/cell
#   humaneval (164 prompts, 1024 reasoning tokens): ~10 min/cell
#
# Per-cell logs: $LOG_DIR/h1_<model>_<dataset>_seed<S>.log
# Per-cell results JSON: $OUTPUTS_ROOT/results/h1/<model>_<dataset>_seed<S>.json
# Verifier audit log: $OUTPUTS_ROOT/results/h1/<cell>_audit.jsonl
#   POSIX guarantees PIPE_BUF-sized writes are atomic, so concurrent appends
#   are safe across cells running on different GPUs.

set -uo pipefail
# NOTE: no `-e` — a single-cell failure logs and we continue, so a flaky
# cell does not abort the whole panel.

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
# Cell grid.
# -----------------------------------------------------------------------------
declare -A NUM_PROMPTS=(
    [gsm8k]=1319
    [math]=1000
    [humaneval]=164
)

# Models can be overridden by passing aliases as positional args.
# Default: full H1 cross-model panel.
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct)
fi
DATASETS=(gsm8k math humaneval)
# Single-seed run answers "does the pipeline work and do the numbers look
# plausible". Re-run with SEEDS="0 1 2" once the seed=0 panel is judged
# worth investing in cross-seed std.
read -ra SEEDS <<< "${SEEDS:-0}"
K="${K:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM="${GPU_MEM:-0.85}"

# DATASET-OUTER cell ordering: each batch of NUM_GPUS cells runs the SAME
# dataset across different models in parallel. This packs cells of similar
# wall time into the same batch (gsm8k ~70 min, math ~50 min, humaneval
# ~10 min) instead of mixing them, which would leave a humaneval GPU idle
# for ~60 min while a gsm8k GPU finishes. Within a batch the per-cell wall
# time is roughly constant across models (8B-class on the same dataset).
CELLS=()
for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        for s in "${SEEDS[@]}"; do
            CELLS+=("${m}|${d}|${s}")
        done
    done
done

TOTAL_BATCHES=$(( (${#CELLS[@]} + NUM_GPUS - 1) / NUM_GPUS ))
rg_log "=== H1 panel ==="
rg_log "Models:   ${MODELS[*]}"
rg_log "Datasets: ${DATASETS[*]}"
rg_log "Seeds:    ${SEEDS[*]}"
rg_log "Cells:    ${#CELLS[@]}"
rg_log "GPUs:     ${NUM_GPUS}"
rg_log "Batches:  ${TOTAL_BATCHES}"
rg_log "K=${K}, max_tokens=${MAX_TOKENS}, gpu_mem=${GPU_MEM}"

# -----------------------------------------------------------------------------
# Per-cell launcher (pinned to one GPU).
# -----------------------------------------------------------------------------
run_one_cell() {
    local gpu_idx="$1"
    local cell_spec="$2"
    IFS='|' read -r model dataset seed <<< "$cell_spec"
    local np="${NUM_PROMPTS[$dataset]}"
    local logfile="${LOG_DIR}/h1_gpu${gpu_idx}_${model}_${dataset}_seed${seed}.log"

    echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${model} × ${dataset} × seed=${seed} N=${np}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu_idx" python -m scripts.run_h1 \
        --model "${model}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --K "${K}" \
        --num-prompts "${np}" \
        --max-tokens "${MAX_TOKENS}" \
        --gpu-mem "${GPU_MEM}" \
        > "$logfile" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) DONE  ${model} × ${dataset} × seed=${seed} (${dur}s)"
    else
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) FAIL  ${model} × ${dataset} × seed=${seed} (rc=$rc, log: $logfile)"
    fi
}

# -----------------------------------------------------------------------------
# Batch loop: NUM_GPUS cells in parallel, wait, repeat.
# -----------------------------------------------------------------------------
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
rg_log "=== H1 PANEL DONE ==="
rg_log "Wall time:        ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr)"
rg_log "GPU-hr billable:  ${GPU_HRS} (${NUM_GPUS} GPUs × wall hr)"
echo
rg_log "Per-cell results: ${OUTPUTS_ROOT}/results/h1/"
ls -la "${OUTPUTS_ROOT}/results/h1/" 2>/dev/null || true
