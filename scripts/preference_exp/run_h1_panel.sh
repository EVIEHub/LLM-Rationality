#!/bin/bash
# Preference-experiment H1 panel — rational gap on open-ended preference
# tasks, with the model under test ALSO serving as its own judge
# (strict-self).
#
# Cells: 3 models × ultrafeedback × seed=0 = 3 cells.
# Each cell:
#   1. Generate K=32 candidate responses per prompt with the generator
#   2. Reload the same model as a judge, make L=5 i.i.d. judge calls per
#      (prompt, candidate) pair comparing it against the human-preferred
#      reference (`chosen` field of ultrafeedback_binarized test_prefs).
#   3. Strict-majority aggregate → ternary utility o ∈ {0, 0.5, 1}.
#   4. Compute (U_circ_K, U_bar_K, R_hat_K) via compute_rational_gap.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/preference_exp/run_h1_panel.sh [--num-gpus N] [model_alias ...]
# -----------------------------------------------------------------------------
#
# Wall-time estimate (M=300, K=32, L=5, on 8B-class models, 2 GPUs):
#   ~50 min/model (generation ~15 min + judge ~30 min + vLLM init twice)
#   3 cells / 2 GPUs => ~2 batches => ~100 min total

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

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct)
fi
DATASETS=(ultrafeedback)
read -ra SEEDS <<< "${SEEDS:-0}"
K="${K:-32}"
MAX_TOKENS="${MAX_TOKENS:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
GPU_MEM="${GPU_MEM:-0.7}"

CELLS=()
for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
        for s in "${SEEDS[@]}"; do
            CELLS+=("${m}|${d}|${s}")
        done
    done
done

TOTAL_BATCHES=$(( (${#CELLS[@]} + NUM_GPUS - 1) / NUM_GPUS ))
rg_log "=== preference_exp H1 panel ==="
rg_log "Models:      ${MODELS[*]}"
rg_log "Datasets:    ${DATASETS[*]}"
rg_log "Seeds:       ${SEEDS[*]}"
rg_log "Cells:       ${#CELLS[@]}"
rg_log "GPUs:        ${NUM_GPUS}"
rg_log "Batches:     ${TOTAL_BATCHES}"
rg_log "K=${K}, M=${NUM_PROMPTS}, max_tokens=${MAX_TOKENS}, gpu_mem=${GPU_MEM}"
rg_log "Judge: strict-self (each model judges its own samples), L=5"

run_one_cell() {
    local gpu_idx="$1"
    local cell_spec="$2"
    IFS='|' read -r model dataset seed <<< "$cell_spec"
    local logfile="${LOG_DIR}/pref_h1_gpu${gpu_idx}_${model}_${dataset}_seed${seed}.log"

    echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${model} × ${dataset} × seed=${seed} N=${NUM_PROMPTS}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu_idx" python -m scripts.run_h1 \
        --model "${model}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --K "${K}" \
        --num-prompts "${NUM_PROMPTS}" \
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
echo
rg_log "=== preference_exp H1 PANEL DONE in ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr) ==="
ls "${OUTPUTS_ROOT}/results/h1/" 2>/dev/null | grep ultrafeedback || true
