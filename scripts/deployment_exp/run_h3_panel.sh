#!/bin/bash
# Deployment H3 panel — rational gap of each inference procedure on
# the unseen-distribution datasets.
#
# Phases:
#   Phase 1 (GPU): direct sampling at tau in {0, 0.7, 1.0}
#       - tau=1.0 cells reuse deployment H1 sample cache (no GPU)
#       - tau=0.0 (greedy) cells use K=1
#       - tau=0.7 cells require fresh K=64 sampling
#     Cells: 3 models * 2 datasets * 3 tau = 18 cells.
#   Phase 2 (CPU): self-consistency at n in {2, 4, 8, 16, 32}
#     Cells: 3 models * 1 dataset (matharena) * 5 n = 15 cells.
#     LiveCodeBench skipped (no answer-key extractor for code).
#
# Disk rotation across the 3 H1 models.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/deployment_exp/run_h3_panel.sh [--num-gpus N]
# -----------------------------------------------------------------------------

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

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct)
fi
DATASETS=(matharena livecodebench)
TAUS=(1.0 0.0 0.7)
SC_DATASETS=(matharena)   # livecodebench skipped — no answer-key extractor
SC_NS=(2 4 8 16 32)
SEED="${SEED:-0}"
K="${K:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM="${GPU_MEM:-0.7}"

declare -A NUM_PROMPTS=(
    [matharena]=60
    [livecodebench]=75
)
declare -A HF_IDS=(
    [tulu3-8b-rlvr]="allenai/Llama-3.1-Tulu-3-8B"
    [qwen2.5-7b-instruct]="Qwen/Qwen2.5-7B-Instruct"
    [llama3.1-8b-instruct]="meta-llama/Llama-3.1-8B-Instruct"
)

rg_log "=== deployment_exp H3 panel ==="
rg_log "Models:    ${MODELS[*]}"
rg_log "Datasets:  ${DATASETS[*]}"
rg_log "Taus:      ${TAUS[*]}"
rg_log "SC n:      ${SC_NS[*]} (matharena only)"
rg_log "GPUs:      ${NUM_GPUS}"

remove_model_cache() {
    local hf_id="$1"
    local p="${HF_HOME}/hub/models--$(echo "$hf_id" | sed 's|/|--|g')"
    [ -d "$p" ] && { rg_log "  removing $p"; rm -rf "$p"; }
}

run_direct_cell() {
    local gpu="$1"
    local model="$2"
    local dataset="$3"
    local tau="$4"
    local np="${NUM_PROMPTS[$dataset]}"
    local logfile="${LOG_DIR}/dep_h3_${model}_${dataset}_t${tau}.log"
    echo "[GPU $gpu] $(date +%H:%M:%S) START direct ${model} x ${dataset} tau=${tau} N=${np}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu python -m scripts.run_h3 \
        --model "${model}" --dataset "${dataset}" --seed "${SEED}" --K "${K}" \
        --procedure direct --tau "${tau}" \
        --num-prompts "${np}" --max-tokens "${MAX_TOKENS}" --gpu-mem "${GPU_MEM}" \
        > "$logfile" 2>&1
    local rc=$?
    echo "[GPU $gpu] $(date +%H:%M:%S) DONE  direct ${model} x ${dataset} tau=${tau} rc=$rc ($(( $(date +%s) - t0 ))s)"
}

run_sc_cell() {
    local model="$1"
    local dataset="$2"
    local n="$3"
    local np="${NUM_PROMPTS[$dataset]}"
    local logfile="${LOG_DIR}/dep_h3_${model}_${dataset}_sc_n${n}.log"
    echo "[CPU] $(date +%H:%M:%S) START SC ${model} x ${dataset} n=${n}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES= python -m scripts.run_h3 \
        --model "${model}" --dataset "${dataset}" --seed "${SEED}" --K "${K}" \
        --procedure sc --sc-n "${n}" \
        --num-prompts "${np}" --max-tokens "${MAX_TOKENS}" \
        > "$logfile" 2>&1
    echo "[CPU] $(date +%H:%M:%S) DONE  SC ${model} x ${dataset} n=${n} rc=$? ($(( $(date +%s) - t0 ))s)"
}

PANEL_T0=$(date +%s)

# ===========================================================================
# Phase 1: GPU work — direct sampling, model-outer with disk rotation.
# Within each model: 2 datasets x 3 taus = 6 cells in batches of NUM_GPUS.
# ===========================================================================
rg_log "============================================================"
rg_log "Phase 1: direct sampling"
rg_log "============================================================"

for model in "${MODELS[@]}"; do
    rg_log "--- direct phase for ${model} ---"
    rg_log "downloading ${HF_IDS[$model]}"
    huggingface-cli download "${HF_IDS[$model]}" 2>&1 | tail -3 || true
    df -h "$HF_HOME" 2>/dev/null | tail -1

    CELLS=()
    for tau in "${TAUS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            CELLS+=("${ds}|${tau}")
        done
    done

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
        for pid in "${pids[@]}"; do wait "$pid" || true; done
    done

    rg_log "freeing ${HF_IDS[$model]}"
    remove_model_cache "${HF_IDS[$model]}"
done

# ===========================================================================
# Phase 2: CPU work — self-consistency bootstrap (matharena only).
# ===========================================================================
rg_log "============================================================"
rg_log "Phase 2: self-consistency bootstrap (CPU)"
rg_log "============================================================"

for model in "${MODELS[@]}"; do
    for ds in "${SC_DATASETS[@]}"; do
        for n in "${SC_NS[@]}"; do
            run_sc_cell "$model" "$ds" "$n"
        done
    done
done

PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
echo
rg_log "=== deployment_exp H3 PANEL DONE in ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr) ==="
ls "${OUTPUTS_ROOT}/results/h3/" 2>/dev/null | grep -E "matharena|livecodebench" | wc -l | xargs echo "  deployment H3 cells:"
