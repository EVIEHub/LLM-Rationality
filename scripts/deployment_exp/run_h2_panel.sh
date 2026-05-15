#!/bin/bash
# Deployment H2 panel — Tülu-3 alignment trajectory on deployment data.
#
# Cells: 3 stages (SFT, DPO, RLVR) × 2 datasets (matharena, livecodebench)
#        × seed=0 = 6 cells.
#
# Strategy: sequential per stage with disk rotation. RLVR cells reuse
# the deployment H1 sample cache (no GPU work). SFT and DPO require
# fresh sampling — download model, run, delete.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/deployment_exp/run_h2_panel.sh [--num-gpus N]
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
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h2"

declare -A NUM_PROMPTS=(
    [matharena]=60
    [livecodebench]=75
)
SEED="${SEED:-0}"
K="${K:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM="${GPU_MEM:-0.7}"   # deployment datasets are long-prompt; H4 default

DATASETS=(matharena livecodebench)
N_DATASETS=${#DATASETS[@]}
GPUS_PER_STAGE=$(( NUM_GPUS < N_DATASETS ? NUM_GPUS : N_DATASETS ))

PANEL_T0=$(date +%s)
rg_log "=== deployment_exp H2 panel ==="
rg_log "Datasets:       ${DATASETS[*]}"
rg_log "Seed:           ${SEED}"
rg_log "GPUs available: ${NUM_GPUS}"
rg_log "GPUs per stage: ${GPUS_PER_STAGE}"

run_stage() {
    local model="$1"
    local label="$2"
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
            local logfile="${LOG_DIR}/dep_h2_${model}_${ds}.log"
            echo "  [GPU ${gpu_idx}] $(date +%H:%M:%S) ${model} × ${ds}  N=${np}"
            CUDA_VISIBLE_DEVICES="${gpu_idx}" python -m scripts.run_h2 \
                --model "${model}" --dataset "${ds}" --seed "${SEED}" --K "${K}" \
                --num-prompts "${np}" --max-tokens "${MAX_TOKENS}" --gpu-mem "${GPU_MEM}" \
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

remove_model_cache() {
    local hf_id="$1"
    local p="${HF_HOME}/hub/models--$(echo "$hf_id" | sed 's|/|--|g')"
    if [ -d "$p" ]; then
        rg_log "  removing $p"
        rm -rf "$p"
    fi
}

# ===========================================================================
# Stage 0: RLVR via cache hit (no GPU work; reuses deployment H1 samples)
# ===========================================================================
echo
rg_log "=== Stage 0: RLVR via deployment H1 cache hits ==="
for ds in "${DATASETS[@]}"; do
    np="${NUM_PROMPTS[$ds]}"
    python -m scripts.run_h2 \
        --model tulu3-8b-rlvr --dataset "${ds}" --seed "${SEED}" --K "${K}" \
        --num-prompts "${np}" --max-tokens "${MAX_TOKENS}" \
        > "${LOG_DIR}/dep_h2_tulu3-8b-rlvr_${ds}.log" 2>&1 \
        || echo "  RLVR ${ds} cell failed — check log (likely no H1 cache)"
done
rg_log "RLVR stage done"

# ===========================================================================
# Stage 1: SFT
# ===========================================================================
echo
rg_log "=== downloading SFT ==="
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B-SFT 2>&1 | tail -3 || \
    echo "(download warning — may already be cached)"
run_stage "tulu3-8b-sft" "Stage 1: SFT"

echo
rg_log "=== freeing SFT cache before DPO download ==="
remove_model_cache "allenai/Llama-3.1-Tulu-3-8B-SFT"

# ===========================================================================
# Stage 2: DPO
# ===========================================================================
echo
rg_log "=== downloading DPO ==="
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B-DPO 2>&1 | tail -3 || \
    echo "(download warning — may already be cached)"
run_stage "tulu3-8b-dpo" "Stage 2: DPO"

remove_model_cache "allenai/Llama-3.1-Tulu-3-8B-DPO"

PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
echo
rg_log "=== deployment_exp H2 PANEL DONE in ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr) ==="
ls "${OUTPUTS_ROOT}/results/h2/" 2>/dev/null | grep -E "matharena|livecodebench"
