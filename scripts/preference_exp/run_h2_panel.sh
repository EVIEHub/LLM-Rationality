#!/bin/bash
# Preference-experiment H2 panel — Tülu-3 alignment trajectory on
# ultrafeedback, judged by a FIXED Tülu-3-RLVR model across all three
# stages.
#
# Cells: 3 stages (SFT, DPO, RLVR) × ultrafeedback × seed=0 = 3 cells.
#
# Per-cell flow (handled inside scripts.run_h2 when ds_cfg.verifier == "self_judge"):
#   1. Load stage model (SFT, DPO, or RLVR), generate K=32 candidates
#   2. Unload stage model, load Tülu-3-RLVR as judge
#   3. L=5 i.i.d. judge calls per (prompt, candidate) vs ultrafeedback's
#      `chosen` reference → ternary o ∈ {0, 0.5, 1}
#   4. compute_rational_gap → (U_circ_K, U_bar_K, R_hat_K)
#
# Why fixed RLVR judge (not strict-self per stage)?
#   SFT models are weak A/B parsers (verbose tangents, refusals, etc.)
#   and self-judge SFT samples produce unreliable verdicts. Using the
#   post-alignment Tülu-3-RLVR as a stable judge across the trajectory
#   gives us "the same judge's opinion of SFT vs DPO vs RLVR samples",
#   which is the scientific claim we can defend.
#
# Strategy: sequential per stage with disk rotation. The 50-60 GB
# autodl-tmp volume can't hold SFT + DPO + RLVR concurrently, so we
# rotate.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/preference_exp/run_h2_panel.sh [--num-gpus N]
# -----------------------------------------------------------------------------
#
# Wall-time estimate (M=300, K=32, L=5, 2 GPUs):
#   ~50 min/stage (download model + generation + load RLVR judge + judge)
#   3 stages sequential ≈ ~150 min total

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

DATASETS=(ultrafeedback)
SEED="${SEED:-0}"
K="${K:-32}"
MAX_TOKENS="${MAX_TOKENS:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
GPU_MEM="${GPU_MEM:-0.7}"

PANEL_T0=$(date +%s)
rg_log "=== preference_exp H2 panel ==="
rg_log "Datasets: ${DATASETS[*]}, Seed: ${SEED}"
rg_log "K=${K}, M=${NUM_PROMPTS}, max_tokens=${MAX_TOKENS}, gpu_mem=${GPU_MEM}"
rg_log "Judge: FIXED Tulu-3-RLVR for all stages (not strict-self), L=5"

run_stage() {
    local model="$1"
    local label="$2"
    echo
    rg_log "--- Stage: ${label} (${model}) ---"
    local stage_t0=$(date +%s)

    # H2 preference cells share GPU sequentially within a stage
    # (only 1 dataset). With 2 GPUs available, we leave the second
    # idle this is the simplest correct flow; a future refactor can
    # parallelise the judge phase on GPU 1.
    for ds in "${DATASETS[@]}"; do
        local logfile="${LOG_DIR}/pref_h2_${model}_${ds}.log"
        rg_log "  [GPU 0] $(date +%H:%M:%S) ${model} × ${ds}  N=${NUM_PROMPTS}"
        CUDA_VISIBLE_DEVICES=0 python -m scripts.run_h2 \
            --model "${model}" \
            --dataset "${ds}" \
            --seed "${SEED}" \
            --K "${K}" \
            --num-prompts "${NUM_PROMPTS}" \
            --max-tokens "${MAX_TOKENS}" \
            --gpu-mem "${GPU_MEM}" \
            > "${logfile}" 2>&1 \
            || echo "  (cell exited non-zero — check log)"
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
# Pre-flight: download Tulu-3-RLVR up front (used as fixed judge for all
# stages — keeping it on disk avoids re-downloading inside each stage's
# Python process).
# ===========================================================================
rg_log "=== pre-downloading FIXED judge Tulu-3-RLVR ==="
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B 2>&1 | tail -3 || \
    echo "(download warning — may already be cached)"

# ===========================================================================
# Stage 0: RLVR
# ===========================================================================
rg_log "=== downloading RLVR ==="
# (already downloaded as the judge; this is the generator+judge case)
run_stage "tulu3-8b-rlvr" "Stage 0: RLVR"

# ===========================================================================
# Stage 1: SFT
# ===========================================================================
echo
rg_log "=== downloading SFT (RLVR stays on disk as the fixed judge) ==="
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B-SFT 2>&1 | tail -3 || \
    echo "(download warning)"
run_stage "tulu3-8b-sft" "Stage 1: SFT"

echo
rg_log "=== freeing SFT cache (DPO comes next; keep RLVR as judge) ==="
remove_model_cache "allenai/Llama-3.1-Tulu-3-8B-SFT"

# ===========================================================================
# Stage 2: DPO
# ===========================================================================
echo
rg_log "=== downloading DPO ==="
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B-DPO 2>&1 | tail -3 || \
    echo "(download warning)"
run_stage "tulu3-8b-dpo" "Stage 2: DPO"

echo
rg_log "=== freeing DPO cache ==="
remove_model_cache "allenai/Llama-3.1-Tulu-3-8B-DPO"

PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
echo
rg_log "=== preference_exp H2 PANEL DONE in ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr) ==="
ls "${OUTPUTS_ROOT}/results/h2/" 2>/dev/null | grep ultrafeedback || true
