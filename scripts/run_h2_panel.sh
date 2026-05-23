#!/bin/bash
# H2 panel — Tülu-3 alignment trajectory (SFT -> DPO -> RLVR) across ALL
# scopes: development (gsm8k, math, humaneval), deployment (matharena,
# livecodebench), preference (ultrafeedback, alpaca_eval).
#
# Merges the former development_exp / h5_exp / preference_exp H2 panels.
# Stages run SEQUENTIALLY with disk rotation (a tight autodl-tmp volume
# cannot hold all stage weights at once): download stage -> run all its
# datasets -> delete weights -> next stage. Preference cells use the fixed
# Tülu-3-RLVR judge inside scripts.run_h2 (held constant across stages).
#
# -----------------------------------------------------------------------------
# Usage:  bash scripts/run_h2_panel.sh [--num-gpus N]
# Overrides:  STAGES="tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr"
#             DATASETS="gsm8k ultrafeedback"   SEEDS="0"   ROTATE=0 (keep weights)
# -----------------------------------------------------------------------------
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_parse_num_gpus "$@"; set -- "${RG_POSITIONAL[@]}"
rg_activate_env; rg_setup_hf
OUTPUTS_ROOT="$(rg_outputs_root)"; LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h2"

# Per-dataset params (same grid as H1).
declare -A DS_K=(      [gsm8k]=64 [math]=64 [humaneval]=64 [matharena]=64 [livecodebench]=64 [ultrafeedback]=32 [alpaca_eval]=32 )
declare -A DS_M=(      [gsm8k]=1319 [math]=1000 [humaneval]=164 [matharena]=60 [livecodebench]=75 [ultrafeedback]=1000 [alpaca_eval]=300 )
declare -A DS_MT=(     [gsm8k]=1024 [math]=1024 [humaneval]=1024 [matharena]=1024 [livecodebench]=1024 [ultrafeedback]=512 [alpaca_eval]=512 )
declare -A DS_GPUMEM=( [gsm8k]=0.85 [math]=0.85 [humaneval]=0.85 [matharena]=0.7 [livecodebench]=0.7 [ultrafeedback]=0.7 [alpaca_eval]=0.7 )
# alias -> HF cache dir name (for rotation deletes).
declare -A HF_DIR=( [tulu3-8b-sft]=models--allenai--Llama-3.1-Tulu-3-8B-SFT
                    [tulu3-8b-dpo]=models--allenai--Llama-3.1-Tulu-3-8B-DPO
                    [tulu3-8b-rlvr]=models--allenai--Llama-3.1-Tulu-3-8B )

read -ra STAGES <<< "${STAGES:-tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr}"
read -ra DATASETS <<< "${DATASETS:-gsm8k math humaneval matharena livecodebench ultrafeedback alpaca_eval}"
SEED="${SEED:-0}"; ROTATE="${ROTATE:-1}"; RG_TP="${RG_TP:-1}"; export RG_TP
# Preference judge override (default: run_h2's fixed Tulu-3-RLVR judge):
JUDGE_LOCAL_MODEL="${JUDGE_LOCAL_MODEL:-}"; JUDGE_L="${JUDGE_L:-}"
declare -A IS_PREF=( [ultrafeedback]=1 [alpaca_eval]=1 )
rg_log "=== H2 panel === stages=${STAGES[*]} datasets=${DATASETS[*]} gpus=$NUM_GPUS rotate=$ROTATE"

run_cell() {  # gpu, model, dataset
  local gpu="$1" model="$2" ds="$3"
  local np="${DS_M[$ds]}" k="${DS_K[$ds]}" mt="${DS_MT[$ds]}" gm="${DS_GPUMEM[$ds]}"
  local lf="${LOG_DIR}/h2_gpu${gpu}_${model}_${ds}_seed${SEED}.log"
  local cvd=""; [ "$RG_TP" -eq 1 ] && cvd="CUDA_VISIBLE_DEVICES=$gpu"
  local judge_args=()
  [ -n "${IS_PREF[$ds]:-}" ] && [ -n "$JUDGE_LOCAL_MODEL" ] && \
    judge_args=(--judge self --judge-local-model "$JUDGE_LOCAL_MODEL" --judge-L "${JUDGE_L:-5}")
  echo "[GPU $gpu] $(date +%H:%M:%S) START ${model} × ${ds} (K=$k N=$np)"; local t0=$(date +%s)
  env $cvd python -m scripts.run_h2 --model "$model" --dataset "$ds" --seed "$SEED" \
    --K "$k" --num-prompts "$np" --max-tokens "$mt" --gpu-mem "$gm" \
    "${judge_args[@]}" > "$lf" 2>&1
  local rc=$?; echo "[GPU $gpu] $(date +%H:%M:%S) $([ $rc -eq 0 ] && echo DONE || echo FAIL) ${model} × ${ds} ($(( $(date +%s)-t0 ))s$([ $rc -ne 0 ] && echo ", $lf"))"
}

# TP/large-model cells are sequential and vLLM may leak VRAM between them;
# clear this stage-model's stragglers + wait for VRAM to drain before each
# TP cell (prevents the OOM cascade hit on 70B). No-op in the parallel path.
wait_gpu_free() {  # $1 = model alias whose stragglers to clear
  pkill -9 -f "run_h2 --model $1" 2>/dev/null; sleep 8
  for _ in $(seq 1 24); do
    local mx; mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${mx:-9999}" -lt 2000 ] && { rg_log "GPU free (${mx}MiB) before next TP cell"; return 0; }
    sleep 10
  done
  rg_log "WARN: GPU not fully free (${mx}MiB) before next TP cell"
}

PANEL_T0=$(date +%s)
for stage in "${STAGES[@]}"; do
  rg_log "--- stage: $stage ---"
  # Batch this stage's datasets across GPUs (TP models run sequentially).
  STRIDE="$NUM_GPUS"; [ "$RG_TP" -gt 1 ] && STRIDE=1
  for ((i=0; i<${#DATASETS[@]}; i+=STRIDE)); do
    [ "$RG_TP" -gt 1 ] && wait_gpu_free "$stage"
    pids=()
    for ((j=0; j<STRIDE; j++)); do
      ci=$((i + j)); [ "$ci" -lt "${#DATASETS[@]}" ] && { run_cell "$j" "$stage" "${DATASETS[$ci]}" & pids+=($!); }
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  done
  # Disk rotation: drop this stage's weights before the next downloads.
  if [ "$ROTATE" = 1 ] && [ -n "${HF_DIR[$stage]:-}" ]; then
    rg_log "rotating: removing ${HF_DIR[$stage]} from HF cache"
    rm -rf "${HF_HOME}/hub/${HF_DIR[$stage]}" 2>/dev/null || true
  fi
done
rg_log "=== H2 PANEL DONE in $(( $(date +%s) - PANEL_T0 ))s ==="
ls "${OUTPUTS_ROOT}/results/h2/" 2>/dev/null | sed 's/^/  /' || true
