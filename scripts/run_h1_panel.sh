#!/bin/bash
# H1 panel — existence of the rational gap (R_hat_K) across ALL scopes in
# ONE script: development (gsm8k, math, humaneval), deployment (matharena,
# livecodebench), and preference (ultrafeedback, alpaca_eval).
#
# This merges the former development_exp / h5_exp / preference_exp H1 panels.
# Per-dataset sampling params (K, num_prompts, max_tokens, gpu_mem) live in
# the lookup tables below; preference datasets additionally route through an
# LLM-as-judge.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/run_h1_panel.sh [--num-gpus N] [model_alias ...]
#
# Common overrides (env):
#   SEEDS="0 1 2"                 cross-seed run (default "0")
#   DATASETS="gsm8k math"         restrict the dataset set
#   MODELS / positional args      restrict models
#   # Preference judge (default strict-self, L=5):
#   JUDGE=api JUDGE_MODEL=deepseek-v4-flash JUDGE_L=3   # hosted API judge
#   JUDGE_LOCAL_MODEL=qwen2.5-14b-instruct JUDGE_L=3    # fixed local judge
#   # Large model via tensor parallel (e.g. 72B on a 4-GPU NVLink node):
#   RG_TP=4 NUM_GPUS=1 MODELS=qwen2.5-72b-instruct      # cells run on all GPUs
#
# Run inside tmux so SSH drops don't kill it:
#   tmux new-session -d -s h1 "bash scripts/run_h1_panel.sh > outputs/logs/h1.log 2>&1"
# -----------------------------------------------------------------------------
set -uo pipefail   # no -e: a flaky cell logs and we continue

# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_parse_num_gpus "$@"
set -- "${RG_POSITIONAL[@]}"
rg_activate_env
rg_setup_hf

OUTPUTS_ROOT="$(rg_outputs_root)"
LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h1"

# --- per-dataset config (all scopes) -----------------------------------------
# Deterministic verifiers use K=64; preference (LLM-judge) uses K=32 with the
# generation capped at max_tokens=512 (matches the cached preference samples).
declare -A DS_K=(       [gsm8k]=64 [math]=64 [humaneval]=64 [matharena]=64 [livecodebench]=64 [ultrafeedback]=32 [alpaca_eval]=32 )
declare -A DS_M=(       [gsm8k]=1319 [math]=1000 [humaneval]=164 [matharena]=60 [livecodebench]=75 [ultrafeedback]=1000 [alpaca_eval]=300 )
declare -A DS_MT=(      [gsm8k]=1024 [math]=1024 [humaneval]=1024 [matharena]=1024 [livecodebench]=1024 [ultrafeedback]=512 [alpaca_eval]=512 )
declare -A DS_GPUMEM=(  [gsm8k]=0.85 [math]=0.85 [humaneval]=0.85 [matharena]=0.7 [livecodebench]=0.7 [ultrafeedback]=0.7 [alpaca_eval]=0.7 )
declare -A DS_PREF=(    [ultrafeedback]=1 [alpaca_eval]=1 )   # preference => LLM judge

# --- selection ---------------------------------------------------------------
if [ "$#" -gt 0 ]; then MODELS=("$@"); else
  MODELS=(tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct); fi
read -ra DATASETS <<< "${DATASETS:-gsm8k math humaneval matharena livecodebench ultrafeedback alpaca_eval}"
read -ra SEEDS <<< "${SEEDS:-0}"
RG_TP="${RG_TP:-1}"; export RG_TP
# Preference judge selection (applied only to preference datasets):
JUDGE="${JUDGE:-self}"; JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_LOCAL_MODEL="${JUDGE_LOCAL_MODEL:-}"; JUDGE_L="${JUDGE_L:-}"

# DATASET-OUTER ordering packs similar-walltime cells into the same batch.
CELLS=()
for d in "${DATASETS[@]}"; do for m in "${MODELS[@]}"; do for s in "${SEEDS[@]}"; do
  CELLS+=("${m}|${d}|${s}"); done; done; done

rg_log "=== H1 panel (all scopes) ==="
rg_log "Models: ${MODELS[*]}"
rg_log "Datasets: ${DATASETS[*]}"
rg_log "Seeds: ${SEEDS[*]} | GPUs: ${NUM_GPUS} | RG_TP: ${RG_TP} | Cells: ${#CELLS[@]}"
[ "$JUDGE" = api ] && rg_log "Preference judge: API ${JUDGE_MODEL} (L=${JUDGE_L:-3})"
[ -n "$JUDGE_LOCAL_MODEL" ] && rg_log "Preference judge: fixed local ${JUDGE_LOCAL_MODEL} (L=${JUDGE_L:-5})"

run_one_cell() {
  local gpu_idx="$1" cell_spec="$2"
  IFS='|' read -r model dataset seed <<< "$cell_spec"
  local np="${DS_M[$dataset]}" k="${DS_K[$dataset]}" mt="${DS_MT[$dataset]}" gm="${DS_GPUMEM[$dataset]}"
  local logfile="${LOG_DIR}/h1_gpu${gpu_idx}_${model}_${dataset}_seed${seed}.log"
  # Preference judge flags (only for preference datasets).
  local judge_args=()
  if [ -n "${DS_PREF[$dataset]:-}" ]; then
    if [ "$JUDGE" = api ]; then
      judge_args=(--judge api --judge-model "$JUDGE_MODEL" --judge-L "${JUDGE_L:-3}")
    elif [ -n "$JUDGE_LOCAL_MODEL" ]; then
      judge_args=(--judge self --judge-local-model "$JUDGE_LOCAL_MODEL" --judge-L "${JUDGE_L:-5}")
    fi
  fi
  # TP models (RG_TP>1) need all GPUs visible; otherwise pin one GPU per cell.
  local cvd_prefix=""
  [ "$RG_TP" -eq 1 ] && cvd_prefix="CUDA_VISIBLE_DEVICES=$gpu_idx"

  echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${model} × ${dataset} × seed=${seed} (K=$k N=$np)"
  local t0=$(date +%s)
  env $cvd_prefix python -m scripts.run_h1 \
    --model "$model" --dataset "$dataset" --seed "$seed" \
    --K "$k" --num-prompts "$np" --max-tokens "$mt" --gpu-mem "$gm" \
    "${judge_args[@]}" > "$logfile" 2>&1
  local rc=$? dur=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ]; then echo "[GPU $gpu_idx] $(date +%H:%M:%S) DONE  ${model} × ${dataset} (${dur}s)"
  else echo "[GPU $gpu_idx] $(date +%H:%M:%S) FAIL  ${model} × ${dataset} (rc=$rc, log: $logfile)"; fi
}

# TP/large-model runs are sequential (one cell uses all GPUs). vLLM does not
# always release VRAM cleanly between cells, so a leaked allocation from one
# cell can OOM the next at model load (a real cascade we hit on 72B). Before
# each TP cell, kill any lingering workers FOR THIS MODEL (never other jobs)
# and wait for VRAM to drain. No-op in the parallel 8B path (cells share GPUs).
wait_gpu_free() {  # $1 = model alias whose stragglers to clear
  pkill -9 -f "run_h1 --model $1" 2>/dev/null; sleep 8
  for _ in $(seq 1 24); do
    local mx; mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${mx:-9999}" -lt 2000 ] && { rg_log "GPU free (${mx}MiB) before next TP cell"; return 0; }
    sleep 10
  done
  rg_log "WARN: GPU not fully free (${mx}MiB) before next TP cell"
}

# Batch loop: NUM_GPUS cells in parallel (pinned); or sequential for TP models.
PANEL_T0=$(date +%s)
STRIDE="$NUM_GPUS"; [ "$RG_TP" -gt 1 ] && STRIDE=1
for ((i=0; i<${#CELLS[@]}; i+=STRIDE)); do
  if [ "$RG_TP" -gt 1 ]; then IFS='|' read -r _gm _ _ <<< "${CELLS[$i]}"; wait_gpu_free "$_gm"; fi
  pids=()
  for ((j=0; j<STRIDE; j++)); do
    ci=$((i + j))
    [ "$ci" -lt "${#CELLS[@]}" ] && { run_one_cell "$j" "${CELLS[$ci]}" & pids+=($!); }
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done
PANEL_DUR=$(( $(date +%s) - PANEL_T0 ))
rg_log "=== H1 PANEL DONE in ${PANEL_DUR}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DUR}/3600}") hr) ==="
ls -la "${OUTPUTS_ROOT}/results/h1/" 2>/dev/null | sed 's/^/  /' || true
