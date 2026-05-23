#!/bin/bash
# H4 panel — rational gap vs reasoning length L, via two-stage budget forcing.
# Single model (Tülu-3-RLVR) × {gsm8k, math, matharena, bbh} × 7 L values.
#
# Merges the former development_exp / h5_exp H4 panels. Code/preference
# datasets are excluded by design: budget forcing ("Final answer:") needs a
# short verifiable answer, which only the math-style and BBH (MC/boolean/
# count) tasks have.
#
# -----------------------------------------------------------------------------
# Usage:  bash scripts/run_h4_panel.sh [--num-gpus N]
# Overrides:  DATASETS="gsm8k bbh"   LS="0 256 1024"   NUM_PROMPTS overrides
# -----------------------------------------------------------------------------
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_parse_num_gpus "$@"; set -- "${RG_POSITIONAL[@]}"
rg_activate_env; rg_setup_hf
OUTPUTS_ROOT="$(rg_outputs_root)"; LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h4"

MODEL="${MODEL:-tulu3-8b-rlvr}"
SEED="${SEED:-0}"
K="${K:-64}"
GPU_MEM="${GPU_MEM:-0.70}"   # long-L cells OOM at higher util
read -ra DATASETS <<< "${DATASETS:-gsm8k math matharena bbh}"
read -ra LS <<< "${LS:-0 64 128 256 512 1024 2048}"
declare -A DS_M=( [gsm8k]=500 [math]=500 [matharena]=60 [bbh]=480 )

CELLS=()
for d in "${DATASETS[@]}"; do for L in "${LS[@]}"; do CELLS+=("${d}|${L}"); done; done
rg_log "=== H4 panel === model=$MODEL datasets=${DATASETS[*]} L=${LS[*]} cells=${#CELLS[@]} gpus=$NUM_GPUS"

run_one_cell() {
  local gpu_idx="$1" cell_spec="$2"; IFS='|' read -r dataset L <<< "$cell_spec"
  local np="${DS_M[$dataset]}" logfile="${LOG_DIR}/h4_gpu${gpu_idx}_${MODEL}_${dataset}_L${L}.log"
  echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${dataset} L=${L} (N=$np)"
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES="$gpu_idx" python -m scripts.run_h4 \
    --model "$MODEL" --dataset "$dataset" --L "$L" --seed "$SEED" \
    --K "$K" --num-prompts "$np" --gpu-mem "$GPU_MEM" > "$logfile" 2>&1
  local rc=$? dur=$(( $(date +%s) - t0 ))
  echo "[GPU $gpu_idx] $(date +%H:%M:%S) $([ $rc -eq 0 ] && echo DONE || echo FAIL) ${dataset} L=${L} (${dur}s$([ $rc -ne 0 ] && echo ", rc=$rc, $logfile"))"
}

PANEL_T0=$(date +%s)
for ((i=0; i<${#CELLS[@]}; i+=NUM_GPUS)); do
  pids=()
  for ((j=0; j<NUM_GPUS; j++)); do
    ci=$((i + j)); [ "$ci" -lt "${#CELLS[@]}" ] && { run_one_cell "$j" "${CELLS[$ci]}" & pids+=($!); }
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done
rg_log "=== H4 PANEL DONE in $(( $(date +%s) - PANEL_T0 ))s ==="
ls "${OUTPUTS_ROOT}/results/h4/" 2>/dev/null | sed 's/^/  /' || true
