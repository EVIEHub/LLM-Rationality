#!/bin/bash
# H3 panel — rational gap of each inference procedure, across ALL scopes.
# Merges the former development_exp / h5_exp H3 panels.
#
#   Phase 1 (direct): tau in {0.0, 0.7, 1.0} for every model × dataset.
#       tau=0.0 -> K=1 (greedy is deterministic)
#       tau=1.0 -> reuses the H1 sample cache (no GPU work)
#       tau=0.7 -> fresh sampling
#   Phase 2 (self-consistency): n in {2,4,8,16,32}, only on datasets with a
#       deterministic answer extractor (gsm8k, math, matharena).
#
# Datasets: development (gsm8k, math, humaneval), deployment (matharena,
# livecodebench), preference (ultrafeedback, alpaca_eval; direct only,
# strict-self judge). Iterates models with disk rotation friendliness.
#
# -----------------------------------------------------------------------------
# Usage:  bash scripts/run_h3_panel.sh [--num-gpus N] [model_alias ...]
# Overrides:  DATASETS, TAUS="0.0 0.7 1.0", SC_NS="2 4 8 16 32", SEEDS
# -----------------------------------------------------------------------------
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_parse_num_gpus "$@"; set -- "${RG_POSITIONAL[@]}"
rg_activate_env; rg_setup_hf
OUTPUTS_ROOT="$(rg_outputs_root)"; LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h3"

# K for tau>0 (greedy tau=0 always K=1); max_tokens; preference flag.
declare -A DS_K=(  [gsm8k]=64 [math]=64 [humaneval]=64 [matharena]=64 [livecodebench]=64 [ultrafeedback]=32 [alpaca_eval]=32 )
declare -A DS_MT=( [gsm8k]=1024 [math]=1024 [humaneval]=1024 [matharena]=1024 [livecodebench]=1024 [ultrafeedback]=512 [alpaca_eval]=512 )
declare -A DS_M=(  [gsm8k]=500 [math]=500 [humaneval]=164 [matharena]=60 [livecodebench]=75 [ultrafeedback]=500 [alpaca_eval]=300 )
declare -A IS_PREF=( [ultrafeedback]=1 [alpaca_eval]=1 )

if [ "$#" -gt 0 ]; then MODELS=("$@"); else
  MODELS=(tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct); fi
read -ra DATASETS <<< "${DATASETS:-gsm8k math humaneval matharena livecodebench ultrafeedback alpaca_eval}"
read -ra TAUS <<< "${TAUS:-0.0 0.7 1.0}"
read -ra SC_NS <<< "${SC_NS:-2 4 8 16 32}"
read -ra SC_DATASETS <<< "${SC_DATASETS:-gsm8k math matharena}"
SEED="${SEED:-0}"; GPU_MEM="${GPU_MEM:-0.7}"
# Preference judge override (default: strict-self):
JUDGE_LOCAL_MODEL="${JUDGE_LOCAL_MODEL:-}"; JUDGE_L="${JUDGE_L:-}"
declare -A IS_PREF=( [ultrafeedback]=1 [alpaca_eval]=1 )

# Build cells: "direct|model|ds|tau" and "sc|model|ds|n".
CELLS=()
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do for t in "${TAUS[@]}"; do CELLS+=("direct|${m}|${d}|${t}"); done; done
  for d in "${SC_DATASETS[@]}"; do for n in "${SC_NS[@]}"; do CELLS+=("sc|${m}|${d}|${n}"); done; done
done
rg_log "=== H3 panel === models=${MODELS[*]} datasets=${DATASETS[*]} taus=${TAUS[*]} sc_n=${SC_NS[*]} cells=${#CELLS[@]} gpus=$NUM_GPUS"

run_one_cell() {
  local gpu="$1" spec="$2"; IFS='|' read -r proc model ds arg <<< "$spec"
  local mt="${DS_MT[$ds]}" np="${DS_M[$ds]}" k proc_args lf
  if [ "$proc" = direct ]; then
    if awk "BEGIN{exit !($arg==0.0)}"; then k=1; else k="${DS_K[$ds]}"; fi
    proc_args=(--procedure direct --tau "$arg"); lf="${LOG_DIR}/h3_gpu${gpu}_${model}_${ds}_t${arg}.log"
  else
    k="${DS_K[$ds]}"; proc_args=(--procedure sc --sc-n "$arg"); lf="${LOG_DIR}/h3_gpu${gpu}_${model}_${ds}_sc${arg}.log"
  fi
  local judge_args=()
  [ -n "${IS_PREF[$ds]:-}" ] && [ -n "$JUDGE_LOCAL_MODEL" ] && \
    judge_args=(--judge-local-model "$JUDGE_LOCAL_MODEL" --judge-L "${JUDGE_L:-5}")
  echo "[GPU $gpu] $(date +%H:%M:%S) START ${model} × ${ds} ${proc}=${arg} (K=$k)"; local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES="$gpu" python -m scripts.run_h3 --model "$model" --dataset "$ds" \
    --seed "$SEED" --K "$k" --num-prompts "$np" --max-tokens "$mt" --gpu-mem "$GPU_MEM" \
    "${proc_args[@]}" "${judge_args[@]}" > "$lf" 2>&1
  local rc=$?; echo "[GPU $gpu] $(date +%H:%M:%S) $([ $rc -eq 0 ] && echo DONE || echo FAIL) ${model} × ${ds} ${proc}=${arg} ($(( $(date +%s)-t0 ))s$([ $rc -ne 0 ] && echo ", $lf"))"
}

PANEL_T0=$(date +%s)
for ((i=0; i<${#CELLS[@]}; i+=NUM_GPUS)); do
  pids=()
  for ((j=0; j<NUM_GPUS; j++)); do
    ci=$((i + j)); [ "$ci" -lt "${#CELLS[@]}" ] && { run_one_cell "$j" "${CELLS[$ci]}" & pids+=($!); }
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done
rg_log "=== H3 PANEL DONE in $(( $(date +%s) - PANEL_T0 ))s ==="
ls "${OUTPUTS_ROOT}/results/h3/" 2>/dev/null | wc -l | sed 's/^/  cells: /'
