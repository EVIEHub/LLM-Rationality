#!/bin/bash
# H5 panel — rational gap in FRONTIER PROPRIETARY LLMs on contamination-
# resistant deployment data. API subjects (gpt-5.2-chat, gpt-5.5,
# deepseek-v4-flash) via the backend:api path — NO GPU, pure API. Safe to
# run alongside GPU jobs on the same box.
#
# Mirrors what the API subjects ran: H1 (saturation/gap) AND H3 (inference
# procedures, direct tau sweep) on matharena + livecodebench, K=64.
#   - H3 tau=1.0 reuses the H1 sample cache (no extra API calls).
#   - H3 tau=0.0 is greedy (K=1); tau=0.7 needs fresh sampling.
# Deterministic verifiers (no LLM judge).
#
# -----------------------------------------------------------------------------
# Requires API creds; sources ~/.config/rg-gap.env (TP_* for OpenAI proxy,
# DS_* for DeepSeek). models.yaml maps each model to its *_BASE_URL/*_API_KEY.
#
# Usage:
#   MODELS="gpt-5.5" bash scripts/run_h5_panel.sh           # one subject
#   bash scripts/run_h5_panel.sh                            # all 3 subjects
# Overrides: DATASETS, HYPS="h1 h3", TAUS="0.0 0.7 1.0", CONC, K, MAXTOK
#
# NOTE: ChatGPT-proxy subjects (gpt-*) have a daily message quota; a full run
# spans days, resuming via ApiRunner's per-(prompt,k) cache. DeepSeek is metered.
# -----------------------------------------------------------------------------
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_activate_env; rg_setup_hf
[ -f /root/.config/rg-gap.env ] && { set -a; source /root/.config/rg-gap.env; set +a; }
OUTPUTS_ROOT="$(rg_outputs_root)"; LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR" "$OUTPUTS_ROOT/results/h1" "$OUTPUTS_ROOT/results/h3"

read -ra MODELS <<< "${MODELS:-gpt-5.2-chat gpt-5.5 deepseek-v4-flash}"
read -ra DATASETS <<< "${DATASETS:-matharena livecodebench}"
read -ra HYPS <<< "${HYPS:-h1 h3}"
read -ra TAUS <<< "${TAUS:-0.0 0.7 1.0}"
declare -A DS_M=( [matharena]=60 [livecodebench]=75 )
SEED="${SEED:-0}"; K="${K:-64}"; CONC="${CONC:-20}"; MAXTOK="${MAXTOK:-16000}"
rg_log "=== H5 panel (API) === models=${MODELS[*]} datasets=${DATASETS[*]} hyps=${HYPS[*]} K=$K conc=$CONC"

api_run() {  # logfile, then python -m args...
  local lf="$1"; shift
  rg_log "START $* "; local t0=$(date +%s)
  python -m "$@" --api-concurrency "$CONC" > "$lf" 2>&1
  local rc=$?; rg_log "$([ $rc -eq 0 ] && echo DONE || echo FAIL) ($(( $(date +%s)-t0 ))s$([ $rc -ne 0 ] && echo ", $lf"))"
}

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    np="${DS_M[$ds]}"
    for hyp in "${HYPS[@]}"; do
      if [ "$hyp" = h1 ]; then
        api_run "${LOG_DIR}/h5_${model}_${ds}_h1.log" \
          scripts.run_h1 --model "$model" --dataset "$ds" --seed "$SEED" \
          --K "$K" --num-prompts "$np" --max-tokens "$MAXTOK"
      elif [ "$hyp" = h3 ]; then
        for tau in "${TAUS[@]}"; do
          k="$K"; awk "BEGIN{exit !($tau==0.0)}" && k=1   # greedy => K=1
          api_run "${LOG_DIR}/h5_${model}_${ds}_h3_t${tau}.log" \
            scripts.run_h3 --model "$model" --dataset "$ds" --seed "$SEED" \
            --K "$k" --num-prompts "$np" --max-tokens "$MAXTOK" \
            --procedure direct --tau "$tau"
        done
      fi
    done
  done
done
rg_log "=== H5 PANEL DONE ==="
