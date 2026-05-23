#!/bin/bash
# End-to-end orchestrator — runs every hypothesis panel then renders all
# figures. One command to reproduce the paper on a node with enough GPUs/disk.
#
# Calls the per-hypothesis panels (each spans all scopes):
#   run_h1_panel.sh  run_h2_panel.sh  run_h3_panel.sh  run_h4_panel.sh
# H5 (hosted-API subjects) is opt-in (needs API creds): set RUN_H5=1.
#
# -----------------------------------------------------------------------------
# Usage:   bash scripts/run_all.sh [--num-gpus N]
#   NUM_GPUS=2 bash scripts/run_all.sh
#   RUN_H5=1 bash scripts/run_all.sh        # also run the API-subject panel
#
# Run inside tmux so SSH drops don't kill it:
#   tmux new-session -d -s rg "bash scripts/run_all.sh > outputs/logs/run_all.log 2>&1"
#
# Idempotent: completed cells cache-hit and re-run only verify/aggregate.
# -----------------------------------------------------------------------------
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
rg_parse_num_gpus "$@"; set -- "${RG_POSITIONAL[@]}"
rg_activate_env; rg_setup_hf
OUTPUTS_ROOT="$(rg_outputs_root)"; LOG_DIR="$(rg_log_dir)"; mkdir -p "$LOG_DIR"
HERE="$(dirname "$0")"

rg_log "=== run_all.sh start (PID $$, NUM_GPUS=${NUM_GPUS}) ==="
run_panel() {
  local name="$1" script="$2"
  rg_log "launching ${name} panel..."; local t0=$(date +%s)
  bash "${script}" --num-gpus "${NUM_GPUS}" > "${LOG_DIR}/run_all_${name}.log" 2>&1
  rg_log "${name} done (exit=$?, wall=$(( $(date +%s)-t0 ))s, results=$(ls "${OUTPUTS_ROOT}/results/${name}/"*.json 2>/dev/null | wc -l))"
}
run_panel h1 "${HERE}/run_h1_panel.sh"
run_panel h2 "${HERE}/run_h2_panel.sh"
run_panel h4 "${HERE}/run_h4_panel.sh"
run_panel h3 "${HERE}/run_h3_panel.sh"   # depends on H1 cache; last
if [ "${RUN_H5:-0}" = 1 ]; then
  rg_log "launching H5 (API subjects)..."
  bash "${HERE}/run_h5_panel.sh" > "${LOG_DIR}/run_all_h5.log" 2>&1 || rg_log "H5 panel exit=$?"
fi

rg_log "rendering figures..."
for hyp in h1 h2 h3 h4; do
  python -m "src.plotting.plot_${hyp}" > "${LOG_DIR}/plot_${hyp}.log" 2>&1 \
    && rg_log "plot_${hyp} OK" || rg_log "plot_${hyp} FAILED (see ${LOG_DIR}/plot_${hyp}.log)"
done
rg_log "=== RUN_ALL DONE ==="
for hyp in h1 h2 h3 h4; do
  rg_log "  ${hyp^^}: $(ls "${OUTPUTS_ROOT}/results/${hyp}/"*.json 2>/dev/null | wc -l) cells"
done
