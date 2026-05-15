#!/bin/bash
# End-to-end orchestrator: H1 → H2 → H4 → plotting.
#
# Single command to reproduce the full paper from scratch on any node with
# enough GPUs and disk. Each panel runs to completion before the next
# starts; cells within a panel run in parallel on NUM_GPUS GPUs.
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/run_all.sh [--num-gpus N]
#
# Examples:
#   bash scripts/run_all.sh                # auto-detect GPUs
#   bash scripts/run_all.sh --num-gpus 3   # explicit
#   NUM_GPUS=2 bash scripts/run_all.sh     # via env var
#
# Run inside tmux so SSH disconnects don't kill it:
#   tmux new-session -d -s rg \
#       "bash scripts/run_all.sh --num-gpus 3 \
#        > outputs/logs/run_all.log 2>&1"
# -----------------------------------------------------------------------------
#
# Idempotent: if cell results already exist, the cells themselves cache-hit
# and re-run only the verifier + aggregates (seconds, no GPU work).
#
# Wall-time estimate (3 GPUs, full panels):
#   H1: ~3.75 hr (one model)  /  ~11 hr (3-model panel)
#   H2: ~3 hr (SFT + DPO sequential, RLVR via cache)
#   H4: ~5 hr (14 cells, 7 length values)
#   Plotting: <1 min
#
# Output layout (under $OUTPUTS_ROOT, default ./outputs):
#   results/h1/<model>_<dataset>_seed<S>.json
#   results/h2/<model>_<dataset>_seed<S>.json
#   results/h4/<model>_<dataset>_L<L>_seed<S>.json
#   results/figures/{h1_*.pdf, h2_*.pdf, h4_*.pdf}
#   sample_caches/<fingerprint>.jsonl.gz
#   verifier_audit/<cell>.jsonl
#   logs/run_all.log, logs/<panel>_*.log

set -uo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/../_common.sh"
rg_parse_num_gpus "$@"
set -- "${RG_POSITIONAL[@]}"
rg_activate_env
rg_setup_hf

OUTPUTS_ROOT="$(rg_outputs_root)"
LOG_DIR="$(rg_log_dir)"
mkdir -p "$LOG_DIR"

rg_log "=== run_all.sh start ==="
rg_log "  PID:           $$"
rg_log "  NUM_GPUS:      ${NUM_GPUS}"
rg_log "  OUTPUTS_ROOT:  ${OUTPUTS_ROOT}"
rg_log "  HF_HOME:       ${HF_HOME}"
[ -n "${HF_ENDPOINT:-}" ] && rg_log "  HF_ENDPOINT:   ${HF_ENDPOINT}"

run_panel() {
    local name="$1"      # h1 / h2 / h4
    local script="$2"    # path to panel script
    local logfile="${LOG_DIR}/run_all_${name}.log"
    rg_log "launching ${name} panel..."
    local t0=$(date +%s)
    bash "${script}" --num-gpus "${NUM_GPUS}" > "${logfile}" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    local count
    count=$(ls "${OUTPUTS_ROOT}/results/${name}/"*.json 2>/dev/null | wc -l)
    rg_log "${name} panel finished. exit=${rc}, wall=${dur}s, results=${count}"
}

# -----------------------------------------------------------------------------
# Panels.
# -----------------------------------------------------------------------------
run_panel h1 "$(dirname "$0")/run_h1_panel.sh"
run_panel h2 "$(dirname "$0")/run_h2_panel.sh"
run_panel h4 "$(dirname "$0")/run_h4_panel.sh"
run_panel h3 "$(dirname "$0")/run_h3_panel.sh"   # depends on H1 cache; runs last

# -----------------------------------------------------------------------------
# Plotting (CPU only, < 1 min).
# -----------------------------------------------------------------------------
rg_log "rendering figures..."
for hyp in h1 h2 h3 h4; do
    if python -m "src.plotting.plot_${hyp}" > "${LOG_DIR}/plot_${hyp}.log" 2>&1; then
        rg_log "plot_${hyp} OK"
    else
        rg_log "plot_${hyp} FAILED (see ${LOG_DIR}/plot_${hyp}.log)"
    fi
done

# -----------------------------------------------------------------------------
# Final summary.
# -----------------------------------------------------------------------------
echo
rg_log "=========================================================="
rg_log "RUN_ALL DONE"
rg_log "=========================================================="
for hyp in h1 h2 h4; do
    n=$(ls "${OUTPUTS_ROOT}/results/${hyp}/"*.json 2>/dev/null | wc -l)
    rg_log "  ${hyp^^}: ${n} cells"
done
rg_log "Figures:"
ls -la "${OUTPUTS_ROOT}/results/figures/" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
rg_log "Disk:"
df -h "${OUTPUTS_ROOT}" "${HF_HOME}" 2>/dev/null | sed 's/^/  /' || true
