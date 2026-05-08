# shellcheck shell=bash
# Shared init for the orchestrator scripts in this directory.
#
# Sourced (not executed) by scripts/run_*_panel.sh and scripts/run_all.sh.
# Provides:
#   - rg_parse_num_gpus "$@"   : sets NUM_GPUS from --num-gpus flag, NUM_GPUS
#                                env var, or auto-detection (nvidia-smi).
#   - rg_activate_env          : activates the conda env named in CONDA_ENV
#                                (default: rg-gap), if conda is on PATH.
#   - rg_setup_hf              : sets HF_HOME and (optionally) HF_ENDPOINT.
#   - rg_outputs_root          : echoes the outputs root (OUTPUTS_ROOT env var
#                                or ./outputs relative to repo root).
#   - rg_log_dir               : echoes the per-cell log dir.
#   - rg_log "msg"             : timestamped echo for orchestrator logs.
#
# All functions read from env vars so the scripts stay declarative.
# Override any of these before invoking a panel script:
#   NUM_GPUS=2   bash scripts/run_h1_panel.sh
#   CONDA_ENV=my-env  OUTPUTS_ROOT=/data/rg  bash scripts/run_all.sh
#   HF_ENDPOINT=https://hf-mirror.com  bash scripts/run_h2_panel.sh

# -----------------------------------------------------------------------------
# Parse --num-gpus N from CLI args; fall back to NUM_GPUS env var or detection.
#
# Removes --num-gpus and its value from the positional args, so the caller
# can still consume remaining args (e.g. model aliases for run_h1_panel).
# Usage in caller: `eval set -- "$(rg_parse_num_gpus "$@")"`  is NOT used;
# instead caller does: `rg_parse_num_gpus "$@"; shift $RG_SHIFT`.
# -----------------------------------------------------------------------------
rg_parse_num_gpus() {
    RG_SHIFT=0
    local positional=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --num-gpus)
                NUM_GPUS="$2"
                shift 2
                RG_SHIFT=$((RG_SHIFT + 2))
                ;;
            --num-gpus=*)
                NUM_GPUS="${1#--num-gpus=}"
                shift
                RG_SHIFT=$((RG_SHIFT + 1))
                ;;
            -h|--help)
                rg_print_help
                exit 0
                ;;
            *)
                positional+=("$1")
                shift
                ;;
        esac
    done
    # Restore positional args for caller: caller should call
    #   rg_parse_num_gpus "$@"
    #   set -- "${RG_POSITIONAL[@]}"
    RG_POSITIONAL=("${positional[@]}")

    if [ -z "${NUM_GPUS:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
    fi
    # Fall back to 1 if detection produced 0 or empty.
    if [ -z "${NUM_GPUS:-}" ] || [ "${NUM_GPUS}" = "0" ]; then
        NUM_GPUS=1
    fi
    if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: NUM_GPUS must be a positive integer, got '${NUM_GPUS}'" >&2
        exit 2
    fi
    export NUM_GPUS
}

rg_print_help() {
    cat <<EOF
Usage: bash $(basename "$0") [--num-gpus N] [extra args...]

Options:
  --num-gpus N       Number of GPUs to run cells on in parallel.
                     Default: auto-detected via 'nvidia-smi -L', else 1.
                     Can also be set via NUM_GPUS env var.
  -h, --help         Show this help.

Environment variables (all optional):
  NUM_GPUS           Same as --num-gpus.
  CONDA_ENV          Conda env to activate (default: rg-gap). Skipped if
                     'conda' is not on PATH (caller is assumed to have
                     activated the right environment already).
  OUTPUTS_ROOT       Root for results, sample caches, run logs.
                     Default: ./outputs (relative to repo root).
  LOG_DIR            Per-cell log dir.
                     Default: \$OUTPUTS_ROOT/logs.
  HF_HOME            Hugging Face cache root.
                     Default: \$OUTPUTS_ROOT/hf_cache.
  HF_ENDPOINT        HF mirror URL (e.g. https://hf-mirror.com for users
                     without direct huggingface.co access). Default: unset.

Per-cell wall-time and dollar cost are constant in total GPU-hours: more
GPUs reduce wall-clock at the same total cost, modulo per-process vLLM
warm-up overhead (~1 min/cell). Use --num-gpus = (#cells in the panel)
for the lowest wall time, or = 1 for the simplest serial run.
EOF
}

# -----------------------------------------------------------------------------
# Activate conda env if available. Skips if conda is not installed (caller
# is expected to have set up Python deps another way).
# -----------------------------------------------------------------------------
rg_activate_env() {
    local env_name="${CONDA_ENV:-rg-gap}"
    # Try common conda locations.
    local conda_sh=""
    for candidate in \
        /root/miniconda3/etc/profile.d/conda.sh \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        /opt/miniconda3/etc/profile.d/conda.sh \
        /opt/conda/etc/profile.d/conda.sh; do
        if [ -f "$candidate" ]; then
            conda_sh="$candidate"
            break
        fi
    done
    if [ -n "$conda_sh" ]; then
        # shellcheck disable=SC1090
        source "$conda_sh"
        if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$env_name"; then
            conda activate "$env_name"
            echo "[init] conda env: $env_name"
        else
            echo "[init] conda env '$env_name' not found; using current Python"
        fi
    else
        echo "[init] conda not found; using current Python ($(command -v python || echo none))"
    fi
}

# -----------------------------------------------------------------------------
# HF cache + (optional) mirror. The mirror is opt-in: if HF_ENDPOINT is set
# (e.g. for clusters without direct huggingface.co access), we propagate it.
# Otherwise we leave it unset and HF clients default to huggingface.co.
# -----------------------------------------------------------------------------
rg_setup_hf() {
    local default_hf="$(rg_outputs_root)/hf_cache"
    export HF_HOME="${HF_HOME:-$default_hf}"
    mkdir -p "$HF_HOME"
    echo "[init] HF_HOME: $HF_HOME"
    if [ -n "${HF_ENDPOINT:-}" ]; then
        export HF_ENDPOINT
        echo "[init] HF_ENDPOINT: $HF_ENDPOINT"
    fi
}

rg_outputs_root() {
    # Resolve relative to the repo root (parent of scripts/).
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "${OUTPUTS_ROOT:-$repo_root/outputs}"
}

rg_log_dir() {
    echo "${LOG_DIR:-$(rg_outputs_root)/logs}"
}

rg_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}
