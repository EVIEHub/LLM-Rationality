#!/bin/bash
# H1 sub-panel for the Tülu-3-RLVR model.
#
# Runs all 9 cells: gsm8k/math/humaneval x seed in {0,1,2} at K=64.
# Per-dataset prompt counts:
#   GSM8K     : 1319  (full test)
#   MATH      : 1000  (subsampled from 5000 for compute budget; statistical
#                      power for prompt-bootstrap CI is ample at M=1000)
#   HumanEval :  164  (full test)
#
# Each cell calls scripts/run_h1.py which:
#   - Honours the cache (a re-run after a verifier change is near-instant)
#   - Writes per-cell results to ${results_dir}/h1/<cell>.json
#   - Appends per-decision audit logs to ${logs_dir}/verifier/<dataset>_log.jsonl
#   - Appends compute usage to ${logs_dir}/compute_budget.jsonl
#
# Total wall time estimate: ~7 GPU-hr (Tulu-3-8B already cached).
#
# Run inside a tmux session so SSH disconnects don't kill the loop:
#   tmux new-session -d -s h1_tulu \
#       'cd ~/rational-gap-of-LLM-reasoning && bash scripts/run_h1_tulu.sh \
#        > /root/autodl-tmp/h1_tulu.log 2>&1'

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

MODEL=tulu3-8b-rlvr

declare -A NUM_PROMPTS=(
    [gsm8k]=1319
    [math]=1000
    [humaneval]=164
)

TOTAL_T0=$(date +%s)
echo "=== $(date) === H1 sub-panel for ${MODEL} ==="

for ds in gsm8k math humaneval; do
    np="${NUM_PROMPTS[$ds]}"
    for seed in 0 1 2; do
        T0=$(date +%s)
        echo
        echo "--- $(date +%H:%M:%S) | START ${ds} seed=${seed} N=${np} ---"
        python -m scripts.run_h1 \
            --model "${MODEL}" \
            --dataset "${ds}" \
            --seed "${seed}" \
            --K 64 \
            --num-prompts "${np}" \
            --max-tokens 1024
        DURATION=$(( $(date +%s) - T0 ))
        echo "--- $(date +%H:%M:%S) | DONE  ${ds} seed=${seed} (${DURATION}s) ---"
    done
done

TOTAL_DURATION=$(( $(date +%s) - TOTAL_T0 ))
echo
echo "=== $(date) === ALL CELLS DONE in ${TOTAL_DURATION}s ($(awk "BEGIN{printf \"%.2f\", ${TOTAL_DURATION}/3600}") GPU-hr) ==="
