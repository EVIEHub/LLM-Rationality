#!/bin/bash
# Parallel H1 panel orchestrator for 3 GPUs (data-parallel across cells).
#
# Each cell runs as a separate vLLM process pinned to a specific GPU via
# CUDA_VISIBLE_DEVICES. Cells are scheduled in batches of 3; the script
# waits for each batch before starting the next so logs and progress are
# tractable.
#
# Wall-time estimates (Tulu-3-8B already cached, ~75 min/cell at K=64):
#   - 9-cell sub-panel  (one model):    3 batches × ~75 min ≈ 3.75 hr
#   - 27-cell full panel (3 models):    9 batches × ~75 min ≈ 11.25 hr
#
# Per-cell logs: /root/autodl-tmp/h1_par_<gpu>_<dataset>_seed<S>.log
# Per-cell results JSON: ${results_dir}/h1/<model>_<dataset>_seed<S>.json
# Verifier audit log appends are atomic for single-line writes (POSIX
# guarantees up to PIPE_BUF) so concurrent appends are safe.
#
# Usage (run inside tmux so SSH disconnects don't kill it):
#
#     # one model (default: tulu3-8b-rlvr)
#     tmux new-session -d -s h1_par \
#         "cd ~/rational-gap-of-LLM-reasoning && bash scripts/run_h1_panel_parallel.sh \
#          > /root/autodl-tmp/h1_par.log 2>&1"
#
#     # full panel (3 models)
#     tmux new-session -d -s h1_par \
#         "cd ~/rational-gap-of-LLM-reasoning && bash scripts/run_h1_panel_parallel.sh \
#          tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct \
#          > /root/autodl-tmp/h1_par.log 2>&1"

set -uo pipefail
# NOTE: no `-e` — we want a single-cell failure to log + continue, not
# abort the whole panel. Each cell's exit status is logged but does not
# stop the orchestrator.

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

declare -A NUM_PROMPTS=(
    [gsm8k]=1319
    [math]=1000
    [humaneval]=164
)

# Models can be overridden by passing aliases as positional args.
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(tulu3-8b-rlvr)
fi
DATASETS=(gsm8k math humaneval)
SEEDS=(0 1 2)
NUM_GPUS=3

# Build the cell list in MODEL-OUTER order so each model's weights stay
# resident across its (dataset, seed) cells (vLLM reloads per process,
# but the disk cache is hot — load is ~30s vs cold-download ~5 min).
CELLS=()
for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
        for s in "${SEEDS[@]}"; do
            CELLS+=("${m}|${d}|${s}")
        done
    done
done

echo "=== $(date) === parallel H1 panel ==="
echo "Models:  ${MODELS[*]}"
echo "Cells:   ${#CELLS[@]}"
echo "GPUs:    ${NUM_GPUS}"
TOTAL_BATCHES=$(( (${#CELLS[@]} + NUM_GPUS - 1) / NUM_GPUS ))
echo "Batches: ${TOTAL_BATCHES}"
echo

run_one_cell() {
    local gpu_idx="$1"
    local cell_spec="$2"
    IFS='|' read -r model dataset seed <<< "$cell_spec"
    local np="${NUM_PROMPTS[$dataset]}"
    local logfile="/root/autodl-tmp/h1_par_gpu${gpu_idx}_${model}_${dataset}_seed${seed}.log"

    echo "[GPU $gpu_idx] $(date +%H:%M:%S) START ${model} × ${dataset} × seed=${seed} N=${np}"
    local t0=$(date +%s)

    CUDA_VISIBLE_DEVICES="$gpu_idx" python -m scripts.run_h1 \
        --model "${model}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --K 64 \
        --num-prompts "${np}" \
        --max-tokens 1024 \
        --gpu-mem 0.85 \
        > "$logfile" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) DONE  ${model} × ${dataset} × seed=${seed} (${dur}s)"
    else
        echo "[GPU $gpu_idx] $(date +%H:%M:%S) FAIL  ${model} × ${dataset} × seed=${seed} (rc=$rc, log: $logfile)"
    fi
}

PANEL_T0=$(date +%s)
batch_idx=0
for ((i=0; i<${#CELLS[@]}; i+=NUM_GPUS)); do
    batch_idx=$((batch_idx + 1))
    echo
    echo "=== Batch ${batch_idx}/${TOTAL_BATCHES} === $(date +%H:%M:%S) ==="

    pids=()
    for ((j=0; j<NUM_GPUS; j++)); do
        cell_idx=$((i + j))
        if [ $cell_idx -lt ${#CELLS[@]} ]; then
            run_one_cell "$j" "${CELLS[$cell_idx]}" &
            pids+=($!)
        fi
    done

    # Wait for all cells in this batch.
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "=== Batch ${batch_idx}/${TOTAL_BATCHES} done (${#pids[@]} cells) ==="
done

PANEL_DURATION=$(( $(date +%s) - PANEL_T0 ))
TOTAL_GPU_HRS=$(awk "BEGIN{printf \"%.2f\", ${NUM_GPUS} * ${PANEL_DURATION} / 3600}")
echo
echo "=== $(date) === ALL CELLS DONE ==="
echo "Wall time:        ${PANEL_DURATION}s ($(awk "BEGIN{printf \"%.2f\", ${PANEL_DURATION}/3600}") hr)"
echo "GPU-hr billable:  ${TOTAL_GPU_HRS} (${NUM_GPUS} GPUs × wall hr)"
echo
echo "Per-cell results: /root/autodl-tmp/rg_outputs/results/h1/"
ls -la /root/autodl-tmp/rg_outputs/results/h1/ 2>/dev/null || true
