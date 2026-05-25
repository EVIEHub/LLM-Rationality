#!/bin/bash
# Single-GPU retry of the 2 H4 cells that failed yesterday:
#   - Qwen2.5-7B math L=256 (deterministic CUDA illegal memory at prompt 280
#     under 4-GPU concurrency; hoping fresh single-GPU CUDA context avoids it)
#   - Llama-3.1-8B matharena L=2048 (OOM on the 4-GPU box; with conservative
#     max_num_seqs and gpu-mem on a fresh single-GPU instance, KV fits)
#
# Sequential: GPU 0 only. ~15-30 min per cell. Result files land in the
# canonical /root/rational_gap_outputs/results/h4/ per paths.yaml.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=16
[ -f /root/.config/rg-gap.env ] && { set -a; source /root/.config/rg-gap.env; set +a; }
LOG=/root/autodl-tmp/h4_2cells_1gpu.log
echo "=== H4 2-cells 1-GPU retry start $(date) ===" > $LOG

run_cell() {  # $1 = label, $2 = model, $3 = dataset, $4 = L, $5 = M, $6 = gpu_mem, $7 = max_seqs
  local label=$1 model=$2 ds=$3 L=$4 M=$5 gm=$6 ms=$7
  local cell_log=/root/autodl-tmp/h4_${label}_cell.log
  echo "[$(date +%T)] START $label  (max_num_seqs=$ms, gpu-mem=$gm)" >> $LOG
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 RG_MAX_NUM_SEQS=$ms python -m scripts.run_h4 \
    --model $model --dataset $ds --L $L --seed 0 \
    --K 64 --num-prompts $M --gpu-mem $gm > "$cell_log" 2>&1
  local rc=$? dur=$(( $(date +%s) - t0 ))
  echo "[$(date +%T)] $([ $rc -eq 0 ] && echo DONE || echo FAIL) $label  (${dur}s, rc=$rc, $cell_log)" >> $LOG
  return $rc
}

# Cell 1: Qwen-7B math L=256 — smaller batch to dodge the vllm bug at prompt 280.
run_cell qwen_math_L256       qwen2.5-7b-instruct  math       256  500  0.85  16
QWEN_RC=$?

# Cell 2: Llama-8B matharena L=2048 — very small batch + lower gpu-mem so KV fits.
run_cell llama_matharena_L2048 llama3.1-8b-instruct matharena  2048 60   0.55  4
LLAMA_RC=$?

echo "=== H4 2-cells 1-GPU retry DONE qwen=$QWEN_RC llama=$LLAMA_RC $(date) ===" >> $LOG
[ $QWEN_RC -eq 0 ] && [ $LLAMA_RC -eq 0 ] && touch /root/autodl-tmp/h4_2cells_1gpu_done.flag
