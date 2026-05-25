#!/bin/bash
# v6 — Llama-8B matharena L=2048 with CPU swap workaround.
# Previous attempts: fp8 / fp8_e4m3 / fp8_e5m2 all hit a Triton CompilationError
# in this vllm-dev install. Workaround here: keep KV in fp16 but give vllm a
# big CPU swap pool so it can spill blocks off GPU when needed (instead of
# the default 4 GiB which aborts the run).
#
# Two attempts: gpu-mem=0.70 + swap=32GB first (more headroom), then 0.55+64GB
# if the first fails to allocate.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16
[ -f /root/.config/rg-gap.env ] && { set -a; source /root/.config/rg-gap.env; set +a; }
LOG=/root/autodl-tmp/h4_llama_only_v6.log
echo "=== H4 Llama v6 (swap-space workaround) start $(date) ===" > $LOG

try_run() {  # $1 = gpu_mem, $2 = swap_gb, $3 = label
  local gm=$1 swap=$2 label=$3
  echo "[$(date +%T)] try gpu-mem=$gm swap=$swap GiB ($label)" >> $LOG
  CUDA_VISIBLE_DEVICES=0 RG_SWAP_SPACE_GB=$swap python -m scripts.run_h4 \
    --model llama3.1-8b-instruct --dataset matharena --L 2048 --seed 0 \
    --K 64 --num-prompts 60 --gpu-mem $gm \
    > /root/autodl-tmp/h4_v6_llama_${label}.log 2>&1
  return $?
}

try_run 0.70 32 attempt1
RC=$?
echo "[$(date +%T)] attempt1 rc=$RC" >> $LOG
if [ $RC -ne 0 ]; then
  try_run 0.55 64 attempt2
  RC=$?
  echo "[$(date +%T)] attempt2 rc=$RC" >> $LOG
fi

echo "=== H4 Llama v6 DONE rc=$RC $(date) ===" >> $LOG
[ $RC -eq 0 ] && touch /root/autodl-tmp/h4_llama_only_v6_done.flag
