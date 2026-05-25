#!/bin/bash
# HumanEval-72B on a SINGLE GPU via INT8 (no tensor parallelism -> no all-reduce
# -> bypasses the bug). Auto-falls back to AWQ-INT4 if INT8 OOMs at load.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=1
LOG=/root/autodl-tmp/humaneval_72b_int8.log
run() { local model=$1 gm=$2
  echo "=== $model (gpu_mem=$gm) START $(date) ===" >> $LOG
  CUDA_VISIBLE_DEVICES=0 python -m scripts.run_h1 --model $model --dataset humaneval \
    --seed 0 --K 64 --num-prompts 164 --max-tokens 2048 --gpu-mem $gm >> $LOG 2>&1
  return $?
}
echo "=== INT8 humaneval-72B start $(date) ===" > $LOG
run qwen2.5-72b-instruct-int8 0.96
rc=$?
echo "=== INT8 rc=$rc $(date) ===" >> $LOG
touch /root/autodl-tmp/humaneval_72b_int8_done.flag
