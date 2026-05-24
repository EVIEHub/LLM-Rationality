#!/bin/bash
# Re-launch alpaca deepseek re-judge (prev run killed by a stray pkill). NO pkill
# here. API-only: alpaca candidates cached + api_judge resume cache continues.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8 RG_TP=1
set -a; source /root/.config/rg-gap.env 2>/dev/null; set +a
LOG=/root/autodl-tmp/redo_alpaca_deepseek_v2.log
echo "=== alpaca deepseek re-judge v2 START $(date) ===" > $LOG
python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset alpaca_eval --seed 0 \
  --K 32 --num-prompts 300 --max-tokens 2048 --gpu-mem 0.90 \
  --judge api --judge-model deepseek-v4-flash --judge-L 3 --api-concurrency 20 >> $LOG 2>&1
echo "=== alpaca deepseek re-judge v2 DONE rc=$? $(date) ===" >> $LOG
