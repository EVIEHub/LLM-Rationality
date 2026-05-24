#!/bin/bash
# Re-judge the 72B alpaca preference with DeepSeek (the original crashed on a
# transient HTTP 544, now retryable). API-only: alpaca candidates are cached,
# so no GPU. New run_h1.py tags the output _judge-deepseek-v4-flash.
# Waits for the recovery+humaneval chain to finish first (clean ordering).
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8 RG_TP=1
set -a; source /root/.config/rg-gap.env 2>/dev/null; set +a
LOG=/root/autodl-tmp/redo_alpaca_deepseek.log
echo "=== waiting for humaneval redo to finish $(date) ===" > $LOG
while [ ! -f /root/autodl-tmp/redo_humaneval_v2_done.flag ]; do sleep 120; done
echo "=== alpaca deepseek re-judge START $(date) ===" >> $LOG
python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset alpaca_eval --seed 0 \
  --K 32 --num-prompts 300 --max-tokens 2048 --gpu-mem 0.90 \
  --judge api --judge-model deepseek-v4-flash --judge-L 3 --api-concurrency 20 >> $LOG 2>&1
echo "=== alpaca deepseek re-judge DONE rc=$? $(date) ===" >> $LOG
touch /root/autodl-tmp/redo_alpaca_deepseek_done.flag
