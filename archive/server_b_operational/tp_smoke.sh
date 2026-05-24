#!/bin/bash
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8 RG_TP=4
LOG=/root/autodl-tmp/tp_smoke.log
echo "=== TP=4 NVLink smoke (small model) $(date) ===" > $LOG
timeout 400 python -m scripts.run_h1 --model qwen2.5-1.5b-instruct --dataset gsm8k \
  --seed 0 --K 4 --num-prompts 8 --max-tokens 256 --gpu-mem 0.85 >> $LOG 2>&1
echo "=== smoke EXIT=$? (124=hang/timeout) $(date) ===" >> $LOG
