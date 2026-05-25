#!/bin/bash
# Redo 72B humaneval with custom all-reduce DISABLED (it hit a custom_all_reduce
# CUDA illegal-memory-access under TP). Waits for recover3 to finish first.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=4 RG_DISABLE_CUSTOM_AR=1
LOG=/root/autodl-tmp/redo_humaneval_v2.log
echo "=== waiting for recover3 $(date) ===" > $LOG
while [ ! -f /root/autodl-tmp/recover3_72b_done.flag ]; do sleep 120; done
pkill -9 -f "run_h1 --model qwen2.5-72b" 2>/dev/null; sleep 10
echo "=== humaneval redo (no custom AR) START $(date) ===" >> $LOG
python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset humaneval --seed 0 \
  --K 64 --num-prompts 164 --max-tokens 2048 --gpu-mem 0.90 >> $LOG 2>&1
echo "=== humaneval redo DONE rc=$? $(date) ===" >> $LOG
touch /root/autodl-tmp/redo_humaneval_v2_done.flag
