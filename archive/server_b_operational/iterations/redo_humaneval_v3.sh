#!/bin/bash
# humaneval 72B retry #3: NCCL (custom-AR off) + max_tokens=1024 (matches 8B,
# smaller per-step footprint) to dodge the TP all-reduce CUDA crash.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=4 RG_DISABLE_CUSTOM_AR=1
LOG=/root/autodl-tmp/redo_humaneval_v3.log
pkill -9 -f "run_h1 --model qwen2.5-72b" 2>/dev/null; sleep 8
echo "=== humaneval v3 (mt=1024) START $(date) ===" > $LOG
python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset humaneval --seed 0 \
  --K 64 --num-prompts 164 --max-tokens 1024 --gpu-mem 0.85 >> $LOG 2>&1
echo "=== humaneval v3 DONE rc=$? $(date) ===" >> $LOG
