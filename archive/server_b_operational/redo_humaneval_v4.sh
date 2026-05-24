#!/bin/bash
# humaneval 72B attempt #4: max_num_seqs=32 (small batch -> small all-reduce
# tensors) + NCCL. Targets the TP all-reduce CUDA crash directly.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16
export RG_TP=4 RG_DISABLE_CUSTOM_AR=1 RG_MAX_NUM_SEQS=32
LOG=/root/autodl-tmp/redo_humaneval_v4.log
echo "=== humaneval v4 (max_num_seqs=32) START $(date) ===" > $LOG
python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset humaneval --seed 0 \
  --K 64 --num-prompts 164 --max-tokens 2048 --gpu-mem 0.85 >> $LOG 2>&1
echo "=== humaneval v4 DONE rc=$? $(date) ===" >> $LOG
