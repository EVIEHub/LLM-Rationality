#!/bin/bash
# Redo the 72B humaneval cell (lost to cleanup churn). Waits for recover2 to
# finish so it does not contend for GPUs, then runs with a VRAM-free guard.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=4
LOG=/root/autodl-tmp/redo_humaneval.log
echo "=== waiting for recover2 to finish $(date) ===" > $LOG
while [ ! -f /root/autodl-tmp/recover2_72b_done.flag ]; do sleep 120; done
pkill -9 -f "run_h1 --model qwen2.5-72b" 2>/dev/null; sleep 10
echo "=== humaneval redo START $(date) ===" >> $LOG
python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset humaneval --seed 0 \
  --K 64 --num-prompts 164 --max-tokens 2048 --gpu-mem 0.90 >> $LOG 2>&1
echo "=== humaneval redo DONE rc=$? $(date) ===" >> $LOG
touch /root/autodl-tmp/redo_humaneval_done.flag
