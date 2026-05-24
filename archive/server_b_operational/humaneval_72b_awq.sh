#!/bin/bash
# HumanEval-72B AWQ-INT4 single-GPU (INT8 OOMed on activation). AWQ ~40GB
# leaves lots of KV/activation headroom on 80GB. Robust download + run.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=1
LOG=/root/autodl-tmp/humaneval_72b_awq.log
M=Qwen/Qwen2.5-72B-Instruct-AWQ
SNAP=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ
echo "=== AWQ download + run start $(date) ===" > $LOG
for a in $(seq 1 30); do
  huggingface-cli download $M --include "*.safetensors" "*.json" "*.txt" "tokenizer*" >> $LOG 2>&1
  inc=$(ls $SNAP/blobs/*.incomplete 2>/dev/null | wc -l)
  n=$(ls $SNAP/snapshots/*/*.safetensors 2>/dev/null | wc -l)
  echo "[$(date +%T)] attempt $a: shards=$n incomplete=$inc" >> $LOG
  [ "$inc" -eq 0 ] && [ "$n" -ge 1 ] && { echo "DOWNLOAD COMPLETE" >> $LOG; break; }
  sleep 15
done
echo "=== run humaneval AWQ (1 GPU) $(date) ===" >> $LOG
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_h1 --model qwen2.5-72b-instruct-awq --dataset humaneval \
  --seed 0 --K 64 --num-prompts 164 --max-tokens 2048 --gpu-mem 0.85 >> $LOG 2>&1
echo "=== AWQ humaneval DONE rc=$? $(date) ===" >> $LOG
touch /root/autodl-tmp/humaneval_72b_awq_done.flag
