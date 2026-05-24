#!/bin/bash
# HumanEval-72B INT8 single-GPU, v2: ROBUST download (retry until complete;
# v1 died on a mid-download network drop). TP=1 -> no all-reduce -> no bug.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=1
LOG=/root/autodl-tmp/humaneval_72b_int8_v2.log
M=Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8
SNAP=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-72B-Instruct-GPTQ-Int8
echo "=== robust INT8 download $(date) ===" > $LOG
for a in $(seq 1 30); do
  huggingface-cli download $M --include "*.safetensors" "*.json" "*.txt" "tokenizer*" >> $LOG 2>&1
  inc=$(ls $SNAP/blobs/*.incomplete 2>/dev/null | wc -l)
  n=$(ls $SNAP/snapshots/*/*.safetensors 2>/dev/null | wc -l)
  echo "[$(date +%T)] attempt $a: shards=$n incomplete=$inc" >> $LOG
  [ "$inc" -eq 0 ] && [ "$n" -ge 1 ] && { echo "DOWNLOAD COMPLETE ($n shards)" >> $LOG; break; }
  sleep 15
done
echo "=== run humaneval INT8 (1 GPU) $(date) ===" >> $LOG
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_h1 --model qwen2.5-72b-instruct-int8 --dataset humaneval \
  --seed 0 --K 64 --num-prompts 164 --max-tokens 2048 --gpu-mem 0.96 >> $LOG 2>&1
echo "=== INT8 humaneval v2 DONE rc=$? $(date) ===" >> $LOG
touch /root/autodl-tmp/humaneval_72b_int8_v2_done.flag
