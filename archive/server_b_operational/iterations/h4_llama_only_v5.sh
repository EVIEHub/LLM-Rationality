#!/bin/bash
# v5 — Llama-8B matharena L=2048 only. Qwen already succeeded in v4.
# fp8 (shorthand) triggered a Triton CompilationError. Try fp8_e4m3 explicit.
# If that also fails, try fp8_e5m2.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16
[ -f /root/.config/rg-gap.env ] && { set -a; source /root/.config/rg-gap.env; set +a; }
LOG=/root/autodl-tmp/h4_llama_only_v5.log
echo "=== H4 Llama v5 start $(date) ===" > $LOG

for kv in fp8_e4m3 fp8_e5m2; do
  echo "[$(date +%T)] try RG_KV_DTYPE=$kv" >> $LOG
  CUDA_VISIBLE_DEVICES=0 RG_KV_DTYPE=$kv python -m scripts.run_h4 \
    --model llama3.1-8b-instruct --dataset matharena --L 2048 --seed 0 \
    --K 64 --num-prompts 60 --gpu-mem 0.90 \
    > /root/autodl-tmp/h4_v5_llama_${kv}.log 2>&1
  rc=$?
  echo "[$(date +%T)] $kv rc=$rc" >> $LOG
  if [ $rc -eq 0 ]; then
    echo "[$(date +%T)] success with $kv" >> $LOG
    touch /root/autodl-tmp/h4_llama_only_v5_done.flag
    exit 0
  fi
done
echo "=== H4 Llama v5 ALL ATTEMPTS FAILED $(date) ===" >> $LOG
exit 1
