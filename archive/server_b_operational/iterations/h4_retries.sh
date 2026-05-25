#!/bin/bash
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8
LOG=/root/autodl-tmp/h4_retries.log
echo "=== H4 retries start $(date) ===" > $LOG
echo "[GPU 0] START qwen2.5-7b-instruct math L=256" >> $LOG &
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_h4 \
  --model qwen2.5-7b-instruct --dataset math --L 256 --seed 0 \
  --K 64 --num-prompts 500 --gpu-mem 0.85 \
  > /root/autodl-tmp/h4_retry_qwen_math_L256.log 2>&1 &
QWEN_PID=$!
echo "qwen retry PID=$QWEN_PID" >> $LOG
CUDA_VISIBLE_DEVICES=1 python -m scripts.run_h4 \
  --model llama3.1-8b-instruct --dataset matharena --L 2048 --seed 0 \
  --K 64 --num-prompts 60 --gpu-mem 0.85 \
  > /root/autodl-tmp/h4_retry_llama_matharena_L2048.log 2>&1 &
LLAMA_PID=$!
echo "llama retry PID=$LLAMA_PID" >> $LOG
wait $QWEN_PID; QR=$?
wait $LLAMA_PID; LR=$?
echo "=== H4 retries DONE qwen=$QR llama=$LR $(date) ===" >> $LOG
[ $QR -eq 0 ] && [ $LR -eq 0 ] && touch /root/autodl-tmp/h4_retries_done.flag
