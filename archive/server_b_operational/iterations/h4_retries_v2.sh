#!/bin/bash
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8
LOG=/root/autodl-tmp/h4_retries_v2.log
echo "=== H4 retries v2 start $(date) ===" > $LOG

# Qwen math L=256: prompt 280 deterministically triggers CUDA illegal memory.
# Try smaller max_num_seqs to change the batch scheduling, may avoid the buggy state.
CUDA_VISIBLE_DEVICES=0 RG_MAX_NUM_SEQS=16 python -m scripts.run_h4 \
  --model qwen2.5-7b-instruct --dataset math --L 256 --seed 0 \
  --K 64 --num-prompts 500 --gpu-mem 0.85 \
  > /root/autodl-tmp/h4_retry_v2_qwen_math_L256.log 2>&1 &
QWEN_PID=$!
echo "qwen retry v2 PID=$QWEN_PID (RG_MAX_NUM_SEQS=16)" >> $LOG

# Llama matharena L=2048: OOM. Use smaller batch + lower gpu-mem to fit.
CUDA_VISIBLE_DEVICES=1 RG_MAX_NUM_SEQS=8 python -m scripts.run_h4 \
  --model llama3.1-8b-instruct --dataset matharena --L 2048 --seed 0 \
  --K 64 --num-prompts 60 --gpu-mem 0.55 \
  > /root/autodl-tmp/h4_retry_v2_llama_matharena_L2048.log 2>&1 &
LLAMA_PID=$!
echo "llama retry v2 PID=$LLAMA_PID (RG_MAX_NUM_SEQS=8, gpu-mem=0.55)" >> $LOG

wait $QWEN_PID; QR=$?
wait $LLAMA_PID; LR=$?
echo "=== H4 retries v2 DONE qwen=$QR llama=$LR $(date) ===" >> $LOG
[ $QR -eq 0 ] && [ $LR -eq 0 ] && touch /root/autodl-tmp/h4_retries_v2_done.flag
