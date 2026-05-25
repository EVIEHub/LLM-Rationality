#!/bin/bash
# v3: drop RG_MAX_NUM_SEQS (it triggered vllm hang). Use panel defaults
# (gpu-mem=0.70) that worked for the 19 successful H4 cells; bump to 0.90
# for the OOM-prone Llama L=2048 cell to give KV cache more room.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16
[ -f /root/.config/rg-gap.env ] && { set -a; source /root/.config/rg-gap.env; set +a; }
LOG=/root/autodl-tmp/h4_2cells_1gpu_v3.log
echo "=== H4 v3 1-GPU retry start $(date) ===" > $LOG

# Cell 1: Qwen-7B math L=256 - panel defaults
echo "[$(date +%T)] START qwen_math_L256 (defaults: gpu-mem=0.70)" >> $LOG
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_h4 \
  --model qwen2.5-7b-instruct --dataset math --L 256 --seed 0 \
  --K 64 --num-prompts 500 --gpu-mem 0.70 \
  > /root/autodl-tmp/h4_v3_qwen_cell.log 2>&1
QRC=$?
echo "[$(date +%T)] qwen rc=$QRC" >> $LOG

# Cell 2: Llama-8B matharena L=2048 - high gpu-mem to fit KV
echo "[$(date +%T)] START llama_matharena_L2048 (gpu-mem=0.90)" >> $LOG
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_h4 \
  --model llama3.1-8b-instruct --dataset matharena --L 2048 --seed 0 \
  --K 64 --num-prompts 60 --gpu-mem 0.90 \
  > /root/autodl-tmp/h4_v3_llama_cell.log 2>&1
LRC=$?
echo "[$(date +%T)] llama rc=$LRC" >> $LOG

echo "=== H4 v3 DONE qwen=$QRC llama=$LRC $(date) ===" >> $LOG
[ $QRC -eq 0 ] && [ $LRC -eq 0 ] && touch /root/autodl-tmp/h4_2cells_1gpu_v3_done.flag
