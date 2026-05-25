#!/bin/bash
# v4 retry: address both deterministic failures with vllm-runtime workarounds.
#
#   Qwen-7B math L=256: CUDA illegal memory at prompt 280 (deterministic across
#     3 prior attempts). Likely a FlashAttention kernel bug at this specific
#     prompt+config. Workaround: VLLM_ATTENTION_BACKEND=XFORMERS switches to a
#     different attention implementation that has separate code paths and
#     should not hit the same bug.
#
#   Llama-8B matharena L=2048: CUDA OOM (5.9 GiB needed, ~800 MiB free) even
#     at gpu-mem=0.90. K=64 × max_tokens=2112 × M=60 KV cache exceeds 80 GiB
#     in fp16. Workaround: RG_KV_DTYPE=fp8 (new env hook in vllm_runner.py)
#     halves KV cache memory, dropping the per-cell footprint by ~30 GiB.
#
# Sequential on GPU 0. Panel defaults otherwise.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16
[ -f /root/.config/rg-gap.env ] && { set -a; source /root/.config/rg-gap.env; set +a; }
LOG=/root/autodl-tmp/h4_2cells_1gpu_v4.log
echo "=== H4 v4 1-GPU retry start $(date) ===" > $LOG

# Cell 1: Qwen-7B math L=256 with XFORMERS attention (dodges FlashAttn bug)
echo "[$(date +%T)] START qwen_math_L256 (VLLM_ATTENTION_BACKEND=XFORMERS, gpu-mem=0.70)" >> $LOG
CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=XFORMERS python -m scripts.run_h4 \
  --model qwen2.5-7b-instruct --dataset math --L 256 --seed 0 \
  --K 64 --num-prompts 500 --gpu-mem 0.70 \
  > /root/autodl-tmp/h4_v4_qwen_cell.log 2>&1
QRC=$?
echo "[$(date +%T)] qwen rc=$QRC" >> $LOG

# Cell 2: Llama-8B matharena L=2048 with FP8 KV cache (fits in 80 GiB)
echo "[$(date +%T)] START llama_matharena_L2048 (RG_KV_DTYPE=fp8, gpu-mem=0.90)" >> $LOG
CUDA_VISIBLE_DEVICES=0 RG_KV_DTYPE=fp8 python -m scripts.run_h4 \
  --model llama3.1-8b-instruct --dataset matharena --L 2048 --seed 0 \
  --K 64 --num-prompts 60 --gpu-mem 0.90 \
  > /root/autodl-tmp/h4_v4_llama_cell.log 2>&1
LRC=$?
echo "[$(date +%T)] llama rc=$LRC" >> $LOG

echo "=== H4 v4 DONE qwen=$QRC llama=$LRC $(date) ===" >> $LOG
[ $QRC -eq 0 ] && [ $LRC -eq 0 ] && touch /root/autodl-tmp/h4_2cells_1gpu_v4_done.flag
