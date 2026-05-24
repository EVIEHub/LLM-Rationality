#!/bin/bash
# H4 reasoning-budget sweep for Qwen2.5-7B-Instruct on B.
# Adds the Qwen family to the H4 table (Tulu-8B-RLVR already done).
# Llama-3.1-8B-Instruct postponed (gated, needs HF token).
# 3 ds x 7 L = 21 cells, ~1.5-2.5 h on 4 GPUs.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
LOG=/root/autodl-tmp/h4_qwen7b_panel.log
echo "=== H4 Qwen-7B panel start $(date) ===" > $LOG
MODEL="qwen2.5-7b-instruct" DATASETS="gsm8k math matharena" \
  bash scripts/run_h4_panel.sh --num-gpus 4 >> $LOG 2>&1
rc=$?
echo "=== H4 Qwen-7B panel DONE rc=$rc $(date) ===" >> $LOG
[ $rc -eq 0 ] && touch /root/autodl-tmp/h4_qwen7b_panel_done.flag
