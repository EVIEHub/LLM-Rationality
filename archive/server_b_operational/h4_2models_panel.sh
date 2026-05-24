#!/bin/bash
# H4 reasoning-budget sweep for 2 new instruct-family models on B.
# Adds Qwen-7B + Llama-8B to the H4 table so it becomes a cross-family
# study (Tülu-8B already done in h4/). 42 new cells.
#
# Per model: 3 datasets × 7 L = 21 cells, batched 4-at-a-time across 4 GPUs
# in run_h4_panel.sh. Estimated walltime ~1.5–2.5 h per model on B.
# Total ~3–5 h plus ~15 min model downloads.
#
# Pre-downloads each model with retry (vllm auto-download is fragile to
# network drops, learned the hard way on H2-70B).
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
LOG=/root/autodl-tmp/h4_2models_panel.log
echo "=== H4 2-models panel start $(date) ===" > $LOG

predownload() {  # $1 = hf_id, $2 = models--<org>--<name>
  local hf_id=$1 snap=/root/autodl-tmp/hf_cache/hub/$2
  for a in $(seq 1 30); do
    huggingface-cli download "$hf_id" --include "*.safetensors" "*.json" "*.txt" "tokenizer*" >> $LOG 2>&1
    local inc=$(ls $snap/blobs/*.incomplete 2>/dev/null | wc -l)
    local n=$(ls $snap/snapshots/*/*.safetensors 2>/dev/null | wc -l)
    echo "[$(date +%T)] $hf_id attempt $a: shards=$n incomplete=$inc" >> $LOG
    [ "$inc" -eq 0 ] && [ "$n" -ge 1 ] && { echo "$hf_id DOWNLOAD COMPLETE" >> $LOG; return 0; }
    sleep 15
  done
  echo "$hf_id DOWNLOAD FAILED after 30 attempts" >> $LOG; return 1
}

run_model() {  # $1 = alias, $2 = hf_id, $3 = snap dir
  predownload "$2" "$3" || { echo "=== ABORT: $1 download failed ===" >> $LOG; return 1; }
  echo "=== running H4 panel for $1 $(date) ===" >> $LOG
  MODEL="$1" DATASETS="gsm8k math matharena" \
    bash scripts/run_h4_panel.sh --num-gpus 4 >> $LOG 2>&1
  echo "=== $1 done $(date) ===" >> $LOG
}

T0=$(date +%s)
run_model qwen2.5-7b-instruct  Qwen/Qwen2.5-7B-Instruct      models--Qwen--Qwen2.5-7B-Instruct
run_model llama3.1-8b-instruct meta-llama/Llama-3.1-8B-Instruct  models--meta-llama--Llama-3.1-8B-Instruct
echo "=== H4 2-models panel DONE in $(( $(date +%s) - T0 ))s ===" >> $LOG
touch /root/autodl-tmp/h4_2models_panel_done.flag
