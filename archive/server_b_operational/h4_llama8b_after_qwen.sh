#!/bin/bash
# Waits for Qwen-7B H4 panel to finish, then downloads + runs Llama-3.1-8B H4.
# Llama is gated → uses HF_TOKEN from /root/.config/rg-gap.env. The predownload
# retry-loop guards against network drops (same pattern as the H2-70B robust
# wrapper). Disk holds both 7B and 8B simultaneously (~30 GB total, fits easily
# in the 190 GB volume).
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
set -a; source /root/.config/rg-gap.env; set +a  # exports HF_TOKEN + others
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8

LOG=/root/autodl-tmp/h4_llama8b_after_qwen.log
QWEN_FLAG=/root/autodl-tmp/h4_qwen7b_panel_done.flag

echo "=== H4 Llama-8B after-Qwen wrapper start $(date) ===" > $LOG

# 1) Block until Qwen's done flag exists.
echo "[wait] polling $QWEN_FLAG every 60s" >> $LOG
while [ ! -f "$QWEN_FLAG" ]; do
  sleep 60
done
echo "[ok] Qwen done flag set at $(date)" >> $LOG

# 2) Predownload Llama-3.1-8B-Instruct with retry (HF_TOKEN already exported).
SNAP=/root/autodl-tmp/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct
for a in $(seq 1 30); do
  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --include "*.safetensors" "*.json" "*.txt" "tokenizer*" >> $LOG 2>&1
  inc=$(ls $SNAP/blobs/*.incomplete 2>/dev/null | wc -l)
  n=$(ls $SNAP/snapshots/*/*.safetensors 2>/dev/null | wc -l)
  echo "[$(date +%T)] llama download attempt $a: shards=$n incomplete=$inc" >> $LOG
  [ "$inc" -eq 0 ] && [ "$n" -ge 1 ] && { echo "Llama DOWNLOAD COMPLETE" >> $LOG; break; }
  sleep 15
done
if [ "$inc" -ne 0 ] || [ "$n" -lt 1 ]; then
  echo "=== ABORT: Llama download failed after 30 attempts ===" >> $LOG
  touch /root/autodl-tmp/h4_llama8b_after_qwen_FAILED.flag
  exit 1
fi

# 3) Run the H4 panel for Llama.
echo "=== running H4 panel for llama3.1-8b-instruct $(date) ===" >> $LOG
MODEL="llama3.1-8b-instruct" DATASETS="gsm8k math matharena" \
  bash scripts/run_h4_panel.sh --num-gpus 4 >> $LOG 2>&1
rc=$?
echo "=== H4 Llama-8B panel DONE rc=$rc $(date) ===" >> $LOG
[ $rc -eq 0 ] && touch /root/autodl-tmp/h4_llama8b_panel_done.flag
