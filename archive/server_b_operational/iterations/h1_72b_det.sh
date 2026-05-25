#!/bin/bash
# H1 Qwen2.5-72B deterministic cells. ROBUST: loop download until all 37 shards
# present (network dropped last time at 19/37), only then run cells.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=4
LOG=/root/autodl-tmp/h1_72b.log
SNAP=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots
echo "=== robust download Qwen2.5-72B $(date) ===" >> $LOG
for attempt in $(seq 1 30); do
  huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
    --include "*.safetensors" "*.json" "*.txt" "tokenizer*" >> $LOG 2>&1
  n=$(ls $SNAP/*/*.safetensors 2>/dev/null | wc -l)
  inc=$(ls /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-72B-Instruct/blobs/*.incomplete 2>/dev/null | wc -l)
  echo "[$(date +%T)] attempt $attempt: shards=$n incomplete=$inc" >> $LOG
  [ "$n" -ge 37 ] && [ "$inc" -eq 0 ] && { echo "DOWNLOAD COMPLETE" >> $LOG; break; }
  sleep 15
done
n=$(ls $SNAP/*/*.safetensors 2>/dev/null | wc -l)
[ "$n" -lt 37 ] && { echo "ABORT: only $n/37 shards after retries" >> $LOG; exit 1; }
run () { local ds=$1 M=$2 MT=$3
  echo "[$(date +%T)] H1-72B $ds (M=$M) START" >> $LOG
  python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset $ds \
    --seed 0 --K 64 --num-prompts $M --max-tokens $MT --gpu-mem 0.90 \
    > /root/autodl-tmp/h1_72b_${ds}.log 2>&1
  echo "[$(date +%T)] H1-72B $ds DONE rc=$?" >> $LOG
}
run matharena   60   2048
run livecodebench 75 2048
run humaneval   164  2048
run gsm8k       1319 2048
run math        1000 2048
echo "=== H1-72B DETERMINISTIC DONE $(date) ===" >> $LOG
touch /root/autodl-tmp/h1_72b_det_done.flag
