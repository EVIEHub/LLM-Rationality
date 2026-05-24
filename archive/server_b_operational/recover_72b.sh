#!/bin/bash
# Recovery: re-run the 72B H1 cells that OOM-failed (VRAM-leak cascade).
# matharena + livecodebench already succeeded; redo humaneval/gsm8k/math +
# preference (deepseek) + preference (strict-self). A wait_gpu_free guard
# kills lingering TP workers and waits for VRAM to drain before each cell.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=16 RG_TP=4
set -a; source /root/.config/rg-gap.env 2>/dev/null; set +a   # DS creds for api judge
LOG=/root/autodl-tmp/recover_72b.log
rm -f /root/autodl-tmp/h1_72b_*_done.flag
echo "=== 72B recovery start $(date) ===" > $LOG

wait_gpu_free() {
  pkill -9 -f "scripts.run_h1" 2>/dev/null; sleep 8
  for i in $(seq 1 24); do
    mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${mx:-9999}" -lt 2000 ] && { echo "[$(date +%T)] GPU free (max ${mx}MiB)" >> $LOG; return 0; }
    sleep 10
  done
  echo "[$(date +%T)] WARN GPU not fully free (max ${mx}MiB)" >> $LOG
}
cell() {  # dataset M extra-args...
  local ds=$1 M=$2; shift 2
  wait_gpu_free
  echo "[$(date +%T)] $ds (M=$M) $* START" >> $LOG
  python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset $ds --seed 0 \
    --K "${K:-64}" --num-prompts $M --max-tokens 2048 --gpu-mem 0.90 "$@" \
    > /root/autodl-tmp/recover_72b_${ds}$(echo "$*" | tr -dc "a-z0-9").log 2>&1
  echo "[$(date +%T)] $ds DONE rc=$?" >> $LOG
}
# 1) deterministic failures (K=64)
cell humaneval 164
cell gsm8k 1319
cell math 1000
# 2) preference, deepseek API judge (K=32)
K=32 cell ultrafeedback 1000 --judge api --judge-model deepseek-v4-flash --judge-L 3 --api-concurrency 20
K=32 cell alpaca_eval   300  --judge api --judge-model deepseek-v4-flash --judge-L 3 --api-concurrency 20
# 3) preference, strict-self judge (K=32, L=5) -- reuses pref candidate cache
K=32 cell ultrafeedback 1000 --judge self --judge-L 5
K=32 cell alpaca_eval   300  --judge self --judge-L 5
echo "=== 72B RECOVERY DONE $(date) ===" >> $LOG
touch /root/autodl-tmp/recover_72b_done.flag
