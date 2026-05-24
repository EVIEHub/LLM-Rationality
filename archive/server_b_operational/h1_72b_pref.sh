#!/bin/bash
set -a; source /root/.config/rg-gap.env; set +a
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=4
LOG=/root/autodl-tmp/h1_72b_pref.log
[ -z "$DS_API_KEY" ] || [ -z "$DS_BASE_URL" ] && { echo "ERROR: creds missing" | tee $LOG; exit 1; }
echo "=== waiting for deterministic battery $(date) ===" > $LOG
while [ ! -f /root/autodl-tmp/h1_72b_det_done.flag ]; do sleep 120; done
echo "=== starting preference $(date) ===" >> $LOG
run_pref () { local ds=$1 M=$2
  echo "[$(date +%T)] $ds (M=$M) judge=api START" >> $LOG
  python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset $ds \
    --seed 0 --K 32 --num-prompts $M --max-tokens 2048 --gpu-mem 0.90 \
    --judge api --judge-model deepseek-v4-flash --judge-L 3 --api-concurrency 20 \
    > /root/autodl-tmp/h1_72b_${ds}.log 2>&1
  echo "[$(date +%T)] $ds DONE rc=$?" >> $LOG
}
run_pref ultrafeedback 1000
run_pref alpaca_eval   300
echo "=== H1-72B PREFERENCE DONE $(date) ===" >> $LOG
touch /root/autodl-tmp/h1_72b_pref_done.flag
