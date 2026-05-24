#!/bin/bash
# 72B preference STRICT-SELF judge (72B judges its own candidates), L=5.
# Matches the original 8B strict-self methodology. Runs AFTER the DeepSeek
# pref pass: reuses its candidate cache (cache-hit, no re-sampling); both need
# the 72B on all 4 GPUs so they cannot overlap.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16 RG_TP=4
LOG=/root/autodl-tmp/h1_72b_pref_self.log
echo "=== waiting for deepseek pref to finish $(date) ===" > $LOG
while [ ! -f /root/autodl-tmp/h1_72b_pref_done.flag ]; do sleep 120; done
echo "=== starting strict-self pref $(date) ===" >> $LOG
run_self () { local ds=$1 M=$2
  echo "[$(date +%T)] $ds strict-self START" >> $LOG
  python -m scripts.run_h1 --model qwen2.5-72b-instruct --dataset $ds --seed 0 \
    --K 32 --num-prompts $M --max-tokens 2048 --gpu-mem 0.90 \
    --judge self --judge-L 5 > /root/autodl-tmp/h1_72b_${ds}_self.log 2>&1
  echo "[$(date +%T)] $ds strict-self DONE rc=$?" >> $LOG
}
run_self ultrafeedback 1000
run_self alpaca_eval   300
echo "=== 72B STRICT-SELF PREF DONE $(date) ===" >> $LOG
touch /root/autodl-tmp/h1_72b_pref_self_done.flag
