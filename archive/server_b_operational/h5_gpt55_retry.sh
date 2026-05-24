#!/bin/bash
# Retry gpt-5.5 H5 panel (H1 + H3 on matharena + livecodebench) after quota refill.
# Sets HF_HOME/OUTPUTS_ROOT explicitly because run_h5_panel.sh's rg_setup_hf
# defaults to the small overlay disk where the datasets aren't cached.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=16
LOG=/root/autodl-tmp/h5_gpt55_retry.log
echo "=== gpt-5.5 H5 retry start $(date) ===" > $LOG
echo "[env] OUTPUTS_ROOT=$OUTPUTS_ROOT HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT" >> $LOG
MODELS="gpt-5.5" bash scripts/run_h5_panel.sh >> $LOG 2>&1
rc=$?
echo "=== gpt-5.5 H5 retry DONE rc=$rc $(date) ===" >> $LOG
[ $rc -eq 0 ] && touch /root/autodl-tmp/h5_gpt55_retry_done.flag
