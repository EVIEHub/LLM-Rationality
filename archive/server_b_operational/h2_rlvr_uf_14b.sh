#!/bin/bash
# Single missing H2 14B-judge cell: tulu3-8b-rlvr x ultrafeedback.
# Previous attempt failed at 04:42:03 during the disk-full event.
# Samples cache + 14B judge should both already be on disk.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16
LOG=/root/autodl-tmp/h2_rlvr_uf_14b.log
echo "=== H2 rlvr-uf 14B-judge start $(date) ===" > $LOG
python -m scripts.run_h2 \
  --model tulu3-8b-rlvr --dataset ultrafeedback --seed 0 \
  --K 32 --num-prompts 1000 --max-tokens 512 --gpu-mem 0.7 \
  --judge self --judge-local-model qwen2.5-14b-instruct --judge-L 5 \
  >> $LOG 2>&1
rc=$?
echo "=== H2 rlvr-uf 14B-judge DONE rc=$rc $(date) ===" >> $LOG
[ $rc -eq 0 ] && touch /root/autodl-tmp/h2_rlvr_uf_14b_done.flag
