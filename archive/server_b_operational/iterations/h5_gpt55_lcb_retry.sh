#!/bin/bash
# Retry the 3 failed GPT-5.5 H5 livecodebench cells at lower concurrency.
# Previous failures: 1 transient HTTP 429 "Upstream rate limit exceeded" per
# cell, taking down a cell that was 95-98% complete. The resume cache from
# the previous run preserves the successful calls — this retry only fills the
# missing ones (~1-10 per cell, minutes total).
#
# Restrict the panel via env overrides:
#   DATASETS="livecodebench"     skips matharena (all done)
#   TAUS="0.7 1.0"               skips livecodebench × h3 × t=0.0 (already done)
#   HYPS="h1 h3"                 keeps livecodebench × h1 + livecodebench × h3 cells
#   CONC=5                       lower concurrency reduces upstream 429 risk
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=16
LOG=/root/autodl-tmp/h5_gpt55_lcb_retry.log
echo "=== gpt-5.5 LCB retry start $(date) ===" > $LOG
MODELS="gpt-5.5" DATASETS="livecodebench" TAUS="0.7 1.0" HYPS="h1 h3" CONC=5 \
  bash scripts/run_h5_panel.sh >> $LOG 2>&1
rc=$?
echo "=== gpt-5.5 LCB retry DONE rc=$rc $(date) ===" >> $LOG
[ $rc -eq 0 ] && touch /root/autodl-tmp/h5_gpt55_lcb_retry_done.flag
