#!/bin/bash
# Retry the GPT-5.5 H5 livecodebench cells on server B.
# Migrated from A so A can be released (API-only — no GPU contention with
# the parallel H2-70B run on B). Resume cache pulled in from A.
#
# NOTE: B's configs/paths.yaml hardcodes outputs_root: ~/rational_gap_outputs
# and paths.py ignores OUTPUTS_ROOT env, so we don't override OUTPUTS_ROOT.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8     # leave cycles for the H2-70B vllm workers
[ -f /root/.config/rg-gap.env ] && { set -a; source /root/.config/rg-gap.env; set +a; }
LOG=/root/autodl-tmp/h5_gpt55_lcb_retry_B.log
echo "=== gpt-5.5 LCB retry on B start $(date) ===" > $LOG
MODELS="gpt-5.5" DATASETS="livecodebench" TAUS="0.7 1.0" HYPS="h1 h3" CONC=5 \
  bash scripts/run_h5_panel.sh >> $LOG 2>&1
rc=$?
echo "=== gpt-5.5 LCB retry on B DONE rc=$rc $(date) ===" >> $LOG
[ $rc -eq 0 ] && touch /root/autodl-tmp/h5_gpt55_lcb_retry_B_done.flag
