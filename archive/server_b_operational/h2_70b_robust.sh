#!/bin/bash
# H2-70B trajectory on matharena + livecodebench, with ROBUST pre-download per
# stage. The repo panel calls run_h2 which lets vLLM auto-download, but that
# has no retry -> a network drop kills the load. We pre-download each stage
# fully with huggingface-cli (retry loop), THEN invoke the panel, which finds
# the model cached. Same repo panel, just robust against download flakiness.
cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=16
LOG=/root/autodl-tmp/h2_70b_robust.log
echo "=== H2-70B robust start $(date) ===" > $LOG

predownload() {  # $1 = hf_id, $2 = snap dir name (models--org--name)
  local hf_id=$1 snap=/root/autodl-tmp/hf_cache/hub/$2
  for a in $(seq 1 30); do
    huggingface-cli download $hf_id --include "*.safetensors" "*.json" "*.txt" "tokenizer*" >> $LOG 2>&1
    local inc=$(ls $snap/blobs/*.incomplete 2>/dev/null | wc -l)
    local n=$(ls $snap/snapshots/*/*.safetensors 2>/dev/null | wc -l)
    echo "[$(date +%T)] $hf_id attempt $a: shards=$n incomplete=$inc" >> $LOG
    [ "$inc" -eq 0 ] && [ "$n" -ge 1 ] && { echo "$hf_id DOWNLOAD COMPLETE" >> $LOG; return 0; }
    sleep 15
  done
  echo "$hf_id DOWNLOAD FAILED after 30 attempts" >> $LOG; return 1
}

run_stage() {  # $1 = stage alias (tulu3-70b-{sft,dpo,rlvr}), $2 = hf_id, $3 = snap
  predownload "$2" "$3" || return 1
  echo "=== running stage $1 via panel $(date) ===" >> $LOG
  RG_TP=4 STAGES="$1" DATASETS="matharena livecodebench" SEED=0 ROTATE=1 \
    bash scripts/run_h2_panel.sh --num-gpus 1 >> $LOG 2>&1
  echo "=== stage $1 done $(date) ===" >> $LOG
}

run_stage tulu3-70b-sft  allenai/Llama-3.1-Tulu-3-70B-SFT  models--allenai--Llama-3.1-Tulu-3-70B-SFT
run_stage tulu3-70b-dpo  allenai/Llama-3.1-Tulu-3-70B-DPO  models--allenai--Llama-3.1-Tulu-3-70B-DPO
run_stage tulu3-70b-rlvr allenai/Llama-3.1-Tulu-3-70B      models--allenai--Llama-3.1-Tulu-3-70B
echo "=== H2-70B ROBUST DONE $(date) ===" >> $LOG
touch /root/autodl-tmp/h2_70b_robust_done.flag
