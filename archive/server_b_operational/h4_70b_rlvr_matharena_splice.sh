#!/bin/bash
# Splice H4 reasoning-budget sweep (tulu3-70b-rlvr × MathArena × 7 L values)
# into the running h2_70b_robust.sh, AFTER the RLVR stage's matharena +
# livecodebench cells finish, BEFORE the panel's rotation deletes the
# 140GB RLVR weights. Strategy:
#
#   1. Poll the panel log for the line that signals RLVR matharena+lcb done.
#   2. Find the run_h2_panel.sh bash PID (the one inside `run_stage tulu3-70b-rlvr`).
#   3. SIGKILL it BEFORE its `rm -rf` rotation runs — RLVR weights survive.
#   4. Run 7 H4 cells sequentially at TP=4 on the cached RLVR weights.
#   5. After all 7 done, delete RLVR weights ourselves to free disk.
#   6. Touch a done flag for monitoring.
#
# Cells: tulu3-70b-rlvr × matharena × L ∈ {0, 64, 128, 256, 512, 1024, 2048}.
# K=64, M=60 (full MathArena), gpu-mem=0.7, TP=4.
# Logs go to /root/autodl-tmp/h4_70b_rlvr_matharena_splice.log + per-cell logs.

cd ~/rational-gap-of-LLM-reasoning
source /root/miniconda3/etc/profile.d/conda.sh; conda activate rg-gap
export OUTPUTS_ROOT=/root/autodl-tmp/rg_outputs
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=16

PANEL_LOG=/root/autodl-tmp/h2_70b_robust.log
LOG=/root/autodl-tmp/h4_70b_rlvr_matharena_splice.log
RLVR_DIR=/root/autodl-tmp/hf_cache/hub/models--allenai--Llama-3.1-Tulu-3-70B
RES_DIR=/root/autodl-tmp/rg_outputs/results/h2

# Match the panel's per-cell DONE log format: "[GPU 0] HH:MM:SS DONE tulu3-70b-rlvr × <ds>".
RLVR_MATHARENA_DONE='DONE tulu3-70b-rlvr × matharena'
RLVR_LCB_DONE='DONE tulu3-70b-rlvr × livecodebench'

echo "=== H4 70B-RLVR × MathArena splice start $(date) ===" > $LOG
echo "[setup] watching $PANEL_LOG for RLVR stage completion" >> $LOG

# 1) Wait for RLVR stage to actually start
while ! grep -q 'running stage tulu3-70b-rlvr' "$PANEL_LOG" 2>/dev/null; do
  echo "[wait] RLVR stage not started yet ($(date +%T))" >> $LOG
  sleep 120
done
echo "[ok] RLVR stage started ($(date))" >> $LOG

# 2) Wait for BOTH RLVR cells to log DONE in the panel log.
#    (The cells write to the panel log via run_cell's echo, before rotation.)
while ! grep -qF "$RLVR_MATHARENA_DONE" "$PANEL_LOG" 2>/dev/null \
   || ! grep -qF "$RLVR_LCB_DONE" "$PANEL_LOG" 2>/dev/null; do
  sleep 30
done
echo "[ok] both RLVR cells DONE in panel log ($(date))" >> $LOG

# 3) Kill the run_h2_panel.sh bash IMMEDIATELY to prevent rotation.
#    Race window: the panel logs "DONE" inside run_cell, then enters the
#    rotation block which logs "rotating: removing ..." then runs rm -rf.
#    If we beat the rm to it by killing the panel shell, weights survive.
PANEL_PIDS=$(pgrep -f "run_h2_panel.sh --num-gpus 1" 2>/dev/null)
echo "[kill] panel PIDs to kill: $PANEL_PIDS" >> $LOG
for pid in $PANEL_PIDS; do kill -9 "$pid" 2>&1 | tee -a $LOG; done
# Also kill any in-progress rm of the RLVR dir (defensive — should not exist
# yet if we beat the race).
pkill -9 -f "rm -rf.*models--allenai--Llama-3.1-Tulu-3-70B" 2>/dev/null
sleep 5

# 4) Sanity-check RLVR weights are still there.
if [ ! -d "$RLVR_DIR" ]; then
  echo "[FATAL] $RLVR_DIR was deleted before we could intercept; aborting." >> $LOG
  touch /root/autodl-tmp/h4_70b_rlvr_matharena_splice_FAILED.flag
  exit 1
fi
n_shards=$(ls $RLVR_DIR/snapshots/*/*.safetensors 2>/dev/null | wc -l)
echo "[ok] RLVR weights intact: $n_shards shards in $RLVR_DIR" >> $LOG

# 5) Run the 7 H4 cells sequentially at TP=4.
LS=(0 64 128 256 512 1024 2048)
SPLICE_T0=$(date +%s)
for L in "${LS[@]}"; do
  cell_log=/root/autodl-tmp/rg_outputs/logs/h4_gpu0_tulu3-70b-rlvr_matharena_L${L}.log
  echo "[cell] START tulu3-70b-rlvr × matharena × L=$L ($(date +%T))" >> $LOG
  T0=$(date +%s)
  RG_TP=4 python -m scripts.run_h4 \
    --model tulu3-70b-rlvr --dataset matharena --L "$L" --seed 0 \
    --K 64 --num-prompts 60 --gpu-mem 0.7 \
    > "$cell_log" 2>&1
  rc=$?
  echo "[cell] $([ $rc -eq 0 ] && echo DONE || echo FAIL) L=$L ($(( $(date +%s) - T0 ))s, rc=$rc, $cell_log)" >> $LOG
done
echo "[done] all 7 cells finished in $(( $(date +%s) - SPLICE_T0 ))s" >> $LOG

# 6) Clean up RLVR weights to free the ~140GB disk we held onto.
echo "[cleanup] removing RLVR weights ($RLVR_DIR)" >> $LOG
rm -rf "$RLVR_DIR"
df -h /root/autodl-tmp | tail -1 >> $LOG

# 7) Final flag
touch /root/autodl-tmp/h4_70b_rlvr_matharena_splice_done.flag
echo "=== H4 70B-RLVR × MathArena splice DONE $(date) ===" >> $LOG
