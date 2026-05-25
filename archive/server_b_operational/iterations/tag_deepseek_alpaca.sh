#!/bin/bash
R=/root/rational_gap_outputs/results/h1
for i in $(seq 1 180); do
  if [ -f $R/qwen2.5-72b-instruct_alpaca_eval_seed0.json ]; then
    mv $R/qwen2.5-72b-instruct_alpaca_eval_seed0.json $R/qwen2.5-72b-instruct_alpaca_eval_judge-deepseek-v4-flash_seed0.json
    echo "$(date) renamed alpaca deepseek -> tagged" >> /root/autodl-tmp/tag_deepseek_alpaca.log
    break
  fi
  sleep 60
done
