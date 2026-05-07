#!/bin/bash
# Bootstrap a fresh AutoDL instance for the rational-gap project.
#
# Designed for AutoDL's standard image (Ubuntu 22.04, miniconda
# pre-installed at /root/miniconda3, system Python 3.10 available).
# Run this AFTER:
#   1) git clone https://<your-repo-url> ~/rational-gap-of-LLM-reasoning
#   2) cd ~/rational-gap-of-LLM-reasoning
#   3) bash scripts/setup_autodl.sh
#
# It will:
#   - Confirm miniconda + GPU + autodl-tmp are present.
#   - Create conda env `rg-gap` (Python 3.11) — recreates if existing.
#   - Install PyTorch (cu124 wheel) → requirements.txt → requirements-gpu.txt.
#   - Configure paths.yaml with outputs_root on /root/autodl-tmp.
#   - Set HF_HOME=/root/autodl-tmp/hf_cache and HF_ENDPOINT=hf-mirror.
#   - Run pytest (must report 222 passed).
#
# Total time on a fresh instance: ~10-15 min (mostly pip downloading vLLM).

set -euo pipefail

cd ~/rational-gap-of-LLM-reasoning

if [ ! -f requirements.txt ] || [ ! -f requirements-gpu.txt ]; then
    echo "ERROR: run this from ~/rational-gap-of-LLM-reasoning"
    exit 1
fi

echo "=== 1. preflight ==="
if [ ! -d /root/miniconda3 ]; then
    echo "ERROR: miniconda not found at /root/miniconda3"
    exit 1
fi
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv | head -10
test -d /root/autodl-tmp || (echo "ERROR: /root/autodl-tmp not present"; exit 1)

source /root/miniconda3/etc/profile.d/conda.sh

echo
echo "=== 2. conda env rg-gap (Python 3.11) ==="
if conda env list | grep -q "^rg-gap "; then
    echo "rg-gap exists; skipping create"
else
    conda create -n rg-gap python=3.11 pip -y >/dev/null
fi
conda activate rg-gap
pip install --upgrade pip wheel setuptools >/dev/null

echo
echo "=== 3. PyTorch (cu124 wheel) ==="
pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

echo
echo "=== 4. project deps ==="
pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt 2>&1 | tail -5

echo
echo "=== 5. paths.yaml on autodl-tmp ==="
if [ ! -f configs/paths.yaml ]; then
    cp configs/paths.template.yaml configs/paths.yaml
fi
python - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/paths.yaml")
data = yaml.safe_load(p.read_text())
data["outputs_root"] = "/root/autodl-tmp/rg_outputs"
p.write_text(yaml.safe_dump(data, sort_keys=False))
print("outputs_root -> /root/autodl-tmp/rg_outputs")
PY

echo
echo "=== 6. HF cache + mirror in ~/.bashrc ==="
mkdir -p /root/autodl-tmp/hf_cache
for line in "export HF_HOME=/root/autodl-tmp/hf_cache" "export HF_ENDPOINT=https://hf-mirror.com"; do
    grep -qF "$line" /root/.bashrc || echo "$line" >> /root/.bashrc
done
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

echo
echo "=== 7. import smoke test ==="
python -c "
import torch, vllm, datasets, transformers, math_verify
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA visible: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}:        {torch.cuda.get_device_name(i)}')
print(f'vLLM:         {vllm.__version__}')
print(f'datasets:     {datasets.__version__}')
print(f'transformers: {transformers.__version__}')
"

echo
echo "=== 8. pytest baseline (must report 222 passed) ==="
pytest tests/ -q 2>&1 | tail -3

cat <<'EOF'

=== Setup done. To kick off the parallel H1 panel: ===

  # one model (Tülu-3-RLVR sub-panel, ~3.75 hr on 3 GPUs)
  tmux new-session -d -s h1_par \
      "cd ~/rational-gap-of-LLM-reasoning && bash scripts/run_h1_panel_parallel.sh \
       > /root/autodl-tmp/h1_par.log 2>&1"

  # OR: full 3-model panel (~11.25 hr on 3 GPUs)
  tmux new-session -d -s h1_par \
      "cd ~/rational-gap-of-LLM-reasoning && bash scripts/run_h1_panel_parallel.sh \
       tulu3-8b-rlvr qwen2.5-7b-instruct llama3.1-8b-instruct \
       > /root/autodl-tmp/h1_par.log 2>&1"

  # check progress (any time, doesn't disturb)
  tail -50 /root/autodl-tmp/h1_par.log
  ls /root/autodl-tmp/rg_outputs/results/h1/

EOF
