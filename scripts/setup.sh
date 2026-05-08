#!/bin/bash
# Bootstrap the rational-gap experiment environment.
#
# Cluster-agnostic setup: creates a conda env, installs PyTorch (CUDA build)
# + project deps, configures paths.yaml, and runs the test suite.
#
# Tested on:
#   - AutoDL Ubuntu 22.04 instance (miniconda preinstalled at /root/miniconda3)
#   - Bare-metal Ubuntu 22.04 with miniconda installed by the user
#   - Conda 24.x with Python 3.11 base
#
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/setup.sh
#
# Environment variables (all optional):
#   CONDA_ENV       Name of the conda env to create. Default: rg-gap.
#   PYTORCH_INDEX   PyTorch wheel index URL (CUDA build).
#                   Default: https://download.pytorch.org/whl/cu124
#                   Examples:
#                     cu118: https://download.pytorch.org/whl/cu118
#                     cu121: https://download.pytorch.org/whl/cu121
#                     cu124: https://download.pytorch.org/whl/cu124
#                     cpu:   https://download.pytorch.org/whl/cpu (no GPU)
#   OUTPUTS_ROOT    Where results, sample caches, run logs live.
#                   Default: ./outputs (relative to repo root).
#   HF_HOME         Hugging Face cache root.
#                   Default: \$OUTPUTS_ROOT/hf_cache.
#   HF_ENDPOINT     HF mirror URL (e.g. https://hf-mirror.com for clusters
#                   without direct huggingface.co access). Default: unset.
#
# Total time: ~10–15 min on a fresh node (most of it pip downloading vLLM).
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-rg-gap}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$REPO_ROOT/outputs}"
HF_HOME="${HF_HOME:-$OUTPUTS_ROOT/hf_cache}"

if [ ! -f requirements.txt ] || [ ! -f requirements-gpu.txt ]; then
    echo "ERROR: run from the repo root (current dir: $(pwd))"
    exit 1
fi

echo "=== 1. preflight ==="
echo "  repo:          $REPO_ROOT"
echo "  conda env:     $CONDA_ENV"
echo "  pytorch index: $PYTORCH_INDEX"
echo "  outputs root:  $OUTPUTS_ROOT"
echo "  HF cache:      $HF_HOME"
[ -n "${HF_ENDPOINT:-}" ] && echo "  HF endpoint:   $HF_ENDPOINT"
mkdir -p "$OUTPUTS_ROOT" "$HF_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv | head -10
else
    echo "  (no nvidia-smi — will install CPU-only PyTorch unless PYTORCH_INDEX overridden)"
fi

# Locate conda.sh.
conda_sh=""
for candidate in \
    /root/miniconda3/etc/profile.d/conda.sh \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    /opt/miniconda3/etc/profile.d/conda.sh \
    /opt/conda/etc/profile.d/conda.sh; do
    if [ -f "$candidate" ]; then
        conda_sh="$candidate"
        break
    fi
done
if [ -z "$conda_sh" ]; then
    echo "ERROR: could not find conda.sh. Install miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
# shellcheck disable=SC1090
source "$conda_sh"

echo
echo "=== 2. conda env $CONDA_ENV (Python 3.11) ==="
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "  $CONDA_ENV exists; skipping create"
else
    conda create -n "$CONDA_ENV" python=3.11 pip -y >/dev/null
fi
conda activate "$CONDA_ENV"
pip install --upgrade pip wheel setuptools >/dev/null

echo
echo "=== 3. PyTorch (index: $PYTORCH_INDEX) ==="
pip install --no-cache-dir torch==2.5.1 --index-url "$PYTORCH_INDEX" 2>&1 | tail -3

echo
echo "=== 4. project deps ==="
pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt 2>&1 | tail -5

echo
echo "=== 5. configs/paths.yaml -> $OUTPUTS_ROOT ==="
if [ ! -f configs/paths.yaml ]; then
    cp configs/paths.template.yaml configs/paths.yaml
fi
python - "$OUTPUTS_ROOT" <<'PY'
import sys, yaml, pathlib
outputs_root = sys.argv[1]
p = pathlib.Path("configs/paths.yaml")
data = yaml.safe_load(p.read_text())
data["outputs_root"] = outputs_root
p.write_text(yaml.safe_dump(data, sort_keys=False))
print(f"outputs_root -> {outputs_root}")
PY

echo
echo "=== 6. HF cache export ==="
mkdir -p "$HF_HOME"
echo "  HF_HOME=$HF_HOME"
[ -n "${HF_ENDPOINT:-}" ] && echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo "  (export these in your shell before running panels; run_all.sh / panel"
echo "   scripts will read them via scripts/_common.sh)"

echo
echo "=== 7. import smoke test ==="
HF_HOME="$HF_HOME" python -c "
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
echo "=== 8. pytest baseline ==="
pytest tests/ -q 2>&1 | tail -3

cat <<EOF

=== Setup done. Next steps ===

  # full reproduction (H1 → H2 → H4 → plotting)
  tmux new-session -d -s rg \\
      "bash scripts/run_all.sh --num-gpus \$(nvidia-smi -L | wc -l) \\
       > $OUTPUTS_ROOT/logs/run_all.log 2>&1"

  # individual panels
  bash scripts/run_h1_panel.sh --num-gpus N
  bash scripts/run_h2_panel.sh --num-gpus N
  bash scripts/run_h4_panel.sh --num-gpus N

  # check progress (any time)
  tail -50 $OUTPUTS_ROOT/logs/run_all.log
  ls $OUTPUTS_ROOT/results/{h1,h2,h4}/

EOF
