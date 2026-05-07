#!/bin/bash
# rational-gap GPU server bootstrap
#
# Run this on a fresh cloud GPU image (Ubuntu 22.04+ with a CUDA 12.x driver).
# After it finishes, clone the repo, run pytest to confirm the 209-test
# baseline, and start Claude in tmux.
#
# Tested target: vLLM 0.6.x on a single >=24 GB NVIDIA GPU with CUDA 12.4+.
# If you have a different stack, see notes near each `pip install`.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Configurable paths
# ---------------------------------------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# ---------------------------------------------------------------------------
# 1. System tools (incl. Python 3.11 via deadsnakes for stock Ubuntu)
# ---------------------------------------------------------------------------
echo "=== Step 1/11: system packages ==="
apt-get update
apt-get install -y --no-install-recommends \
    git tmux htop vim curl wget unzip ca-certificates \
    software-properties-common \
    build-essential

# Python 3.11 (avoid 3.12 — vLLM 0.6.x has known compatibility issues there).
if ! command -v python3.11 >/dev/null 2>&1; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev
else
    echo "python3.11 already present"
fi

# ---------------------------------------------------------------------------
# 2. GPU sanity
# ---------------------------------------------------------------------------
echo "=== Step 2/11: GPU check ==="
nvidia-smi

# ---------------------------------------------------------------------------
# 3. Python 3.11 venv
# ---------------------------------------------------------------------------
echo "=== Step 3/11: venv ==="
python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 4. Build tooling
# ---------------------------------------------------------------------------
pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 5. PyTorch (CUDA 12.4 wheel; works on CUDA 12.4–12.8 drivers).
#    If you change versions: vLLM 0.6.3/0.6.4 expects torch 2.4–2.5;
#    vLLM 0.7+ expects torch 2.5+. Mismatch => pip will error.
# ---------------------------------------------------------------------------
echo "=== Step 5/11: PyTorch ==="
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 6. vLLM. If 0.6.3 fails to resolve against torch 2.5.1, fall back to
#    `vllm==0.6.4.post1` or the latest 0.7.x.
# ---------------------------------------------------------------------------
echo "=== Step 6/11: vLLM ==="
pip install vllm==0.6.3

# ---------------------------------------------------------------------------
# 7. Project deps (everything the rational-gap pipeline imports).
#    These overlap with requirements.txt (which we'll re-apply post-clone)
#    but installing here makes the smoke test in step 11 self-contained.
# ---------------------------------------------------------------------------
echo "=== Step 7/11: project deps ==="
pip install \
    transformers==4.45.2 \
    accelerate \
    datasets \
    huggingface_hub \
    numpy \
    pyyaml \
    pytest \
    tqdm \
    pandas \
    matplotlib \
    math-verify

# Intentionally NOT installed:
#   - seaborn      (AGENT.md §5 / HANDOFF.md §9: matplotlib-only figures)
#   - human-eval   (PyPI package; we ship our own sandboxed verifier in
#                   src/verification/humaneval.py — see HANDOFF §4.2)

# ---------------------------------------------------------------------------
# 8. HuggingFace cache on the persistent volume (model weights are big).
# ---------------------------------------------------------------------------
echo "=== Step 8/11: HF cache ==="
mkdir -p "$WORKSPACE/hf_cache"
export HF_HOME="$WORKSPACE/hf_cache"
if ! grep -q "HF_HOME=$WORKSPACE/hf_cache" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$WORKSPACE/hf_cache" >> ~/.bashrc
fi

# ---------------------------------------------------------------------------
# 9. HuggingFace login (interactive — paste a token with read access).
#    Gated models (meta-llama/*) require the token's account to have
#    accepted the model's licence on huggingface.co.
# ---------------------------------------------------------------------------
echo "=== Step 9/11: HuggingFace login ==="
huggingface-cli login

# ---------------------------------------------------------------------------
# 10. Import smoke test
# ---------------------------------------------------------------------------
echo "=== Step 10/11: import smoke test ==="
python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA visible:  {torch.cuda.is_available()}')
print(f'GPU:           {torch.cuda.get_device_name(0)}')
import vllm
print(f'vLLM:          {vllm.__version__}')
import datasets, transformers, math_verify
print(f'datasets:      {datasets.__version__}')
print(f'transformers:  {transformers.__version__}')
print(f'math_verify:   {math_verify.__version__ if hasattr(math_verify, \"__version__\") else \"OK\"}')
"

# ---------------------------------------------------------------------------
# 11. End-to-end vLLM smoke test (downloads ~3 GB to \$HF_HOME).
# ---------------------------------------------------------------------------
echo "=== Step 11/11: vLLM end-to-end ==="
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-1.5B-Instruct', gpu_memory_utilization=0.5)
params = SamplingParams(n=4, temperature=1.0, seed=42, max_tokens=20)
outputs = llm.generate(['What is 2+2?'], params)
for o in outputs[0].outputs:
    print(repr(o.text))
print('vLLM end-to-end OK')
"

# ---------------------------------------------------------------------------
# Done.
# ---------------------------------------------------------------------------
cat <<EOF

=== Setup complete ===

Now do the project-side setup:

  cd "$WORKSPACE"
  git clone <your-repo-url> rational-gap-of-LLM-reasoning
  cd rational-gap-of-LLM-reasoning
  cp configs/paths.template.yaml configs/paths.yaml
  # edit configs/paths.yaml: set outputs_root to e.g. $WORKSPACE/rg_outputs

  pip install -r requirements.txt    # no-op if everything is already installed
  pytest tests/                       # MUST report: 209 passed

If pytest reports anything other than 209 passed, STOP and investigate
before running anything else (per AGENT/HANDOFF.md §2).

Then install Claude Code (see https://docs.claude.com/en/docs/claude-code
for the current install command), set ANTHROPIC_API_KEY, and:

  tmux new -s claude
  claude

Inside Claude, paste:

  Read AGENT/AGENT.md, AGENT/TODO.md, AGENT/HANDOFF.md, and
  AGENT/methodology/. Then start Phase 3.8 per HANDOFF.md §3.

EOF
