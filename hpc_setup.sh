#!/usr/bin/env bash
# =============================================================================
# hpc_setup.sh — ONE-TIME prep on the TOBB HPC *login node* (tobb01).
#
# Run this once, on the login node (NOT via sbatch), after copying the repo:
#     bash hpc_setup.sh
#
# NOTE on the conda environment: the cluster user guide tells you to use a
# shared env called `env1`, but that environment does not exist on tobb01 —
# only a bare, read-only `base`. So we do what the guide's FAQ allows as the
# fallback and build our own environment under $HOME. It costs ~8 GB of the
# 30 GB quota and only needs to be built once.
#
# What this does:
#   1) creates $ENV_PREFIX with torch (CUDA 12.4), transformers, sklearn, numpy,
#   2) pre-downloads the DistilBERT checkpoint into $HOME/.cache/huggingface so
#      the compute node never needs internet access,
#   3) creates the logs/ directory the Slurm script writes into.
#
# Re-running is cheap: an existing, working environment is detected and reused.
# Pass FORCE=1 to rebuild it from scratch.
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-distilbert-base-uncased}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/lexicore}"
FORCE="${FORCE:-0}"

echo "=================================================================="
echo " LexiCore HPC setup (login node)"
echo " env   : $ENV_PREFIX"
echo " model : $MODEL"
echo "=================================================================="

module load miniconda
eval "$(conda shell.bash hook)"

mkdir -p logs gpu_results

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------- 1) env ----
if [[ "$FORCE" == "1" && -d "$ENV_PREFIX" ]]; then
  echo ">>> FORCE=1 — removing existing $ENV_PREFIX ..."
  conda env remove -y -p "$ENV_PREFIX" || rm -rf "$ENV_PREFIX"
fi

if [[ -x "$ENV_PREFIX/bin/python" ]] && "$ENV_PREFIX/bin/python" -c "import torch, transformers, sklearn" 2>/dev/null; then
  echo ">>> [1/3] environment already present and complete — reusing it."
else
  echo ">>> [1/3] building $ENV_PREFIX (a few minutes, ~8 GB) ..."
  conda create -y -p "$ENV_PREFIX" python=3.11
  conda activate "$ENV_PREFIX"

  # --no-cache-dir keeps pip's cache from eating into the 30 GB home quota.
  # CUDA 12.4 wheels to match the cluster's driver/toolkit.
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch
  pip install --no-cache-dir transformers accelerate scikit-learn numpy
fi

conda activate "$ENV_PREFIX"

# ------------------------------------------------------------- 2) check ----
echo ">>> [2/3] verifying packages ..."
python - <<'PY'
import importlib, sys

missing = []
for m in ["torch", "transformers", "sklearn", "numpy", "accelerate"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:<14} {getattr(mod, '__version__', '?')}")
    except ImportError:
        missing.append(m)
        print(f"  --  {m:<14} MISSING")

if missing:
    sys.exit("Missing: " + ", ".join(missing) + " — re-run with FORCE=1.")

import torch
print("\n  torch CUDA build :", torch.version.cuda)
print("  (the GPU itself is only visible inside a Slurm job — none seen here is normal)")
PY

# ---------------------------------------------------------- 3) prefetch ----
echo ">>> [3/3] pre-downloading $MODEL into $HF_HOME ..."
MODEL="$MODEL" python - <<'PY'
import os
from transformers import AutoTokenizer, AutoModel

name = os.environ["MODEL"]
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print("  cached:", name)
PY

echo "=================================================================="
echo " Setup OK ($(du -sh "$ENV_PREFIX" | cut -f1) env, $(du -sh "$HOME" | cut -f1) home total)."
echo " Now submit the job with:"
echo "   sbatch run_hpc.sh"
echo " and watch it with:"
echo "   squeue -u \$USER"
echo "=================================================================="
