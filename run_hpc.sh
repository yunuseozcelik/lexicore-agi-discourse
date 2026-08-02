#!/bin/bash
# =============================================================================
# run_hpc.sh — Slurm batch script for the TOBB HPC GPU cluster (tobb01).
#
# Same three trainings as run_gpu.sh, but adapted for the cluster:
#   - no `pip install` at job time — the environment is built once, up front, by
#     hpc_setup.sh on the login node (the guide's shared `env1` does not exist
#     on tobb01, so we use our own $HOME/envs/lexicore),
#   - HuggingFace cache pinned to $HOME so the compute node needs no internet
#     (hpc_setup.sh fills that cache too),
#   - single GPU, as the cluster guide requires.
#
# Submit from the repo root on tobb01:
#     sbatch run_hpc.sh
#
# Tunables (pass through sbatch --export):
#     sbatch --export=ALL,BATCH=32 run_hpc.sh
# =============================================================================
#SBATCH --partition=compute            # debug = 15 min max, compute = 12 h max
#SBATCH --job-name=lexicore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1                   # single GPU — cluster policy
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00                # dataset is small; well under the 12 h cap

set -euo pipefail

MODEL="${MODEL:-distilbert-base-uncased}"
BATCH="${BATCH:-16}"                   # 16 is deliberate: 64 underfits speaker/claim
MAXLEN="${MAXLEN:-256}"
OUT="${OUT:-gpu_results}"              # override to keep an ablation's results separate
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/lexicore}"

# Slurm starts us in the submission directory, but be explicit about it.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$OUT" logs

module load miniconda
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

# Use the cache filled by hpc_setup.sh on the login node; never hit the network.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=================================================================="
echo " LexiCore HPC run"
echo " job / node   : ${SLURM_JOB_ID:-?} on $(hostname)"
echo " encoder      : $MODEL"
echo " batch / len  : $BATCH / $MAXLEN"
echo " results dir  : $OUT/"
echo " started      : $(date)"
echo "=================================================================="

# 0) sanity: did Slurm actually give us a GPU?
python - <<'PY'
import sys, torch
print("CUDA available :", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: no GPU visible inside the job — check --gres=gpu:1.")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print("GPU            :", p.name, f"({p.total_memory/1e9:.0f} GB)")
PY

# 1) Task 1 — speaker identification (25 speakers, 5-fold CV)
echo ">>> [1/3] speaker DistilBERT (5-fold) ..."
python src/train_speaker.py --model "$MODEL" --batch "$BATCH" --max-len "$MAXLEN" --cv 2>&1 | tee "$OUT/speaker_bert.txt"

# 2) Task 2 — multi-label canonical claim classification (14 categories, 5-fold)
echo ">>> [2/3] multi-label claim DistilBERT (5-fold) ..."
python src/train_claim_multilabel.py --model "$MODEL" --batch "$BATCH" --max-len "$MAXLEN" --cv 2>&1 | tee "$OUT/claim_bert.txt"

# 3) Task 3 — NLI-style stance detection (Support/Refute/Neutral, 5-fold)
echo ">>> [3/3] stance DistilBERT (5-fold) ..."
python src/train_bert_stance.py --model "$MODEL" --batch "$BATCH" --max-len "$MAXLEN" --cv 2>&1 | tee "$OUT/stance_bert.txt"

echo "=================================================================="
echo " DONE — all three encoders trained."
echo " Macro-F1 / Micro-F1 lines are in:"
echo "   $OUT/speaker_bert.txt   (Task 1)"
echo "   $OUT/claim_bert.txt     (Task 2)"
echo "   $OUT/stance_bert.txt    (Task 3)"
echo " finished     : $(date)"
echo "=================================================================="
