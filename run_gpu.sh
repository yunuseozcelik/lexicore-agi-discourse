#!/usr/bin/env bash
# =============================================================================
# run_gpu.sh — ONE-SHOT GPU training run for LexiCore AGI-Discourse.
#
# Purpose: use a GPU exactly once, end to end. This fine-tunes all three
# DistilBERT "student" encoders that the project spec requires (distil
# zero-shot LLM silver labels -> efficient encoders) and writes every metric
# to gpu_results/. No API key and no paid service is needed — everything here
# is local GPU compute. The LLM labeling and LLM-as-a-judge steps (the only
# paid/API parts) are ALREADY DONE and committed; do not re-run them.
#
# Run on any CUDA machine (Colab, local GPU, Linux, Windows Git Bash):
#     bash run_gpu.sh
#
# Tunable via environment variables (defaults are safe for ~8GB GPUs):
#     MODEL   encoder checkpoint     (default: distilbert-base-uncased)
#     BATCH   per-device batch size  (default: 16)
#     MAXLEN  max token length       (default: 256)
#
# Examples:
#     BATCH=64 bash run_gpu.sh                 # A100 / big GPU — faster
#     BATCH=8 MAXLEN=192 bash run_gpu.sh       # 4GB laptop GPU — avoid OOM
#     MODEL=bert-base-uncased BATCH=32 bash run_gpu.sh
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-distilbert-base-uncased}"
BATCH="${BATCH:-16}"
MAXLEN="${MAXLEN:-256}"
OUT="gpu_results"
mkdir -p "$OUT"

echo "=================================================================="
echo " LexiCore GPU run"
echo " encoder      : $MODEL"
echo " batch / len  : $BATCH / $MAXLEN"
echo " results dir  : $OUT/"
echo " started      : $(date)"
echo "=================================================================="

# 0) sanity: is a CUDA GPU visible?
python - <<'PY'
import torch
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU            :", p.name, f"({p.total_memory/1e9:.0f} GB)")
else:
    print("WARNING: no CUDA GPU seen — this will run on CPU and be very slow.")
PY

# 1) dependencies
echo ">>> installing requirements ..."
pip install -q -r requirements.txt

# 2) Task 1 — speaker identification (25 speakers, 5-fold CV)
echo ">>> [1/3] speaker DistilBERT (5-fold) ..."
python src/train_speaker.py --model "$MODEL" --batch "$BATCH" --max-len "$MAXLEN" --cv 2>&1 | tee "$OUT/speaker_bert.txt"

# 3) Task 2 — multi-label canonical claim classification (14 categories, 5-fold)
echo ">>> [2/3] multi-label claim DistilBERT (5-fold) ..."
python src/train_claim_multilabel.py --model "$MODEL" --batch "$BATCH" --max-len "$MAXLEN" --cv 2>&1 | tee "$OUT/claim_bert.txt"

# 4) Task 3 — NLI-style stance detection (Support/Refute/Neutral, 5-fold)
echo ">>> [3/3] stance DistilBERT (5-fold) ..."
python src/train_bert_stance.py --model "$MODEL" --batch "$BATCH" --max-len "$MAXLEN" --cv 2>&1 | tee "$OUT/stance_bert.txt"

echo "=================================================================="
echo " DONE — all three encoders trained."
echo " Grab the Macro-F1 / Micro-F1 lines from:"
echo "   $OUT/speaker_bert.txt   (Task 1)"
echo "   $OUT/claim_bert.txt     (Task 2)"
echo "   $OUT/stance_bert.txt    (Task 3)"
echo " finished     : $(date)"
echo "=================================================================="
