#!/usr/bin/env bash
#
# Helper: label the newly added video and refresh the knowledge graph.
#
# Usage (OpenAI):
#   export OPENAI_API_KEY="sk-..."
#   bash run_labeling.sh
#
# Usage (Gemini — free tier, no card needed):
#   export GEMINI_API_KEY="..."
#   bash run_labeling.sh
#
#   bash run_labeling.sh <video_id> ...   # or label specific video IDs
#
# The provider is picked automatically from whichever key is set. Keys are read
# from the environment, never written to disk, never committed (.env / *.key are
# gitignored).

set -euo pipefail

# --- move to the repo root (folder this script lives in) ---
cd "$(dirname "$0")"

# --- which videos to label (default: the new Yann LeCun video) ---
VIDEOS=("$@")
if [ ${#VIDEOS[@]} -eq 0 ]; then
  VIDEOS=("5t1vTLU7s40")
fi

# --- pick provider from whichever key is present ---
if [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]; then
  PROVIDER="gemini"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
  PROVIDER="openai"
else
  echo "ERROR: no API key set."
  echo "  Gemini (free):  export GEMINI_API_KEY=\"...\""
  echo "  OpenAI:         export OPENAI_API_KEY=\"sk-...\""
  exit 1
fi
echo "Provider: $PROVIDER (key hidden). Labeling videos: ${VIDEOS[*]}"

# --- set up an isolated virtualenv with the needed libraries ---
# Create the venv if missing, then always ensure deps are present (idempotent —
# pip skips anything already installed, so re-running is cheap and safe).
if [ ! -d ".venv" ]; then
  echo "[setup] creating virtualenv..."
  python3 -m venv .venv
fi
echo "[setup] ensuring dependencies are installed..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet openai google-genai scikit-learn numpy networkx matplotlib

# --- 1. label only the requested videos (resumes if interrupted) ---
# Gemini free tier allows only ~5 requests/minute, so pace calls ~13s apart.
# OpenAI has no such tight limit, so a short pause is enough there.
if [ "$PROVIDER" = "gemini" ]; then
  SLEEP=13
else
  SLEEP=0.2
fi
echo "[1/2] labeling segments (pacing ${SLEEP}s between calls)..."
.venv/bin/python src/label_segments.py --provider "$PROVIDER" --sleep "$SLEEP" --videos "${VIDEOS[@]}"

# --- 2. rebuild the knowledge graph with the new labels ---
echo "[2/2] rebuilding knowledge graph..."
.venv/bin/python src/build_graph.py

echo ""
echo "Done. Labeled files are in data/labeled/ and the graph in data/graph/."
echo "Tip: re-run the baselines to see updated scores:"
echo "  .venv/bin/python src/train_baseline.py"
echo "  .venv/bin/python src/train_claim_baseline.py"
