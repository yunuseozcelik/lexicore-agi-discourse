# AGI-Discourse

Constructing person–claim–stance semantic graphs from long-form YouTube AI-safety / AGI discourse (2022–2026).

**Group:** LexiCore (Group 1) · BIL471 Project · Project ID 1

## What this project does

We collect long-form YouTube podcast/panel discussions about AGI safety and timelines
(Lex Fridman, Dwarkesh Patel, Machine Learning Street Talk, …), pull their transcripts,
segment them, and label each segment with **speaker**, **claim**, and **stance**
(Support / Refute / Neutral). From these labels we build a time-aware
**person–claim–stance knowledge graph** and evaluate retrieval over it against BM25 and
zero-shot LLM baselines.

## Pipeline (current → planned)

1. **Transcript collection** — pull YouTube captions (no speech-to-text). → `src/fetch_transcripts.py`
2. **Segmentation** — baseline time-window. → `src/fetch_transcripts.py` (semantic LLM segmentation planned)
3. **Labeling** — speaker + claim + stance via zero-shot LLM. → `src/label_segments.py`
4. **Baseline classifier** — TF-IDF + Logistic Regression for stance, Macro-F1. → `src/train_baseline.py`
5. **Graph construction** — person–claim–stance graph with time dimension. *(planned)*
6. **Evaluation** — Macro-F1 for stance; MRR / nDCG@10 for retrieval vs. BM25 & zero-shot LLM. *(planned, BERT distillation)*

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Collect transcripts and segment them (add video IDs to `SEED_VIDEOS` first):

```bash
python src/fetch_transcripts.py
```

Saves raw transcripts to `data/raw/`, baseline segments to `data/segments/`.

2. Label segments with speaker / claim / stance (needs an API key):

```bash
export OPENAI_API_KEY="..."
python src/label_segments.py
```

Writes labeled segments to `data/labeled/`.

3. Train and evaluate the baseline stance classifier:

```bash
python src/train_baseline.py
```

## Repo layout

```
src/    pipeline code
data/   transcripts, segments, labels (gitignored — lives in Google Drive)
```
