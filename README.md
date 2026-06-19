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
2. **Segmentation** — baseline time-window now; semantic (LLM) segmentation planned.
3. **Labeling** — speaker + claim + stance via LLM (zero-shot), then distilled into a BERT encoder. *(planned)*
4. **Graph construction** — person–claim–stance graph with time dimension. *(planned)*
5. **Evaluation** — Macro-F1 for stance; MRR / nDCG@10 for retrieval vs. BM25 & zero-shot LLM. *(planned)*

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Add YouTube video IDs to `SEED_VIDEOS` in `src/fetch_transcripts.py`
   (the 11-char string after `v=` in a watch URL).
2. Run:

```bash
python src/fetch_transcripts.py
```

This saves raw transcripts to `data/raw/`, baseline segments to `data/segments/`,
and prints dataset stats (videos / segments / hours) for the progress report.

## Repo layout

```
src/         pipeline code
notebooks/   exploration / labeling experiments
data/        transcripts & segments (gitignored — lives in Google Drive)
docs/        report drafts, notes
```
