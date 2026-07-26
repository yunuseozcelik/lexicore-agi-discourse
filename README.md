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
2. **Segmentation** — time-window baseline **and** semantic segmentation (min-duration + natural break points). → `src/fetch_transcripts.py`
3. **Labeling** — speaker + claim + stance via zero-shot LLM (silver labels). → `src/label_segments.py`
3b. **Gold labeling** — human review of the silver labels (Streamlit tool) + LLM-vs-human agreement (Cohen's κ). → `src/label_manual.py`, `src/compute_agreement.py`
4. **Baseline classifiers** — TF-IDF + Logistic Regression: stance (Macro-F1 0.562) and claim detection (Macro-F1 0.782). → `src/train_baseline.py`, `src/train_claim_baseline.py`
5. **Core ML tasks** — the three graph-populating classifiers, each with a baseline + a distilled encoder:
   - **Speaker identification** (TF-IDF Macro-F1 0.693 on 5 videos, + DistilBERT). → `src/train_speaker.py`
   - **Multi-label canonical claim classification** over a 14-category taxonomy. → `src/claim_taxonomy.py`, `src/assign_claim_labels.py`, `src/train_claim_multilabel.py`
   - **NLI-style stance detection** (TF-IDF Macro-F1 0.562, DistilBERT 0.518). → `src/train_bert_stance.py`
6. **Canonical claim clustering** — group raw claims into canonical claims, two methods (TF-IDF+KMeans vs. sentence embeddings). → `src/cluster_claims.py`, `src/cluster_claims_embed.py`
7. **Graph construction** — mini person–stance graph and the unified **person–claim–stance** graph over canonical claim categories. → `src/build_graph.py`, `src/build_graph_full.py`
8. **Stance-aware retrieval evaluation** — MRR / nDCG@10 for BM25 vs. dense vs. graph/stance-aware retrieval. → `src/eval_retrieval.py`
9. **Graph validation** — LLM-as-a-judge structural-integrity audit of graph edges. → `src/graph_judge.py`

See `PROGRESS.md` for the step-by-step sprint log, ablations, and current results.

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

3. (Optional) Build a human-verified **gold** set by reviewing the silver labels:

```bash
streamlit run src/label_manual.py     # confirm/correct pre-filled labels -> data/gold/
python src/compute_agreement.py       # LLM-vs-human Cohen's kappa
```

4. Train and evaluate the baseline stance classifier:

```bash
python src/train_baseline.py
```

5. Run the three core ML tasks (each has a `--baseline` and a BERT mode; add `--cv` for 5-fold):

```bash
python src/train_speaker.py --baseline            # task 1: speaker id
python src/assign_claim_labels.py                 # bootstrap silver claim_labels
python src/train_claim_multilabel.py --baseline   # task 2: multi-label claim
python src/train_bert_stance.py --cv              # task 3: stance (DistilBERT)
```

6. Build the graph, evaluate retrieval, and audit the graph:

```bash
python src/build_graph_full.py                    # person–claim–stance graph
python src/eval_retrieval.py                      # MRR / nDCG@10 (BM25 vs dense vs stance-aware)
export OPENAI_API_KEY="..." && python src/graph_judge.py --sample 20
```

## Repo layout

```
src/    pipeline code
data/   raw transcripts, segments, labeled (silver) + gold segments, graph artifacts
```
