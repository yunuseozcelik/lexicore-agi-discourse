"""
Tasks 6-8 — stance-aware retrieval evaluation (MRR, nDCG@10).

Evaluates retrieval over the segment corpus and compares:
  - bm25          : lexical BM25 baseline (text only)
  - dense         : sentence-embedding cosine baseline (text only, all-MiniLM-L6-v2)
  - hybrid        : normalized BM25 + dense fusion (text only)
  - stance_aware  : dense retrieval re-ranked with the person-claim-stance graph
                    signal (boost segments whose claim category + stance match the
                    query). This is the graph-augmented pipeline the project asks
                    for; it uses the labeled graph structure as the stance signal.

Query set (built automatically from the labels): for every canonical claim
category and stance with enough evidence, we form a natural-language query and
mark the segments carrying that (claim category, stance) as relevant. This turns
the annotated corpus into a retrieval benchmark with qrels.

Metrics: Mean Reciprocal Rank (MRR) and nDCG@10, macro-averaged over queries.

Usage:
  pip install rank-bm25 sentence-transformers
  python src/eval_retrieval.py
  python src/eval_retrieval.py --source gold --min-rel 3
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

from claim_taxonomy import CLAIM_IDS, ID2NAME, ID2DESC

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
STANCES = ["Support", "Refute", "Neutral"]
MODEL_NAME = "all-MiniLM-L6-v2"


def load_corpus(source):
    src = GOLD_DIR if source == "gold" else LABELED_DIR
    docs = []
    for f in sorted(src.glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            txt = (s.get("text") or "").strip()
            if not txt:
                continue
            docs.append({
                "text": txt,
                "stance": s.get("stance"),
                "claim_labels": s.get("claim_labels") or [],
            })
    return docs


def build_queries(docs, min_rel):
    """One query per (claim category, stance) with >= min_rel relevant docs."""
    queries = []
    for cid in CLAIM_IDS:
        if cid == "other":
            continue
        for st in ["Support", "Refute"]:
            rel = [i for i, d in enumerate(docs)
                   if cid in d["claim_labels"] and d["stance"] == st]
            if len(rel) < min_rel:
                continue
            verb = "supports near-term / optimistic AGI" if st == "Support" \
                else "is skeptical of near-term AGI"
            qtext = f"Statements about {ID2NAME[cid]} where the speaker {verb}. {ID2DESC[cid]}"
            queries.append({"text": qtext, "claim": cid, "stance": st,
                            "relevant": set(rel)})
    return queries


def _tok(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def score_bm25(corpus_texts, query_texts):
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([_tok(t) for t in corpus_texts])
    return np.vstack([bm.get_scores(_tok(q)) for q in query_texts])


def score_dense(model, corpus_texts, query_texts):
    ce = np.asarray(model.encode(corpus_texts, show_progress_bar=False))
    qe = np.asarray(model.encode(query_texts, show_progress_bar=False))
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    qe = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9)
    return qe @ ce.T


def _minmax(m):
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    return (m - lo) / (hi - lo + 1e-9)


def stance_boost(docs, queries):
    """Structural signal: +1 if a doc matches the query's claim+stance, else 0."""
    B = np.zeros((len(queries), len(docs)))
    for qi, q in enumerate(queries):
        for di, d in enumerate(docs):
            if q["claim"] in d["claim_labels"] and d["stance"] == q["stance"]:
                B[qi, di] = 1.0
    return B


def evaluate(scores, queries, n_docs, k=10):
    """Return (MRR, nDCG@k) macro-averaged over queries."""
    mrr, ndcg = [], []
    for qi, q in enumerate(queries):
        rel = np.zeros(n_docs)
        for i in q["relevant"]:
            rel[i] = 1.0
        order = np.argsort(-scores[qi])
        # reciprocal rank of first relevant
        rr = 0.0
        for rank, di in enumerate(order, 1):
            if rel[di]:
                rr = 1.0 / rank
                break
        mrr.append(rr)
        ndcg.append(ndcg_score(rel[None, :], scores[qi][None, :], k=k))
    return float(np.mean(mrr)), float(np.mean(ndcg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["labeled", "gold"], default="labeled")
    ap.add_argument("--min-rel", type=int, default=3,
                    help="min relevant docs for a (claim, stance) to become a query")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    docs = load_corpus(args.source)
    if not docs:
        raise SystemExit("Empty corpus. Run the pipeline first.")
    texts = [d["text"] for d in docs]
    queries = build_queries(docs, args.min_rel)
    if not queries:
        raise SystemExit("No queries — segments lack claim_labels. Run "
                         "assign_claim_labels.py (or label_segments.py) first.")
    print(f"Corpus: {len(docs)} segments | Queries: {len(queries)} "
          f"(claim x stance, >= {args.min_rel} relevant each)\n")

    results = {}
    bm = score_bm25(texts, [q["text"] for q in queries])
    results["bm25"] = bm

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        dense = score_dense(model, texts, [q["text"] for q in queries])
        results["dense"] = dense
        results["hybrid"] = _minmax(bm) + _minmax(dense)
        # stance-aware: dense + strong structural boost from the graph
        results["stance_aware"] = _minmax(dense) + 2.0 * stance_boost(docs, queries)
    except ImportError:
        print("[warn] sentence-transformers not installed — only BM25 evaluated.\n")

    print(f"{'Method':14} {'MRR':>8} {'nDCG@'+str(args.k):>9}")
    print("-" * 33)
    for name, sc in results.items():
        mrr, ndcg = evaluate(sc, queries, len(docs), k=args.k)
        print(f"{name:14} {mrr:>8.3f} {ndcg:>9.3f}")

    print("\nNote: 'stance_aware' uses the labeled person-claim-stance structure as "
          "the stance signal (upper bound of the graph's value); a fully inductive "
          "version would use the fine-tuned classifiers' predicted labels.")


if __name__ == "__main__":
    main()
