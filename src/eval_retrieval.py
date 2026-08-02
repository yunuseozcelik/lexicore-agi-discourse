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


def _load_taxonomy_v2():
    """Swap the module-level taxonomy for the collapsed 8-category v2."""
    from claim_taxonomy_v2 import CLAIM_IDS_V2, ID2NAME_V2, ID2DESC_V2, collapse_labels
    return {"CLAIM_IDS": list(CLAIM_IDS_V2), "ID2NAME": ID2NAME_V2,
            "ID2DESC": ID2DESC_V2, "_COLLAPSE": collapse_labels}


_COLLAPSE = None


def load_corpus(source):
    src = GOLD_DIR if source == "gold" else LABELED_DIR
    docs = []
    for f in sorted(src.glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            txt = (s.get("text") or "").strip()
            if not txt:
                continue
            labels = s.get("claim_labels") or []
            if _COLLAPSE is not None:
                labels = _COLLAPSE(labels)
            docs.append({
                "text": txt,
                "stance": s.get("stance"),
                "claim_labels": labels,
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


def stance_boost(docs, queries, field_claim="claim_labels", field_stance="stance"):
    """Structural signal: +1 if a doc matches the query's claim+stance, else 0.

    With the default (gold) fields this is an ORACLE boost: the qrels in
    build_queries() are defined from those same fields, so the boost reproduces
    the answer key and the metrics saturate at 1.000. Pass the predicted fields
    (`pred_claim_labels` / `pred_stance`) for the honest, inductive version.
    """
    B = np.zeros((len(queries), len(docs)))
    for qi, q in enumerate(queries):
        for di, d in enumerate(docs):
            if q["claim"] in (d.get(field_claim) or []) and d.get(field_stance) == q["stance"]:
                B[qi, di] = 1.0
    return B


def attach_predictions(docs, taxonomy):
    """Merge OOF predicted labels (predict_labels_oof.py) onto the corpus.

    Matches on segment text, which is what both loaders key off. Returns the
    number of docs that received predictions.
    """
    path = DATA_DIR / "predicted" / f"oof_labels_{taxonomy}.json"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}.\nRun first:  python src/predict_labels_oof.py"
            f"{' --taxonomy v2' if taxonomy == 'v2' else ''}")
    by_text = {s["text"]: s for s in json.loads(path.read_text(encoding="utf-8"))}
    n = 0
    for d in docs:
        p = by_text.get(d["text"])
        if p is not None:
            d["pred_claim_labels"] = p.get("pred_claim_labels") or []
            d["pred_stance"] = p.get("pred_stance")
            n += 1
        else:
            d["pred_claim_labels"] = []
            d["pred_stance"] = None
    return n


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
    ap.add_argument("--taxonomy", choices=["v1", "v2"], default="v1")
    ap.add_argument("--inductive", action="store_true",
                    help="add stance_aware_inductive, re-ranked with OOF PREDICTED "
                         "labels instead of gold (the honest version)")
    args = ap.parse_args()

    if args.taxonomy == "v2":
        globals().update(_load_taxonomy_v2())

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
        # stance_aware (ORACLE): boost comes from the same gold fields that
        # define the qrels -> saturates at 1.000. Kept as the graph's ceiling.
        results["stance_aware_oracle"] = _minmax(dense) + 2.0 * stance_boost(docs, queries)

        if args.inductive:
            n = attach_predictions(docs, args.taxonomy)
            print(f"[inductive] OOF predictions attached to {n}/{len(docs)} segments\n")
            results["stance_aware_inductive"] = _minmax(dense) + 2.0 * stance_boost(
                docs, queries, "pred_claim_labels", "pred_stance")
    except ImportError:
        print("[warn] sentence-transformers not installed — only BM25 evaluated.\n")

    print(f"{'Method':26} {'MRR':>8} {'nDCG@'+str(args.k):>9}")
    print("-" * 45)
    for name, sc in results.items():
        mrr, ndcg = evaluate(sc, queries, len(docs), k=args.k)
        print(f"{name:26} {mrr:>8.3f} {ndcg:>9.3f}")

    print("\nNotes:")
    print(" * stance_aware_oracle is NOT a valid comparison: the qrels are defined")
    print("   from the gold claim/stance fields and the boost reads those same")
    print("   fields, so it reproduces the answer key. Read it as the graph's")
    print("   CEILING — what stance-aware re-ranking would be worth if the")
    print("   classifiers were perfect.")
    if args.inductive:
        print(" * stance_aware_inductive is the honest number: the same re-ranking")
        print("   driven by 5-fold out-of-fold PREDICTED labels, so no segment is")
        print("   scored by a model that saw it. The gap to the oracle is the cost")
        print("   of classifier error; the gap to BM25/dense is the graph's real")
        print("   contribution.")
    else:
        print(" * Add --inductive for the leakage-free version (needs")
        print("   predict_labels_oof.py to have been run).")


if __name__ == "__main__":
    main()
