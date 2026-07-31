"""
Out-of-fold (OOF) label prediction for the inductive retrieval evaluation.

WHY
---
`eval_retrieval.py`'s `stance_aware` method scored a perfect MRR/nDCG of 1.000.
That is not a result — it is leakage. The relevance judgements (qrels) are
defined as "segments whose gold claim_labels contain C and whose gold stance is
S", and the re-ranking boost was computed from those *same* gold fields. The
method was reading its own answer key.

This module removes the leak by producing **predicted** claim_labels and stance
for every segment, using 5-fold cross-validation so that each segment's labels
come from a model that never saw it during training. Feeding those predictions
to the boost makes the evaluation inductive: it measures what the graph is worth
when built by the classifiers, not when handed the answers.

The predictors mirror the `--baseline` models in train_claim_multilabel.py and
train_baseline.py (TF-IDF + logistic regression), so this runs on CPU in about a
minute and needs no GPU. The DistilBERT/BERT-base encoders score higher (see
PROGRESS.md section 14) and would raise the inductive numbers; swapping them in
is a drop-in change to `predict_*_oof` if a GPU is available.

Usage:
  python src/predict_labels_oof.py                    # writes data/predicted/oof_labels.json
  python src/predict_labels_oof.py --taxonomy v2      # collapsed 8-category version
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
OUT_DIR = DATA_DIR / "predicted"
STANCES = ["Support", "Refute", "Neutral"]
SEED = 42


def load_segments(source="labeled", taxonomy="v1"):
    """Load segments with their gold claim_labels and stance."""
    if taxonomy == "v2":
        from claim_taxonomy_v2 import CLAIM_IDS_V2 as CLAIM_IDS, collapse_labels
    else:
        from claim_taxonomy import CLAIM_IDS
        collapse_labels = None

    src = GOLD_DIR if source == "gold" else LABELED_DIR
    segs = []
    for f in sorted(src.glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            txt = (s.get("text") or "").strip()
            if not txt:
                continue
            labels = s.get("claim_labels") or []
            if collapse_labels is not None:
                labels = collapse_labels(labels)
            segs.append({"text": txt, "claim_labels": labels,
                         "stance": s.get("stance")})
    return segs, list(CLAIM_IDS)


def _tfidf_ovr():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
        OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")))


def _tfidf_clf():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
        LogisticRegression(max_iter=1000, class_weight="balanced"))


def predict_claims_oof(texts, Y, n_splits=5):
    """5-fold OOF multi-label claim predictions (multi-hot, same shape as Y)."""
    X = np.array(texts, dtype=object)
    pred = np.zeros_like(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        clf = _tfidf_ovr()
        clf.fit(X[tr], Y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def predict_stance_oof(texts, y, n_splits=5):
    """5-fold OOF stance predictions (array of stance strings)."""
    X = np.array(texts, dtype=object)
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        clf = _tfidf_clf()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def build_oof(source="labeled", taxonomy="v1"):
    """Return the segment list with `pred_claim_labels` and `pred_stance` added."""
    segs, claim_ids = load_segments(source, taxonomy)
    id2col = {c: i for i, c in enumerate(claim_ids)}
    texts = [s["text"] for s in segs]

    # ---- claims (multi-label) ----
    Y = np.zeros((len(segs), len(claim_ids)), dtype=np.float32)
    for i, s in enumerate(segs):
        for lab in s["claim_labels"]:
            if lab in id2col:
                Y[i, id2col[lab]] = 1.0
    P_claim = predict_claims_oof(texts, Y)

    # ---- stance (single-label); only segments that actually carry one ----
    idx = [i for i, s in enumerate(segs) if s["stance"] in STANCES]
    P_stance = np.empty(len(segs), dtype=object)
    if idx:
        pred = predict_stance_oof([texts[i] for i in idx],
                                  [segs[i]["stance"] for i in idx])
        for j, i in enumerate(idx):
            P_stance[i] = pred[j]

    for i, s in enumerate(segs):
        s["pred_claim_labels"] = [claim_ids[c] for c in np.where(P_claim[i] > 0)[0]]
        s["pred_stance"] = P_stance[i]

    # ---- how far the predictions are from gold (context for the eval) ----
    from sklearn.metrics import f1_score, accuracy_score
    stats = {
        "claim_micro_f1": float(f1_score(Y, P_claim, average="micro", zero_division=0)),
        "claim_macro_f1": float(f1_score(Y, P_claim, average="macro", zero_division=0)),
        "stance_accuracy": float(accuracy_score(
            [segs[i]["stance"] for i in idx], [segs[i]["pred_stance"] for i in idx])) if idx else None,
        "stance_macro_f1": float(f1_score(
            [segs[i]["stance"] for i in idx], [segs[i]["pred_stance"] for i in idx],
            average="macro", zero_division=0)) if idx else None,
        "n_segments": len(segs),
        "taxonomy": taxonomy,
    }
    return segs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["labeled", "gold"], default="labeled")
    ap.add_argument("--taxonomy", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()

    segs, stats = build_oof(args.source, args.taxonomy)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"oof_labels_{args.taxonomy}.json"
    out.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Taxonomy {args.taxonomy} | {stats['n_segments']} segments")
    print(f"  OOF claim   micro-F1 {stats['claim_micro_f1']:.3f}  "
          f"macro-F1 {stats['claim_macro_f1']:.3f}")
    if stats["stance_accuracy"] is not None:
        print(f"  OOF stance  accuracy {stats['stance_accuracy']:.3f}  "
              f"macro-F1 {stats['stance_macro_f1']:.3f}")
    print(f"\nWrote {out}")
    print("Now run:  python src/eval_retrieval.py --inductive")


if __name__ == "__main__":
    main()
