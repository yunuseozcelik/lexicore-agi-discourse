"""
Step 3b — baseline claim-detection classifier (TF-IDF + Logistic Regression).

Companion to train_baseline.py. Instead of stance, this predicts whether a
segment *contains a substantive claim* (the `is_claim` field the LLM produced):

  claim  = is_claim == True   (segment states an assertion)
  no-claim = is_claim == False (question / filler / off-topic)

This gives a first baseline for the multi-label claim-classification task,
which is the second of the project's three NLU components (speaker / claim /
stance). Like train_baseline.py, the labels are LLM "silver" labels, so the
score reflects how learnable they are from surface text.

  1. Loads labeled segments from ../data/labeled/*.json.
  2. Trains TF-IDF + Logistic Regression to predict claim / no-claim.
  3. Reports Macro-F1 via stratified 5-fold CV and an 80/20 split, plus a
     majority-class floor and a confusion matrix.

Usage:
  pip install scikit-learn
  python train_claim_baseline.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
LABELS = ["claim", "no-claim"]


def load_data():
    """Return (texts, labels) for every segment with an is_claim flag + text."""
    texts, labels = [], []
    for f in sorted(LABELED_DIR.glob("*.json")):
        for seg in json.loads(f.read_text(encoding="utf-8")):
            ic = seg.get("is_claim")
            txt = (seg.get("text") or "").strip()
            if ic is None or not txt:
                continue
            texts.append(txt)
            labels.append("claim" if ic else "no-claim")
    return texts, labels


def build_model():
    """TF-IDF (1-2 grams) + Logistic Regression (same recipe as stance baseline)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20000,
            sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", C=3.0)),
    ])


def main():
    texts, y = load_data()
    n = len(texts)
    if n == 0:
        print("No labeled data with is_claim. Run label_segments.py first.")
        return

    print(f"Dataset: {n} labeled segments")
    print(f"Class distribution: {dict(Counter(y))}\n")

    X = np.array(texts, dtype=object)
    y = np.array(y)

    # ---- 5-fold stratified CV ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(build_model(), X, y, cv=skf)
    macro_cv = f1_score(y, y_pred_cv, average="macro", labels=LABELS)

    print("=== 5-fold cross-validation ===")
    print(f"Macro-F1: {macro_cv:.3f}\n")
    print(classification_report(y, y_pred_cv, labels=LABELS, digits=3, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred) order", LABELS)
    print(confusion_matrix(y, y_pred_cv, labels=LABELS))

    # ---- single 80/20 split ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_model().fit(X_tr, y_tr)
    macro_split = f1_score(y_te, model.predict(X_te), average="macro", labels=LABELS)
    print(f"\n=== Held-out 80/20 split ===")
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}  Macro-F1: {macro_split:.3f}")

    # ---- majority-class reference ----
    majority = Counter(y).most_common(1)[0][0]
    maj_f1 = f1_score(y, [majority] * n, average="macro", labels=LABELS, zero_division=0)
    print(f"\nMajority-class baseline ('{majority}' always): Macro-F1 {maj_f1:.3f}")


if __name__ == "__main__":
    main()
