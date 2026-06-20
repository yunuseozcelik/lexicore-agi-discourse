"""
Step 3 — baseline stance classifier (TF-IDF + Logistic Regression).

  1. Loads the labeled segments from ../data/labeled/*.json.
  2. Trains a TF-IDF + Logistic Regression model to predict stance
     (Support / Refute / Neutral) from segment text.
  3. Reports Macro-F1 via stratified 5-fold cross-validation (robust on a
     small set) and a single 80/20 split, plus a confusion matrix.

A simple reference baseline. The labels are zero-shot LLM ("silver") labels,
so the score reflects how learnable those labels are from surface text rather
than ground-truth accuracy; a hand-checked gold subset is the proper test set.
A distilled BERT encoder is expected to improve on this.

Usage:
  pip install scikit-learn
  python train_baseline.py
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
LABELS = ["Support", "Refute", "Neutral"]


def load_data():
    """Return (texts, stances) for every labeled segment that has a stance."""
    texts, stances = [], []
    for f in sorted(LABELED_DIR.glob("*.json")):
        for seg in json.loads(f.read_text(encoding="utf-8")):
            st = seg.get("stance")
            txt = (seg.get("text") or "").strip()
            if st in LABELS and txt:
                texts.append(txt)
                stances.append(st)
    return texts, stances


def build_model():
    """TF-IDF (1-2 grams) + multinomial Logistic Regression."""
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
        print("No labeled data. Run label_segments.py first.")
        return

    print(f"Dataset: {n} labeled segments")
    print(f"Class distribution: {dict(Counter(y))}\n")

    X = np.array(texts, dtype=object)
    y = np.array(y)

    # ---- 5-fold stratified CV (robust estimate on a small set) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(build_model(), X, y, cv=skf)
    macro_cv = f1_score(y, y_pred_cv, average="macro", labels=LABELS)

    print("=== 5-fold cross-validation ===")
    print(f"Macro-F1: {macro_cv:.3f}\n")
    print(classification_report(y, y_pred_cv, labels=LABELS, digits=3, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred) order", LABELS)
    print(confusion_matrix(y, y_pred_cv, labels=LABELS))

    # ---- single 80/20 split (clean held-out check) ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_model().fit(X_tr, y_tr)
    macro_split = f1_score(y_te, model.predict(X_te), average="macro", labels=LABELS)
    print(f"\n=== Held-out 80/20 split ===")
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}  Macro-F1: {macro_split:.3f}")

    # ---- majority-class reference (sanity floor) ----
    majority = Counter(y).most_common(1)[0][0]
    maj_f1 = f1_score(y, [majority] * n, average="macro", labels=LABELS, zero_division=0)
    print(f"\nMajority-class baseline ('{majority}' always): Macro-F1 {maj_f1:.3f}")


if __name__ == "__main__":
    main()
