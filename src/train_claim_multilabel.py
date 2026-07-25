"""
Task 2 — multi-label canonical claim classification.

Predicts, for each segment, WHICH canonical claim categories it expresses
(claim_taxonomy.py). A segment can carry several labels at once, so this is a
multi-label problem (sigmoid + BCE), unlike the single-label stance task.

Two models, same 5-fold protocol so they are comparable:
  --baseline : TF-IDF + One-vs-Rest Logistic Regression (the cheap baseline)
  (default)  : DistilBERT fine-tuned with multi-label BCE (the distillation
               student of the LLM's silver claim_labels)

Labels come from `claim_labels` on each segment (LLM silver, embedding bootstrap
via assign_claim_labels.py, or human gold). Reports micro-F1, macro-F1, and a
per-category F1 table.

Usage:
  python src/train_claim_multilabel.py --baseline        # TF-IDF baseline, 5-fold
  python src/train_claim_multilabel.py --cv              # DistilBERT, 5-fold CV
  python src/train_claim_multilabel.py                    # DistilBERT, 80/20 split
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, classification_report

from claim_taxonomy import CLAIM_IDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
C = len(CLAIM_IDS)
ID2COL = {cid: i for i, cid in enumerate(CLAIM_IDS)}
SEED = 42


def load_data(source="labeled"):
    """Return (texts, Y) where Y is an (n, C) multi-hot float matrix."""
    src = GOLD_DIR if source == "gold" else LABELED_DIR
    texts, rows = [], []
    for f in sorted(src.glob("*.json")):
        for seg in json.loads(f.read_text(encoding="utf-8")):
            labels = seg.get("claim_labels")
            txt = (seg.get("text") or "").strip()
            if labels is None or not txt:
                continue
            vec = np.zeros(C, dtype=np.float32)
            for lab in labels:
                if lab in ID2COL:
                    vec[ID2COL[lab]] = 1.0
            texts.append(txt)
            rows.append(vec)
    return texts, (np.vstack(rows) if rows else np.zeros((0, C), dtype=np.float32))


def report(y_true, y_pred):
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\nMicro-F1: {micro:.3f}   Macro-F1: {macro:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=CLAIM_IDS,
                                digits=3, zero_division=0))


# ---------------------------------------------------------------------------
# TF-IDF + One-vs-Rest Logistic Regression baseline
# ---------------------------------------------------------------------------
def run_baseline(texts, Y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline

    X = np.array(texts, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    y_pred = np.zeros_like(Y)
    for tr, te in kf.split(X):
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
            OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")))
        clf.fit(X[tr], Y[tr])
        y_pred[te] = clf.predict(X[te])
    print("=== TF-IDF + OvR LogReg (5-fold) ===")
    report(Y, y_pred)


# ---------------------------------------------------------------------------
# DistilBERT multi-label fine-tuning
# ---------------------------------------------------------------------------
def build_bert_dataset(tokenizer, texts, Y, max_len):
    import torch

    enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(Y)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in enc.items()}
            item["labels"] = torch.tensor(Y[i], dtype=torch.float)
            return item

    return DS()


def train_bert_fold(model_name, tokenizer, X_tr, Y_tr, X_te, Y_te,
                    epochs, batch, max_len, out_dir):
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, set_seed)
    set_seed(SEED)
    ds_tr = build_bert_dataset(tokenizer, X_tr, Y_tr, max_len)
    ds_te = build_bert_dataset(tokenizer, X_te, Y_te, max_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=C, problem_type="multi_label_classification",
        id2label={i: c for i, c in enumerate(CLAIM_IDS)},
        label2id=ID2COL)
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        logging_strategy="no", save_strategy="no", report_to="none",
        seed=SEED, fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr)
    trainer.train()
    logits = trainer.predict(ds_te).predictions
    probs = 1 / (1 + np.exp(-logits))          # sigmoid
    return (probs >= 0.5).astype(np.float32)


def run_bert(texts, Y, args):
    import torch
    from transformers import AutoTokenizer
    X = np.array(texts, dtype=object)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out_dir = str(DATA_DIR / "bert_out")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"DistilBERT multi-label | device: {device} | {len(texts)} segments, {C} labels")

    if args.cv:
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred = np.zeros_like(Y)
        for k, (tr, te) in enumerate(kf.split(X), 1):
            print(f"--- fold {k}/5 (train={len(tr)}, test={len(te)}) ---")
            y_pred[te] = train_bert_fold(args.model, tokenizer, X[tr], Y[tr],
                                         X[te], Y[te], args.epochs, args.batch,
                                         args.max_len, out_dir)
        print("=== DistilBERT multi-label (5-fold CV) ===")
        report(Y, y_pred)
    else:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED)
        y_pred = train_bert_fold(args.model, tokenizer, X[tr], Y[tr], X[te], Y[te],
                                 args.epochs, args.batch, args.max_len, out_dir)
        print("=== DistilBERT multi-label (80/20 split) ===")
        report(Y[te], y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--cv", action="store_true", help="5-fold CV (BERT)")
    ap.add_argument("--baseline", action="store_true", help="TF-IDF OvR baseline instead of BERT")
    ap.add_argument("--source", choices=["labeled", "gold"], default="labeled")
    args = ap.parse_args()

    texts, Y = load_data(args.source)
    if len(texts) == 0:
        print("No segments with claim_labels. Run label_segments.py (or "
              "assign_claim_labels.py to bootstrap) first.")
        return
    print(f"Loaded {len(texts)} segments from '{args.source}'. "
          f"Label frequency: { {CLAIM_IDS[i]: int(Y[:, i].sum()) for i in range(C)} }")

    if args.baseline:
        run_baseline(texts, Y)
    else:
        run_bert(texts, Y, args)


if __name__ == "__main__":
    main()
