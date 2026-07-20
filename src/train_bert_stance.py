"""
Step 6 — BERT stance classifier (silver-label distillation).

  1. Loads the labeled segments from ../data/labeled/*.json.
  2. Fine-tunes a transformer encoder (default DistilBERT) to predict stance
     (Support / Refute / Neutral) from segment text.
  3. Reports Macro-F1 on a stratified 80/20 held-out split (default), or via
     stratified 5-fold cross-validation (--cv) to match the TF-IDF baseline's
     methodology, plus a classification report and confusion matrix.

This is the "student" model of the project: the labels are zero-shot LLM
("silver") labels, so we are distilling the teacher LLM's stance decisions into
a small, fast BERT encoder. The comparison point is the TF-IDF + Logistic
Regression baseline (train_baseline.py, Macro-F1 = 0.562 on 546 segments).

Usage:
  pip install "transformers>=4.40" "torch" "scikit-learn"
  python train_bert_stance.py                 # 80/20 held-out split
  python train_bert_stance.py --cv            # 5-fold cross-validation
  python train_bert_stance.py --model bert-base-uncased --epochs 4
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
LABELS = ["Support", "Refute", "Neutral"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
SEED = 42


def load_data():
    """Return (texts, stance_ids) for every labeled segment that has a stance."""
    texts, y = [], []
    for f in sorted(LABELED_DIR.glob("*.json")):
        for seg in json.loads(f.read_text(encoding="utf-8")):
            st = seg.get("stance")
            txt = (seg.get("text") or "").strip()
            if st in LABEL2ID and txt:
                texts.append(txt)
                y.append(LABEL2ID[st])
    return texts, np.array(y)


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        # cross-entropy needs int64 labels; numpy defaults to int32 on Windows
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def macro_f1_metric(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (balanced), so the model does
    not collapse to the majority Neutral class on this imbalanced set."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def train_fold(model_name, tokenizer, X_tr, y_tr, X_te, y_te, epochs, batch, max_len, out_dir):
    """Fine-tune on one train split, return predicted label ids for X_te."""
    set_seed(SEED)
    enc_tr = tokenizer(list(X_tr), truncation=True, padding=True, max_length=max_len)
    enc_te = tokenizer(list(X_te), truncation=True, padding=True, max_length=max_len)
    ds_tr = SegmentDataset(enc_tr, y_tr)
    ds_te = SegmentDataset(enc_te, y_te)

    # balanced class weights (inverse frequency), mirroring the TF-IDF baseline
    cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS),
        id2label={i: l for l, i in LABEL2ID.items()}, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te,
        compute_metrics=macro_f1_metric,
    )
    trainer.train()
    logits = trainer.predict(ds_te).predictions
    return np.argmax(logits, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--cv", action="store_true", help="5-fold CV instead of 80/20 split")
    args = ap.parse_args()

    texts, y = load_data()
    n = len(texts)
    if n == 0:
        print("No labeled data. Run label_segments.py first.")
        return

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Dataset: {n} labeled segments  |  device: {device}  |  model: {args.model}")
    print(f"Class distribution: { {LABELS[i]: c for i, c in sorted(Counter(y).items())} }\n")

    X = np.array(texts, dtype=object)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out_dir = str(DATA_DIR / "bert_out")

    if args.cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred = np.empty_like(y)
        for k, (tr, te) in enumerate(skf.split(X, y), 1):
            print(f"--- fold {k}/5 (train={len(tr)}, test={len(te)}) ---")
            y_pred[te] = train_fold(
                args.model, tokenizer, X[tr], y[tr], X[te], y[te],
                args.epochs, args.batch, args.max_len, out_dir)
        macro = f1_score(y, y_pred, average="macro")
        print(f"\n=== 5-fold cross-validation ===\nMacro-F1: {macro:.3f}\n")
        print(classification_report(y, y_pred, target_names=LABELS, digits=3, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred) order", LABELS)
        print(confusion_matrix(y, y_pred))
    else:
        tr, te = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=SEED)
        y_pred_te = train_fold(
            args.model, tokenizer, X[tr], y[tr], X[te], y[te],
            args.epochs, args.batch, args.max_len, out_dir)
        macro = f1_score(y[te], y_pred_te, average="macro")
        print(f"\n=== Held-out 80/20 split ===")
        print(f"Train: {len(tr)}  Test: {len(te)}  Macro-F1: {macro:.3f}\n")
        print(classification_report(y[te], y_pred_te, target_names=LABELS, digits=3, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred) order", LABELS)
        print(confusion_matrix(y[te], y_pred_te))

    print("\nReference: TF-IDF + LogReg baseline Macro-F1 = 0.562 (train_baseline.py)")


if __name__ == "__main__":
    main()
