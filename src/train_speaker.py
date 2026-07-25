"""
Task 1 — speaker identification.

Predicts WHO is speaking in a segment from its text alone. Speakers with too few
segments and the "Unknown" bucket are dropped so the label set is learnable and
the metric is meaningful (identifying named speakers is the graph-relevant task).

Two comparable models (same 5-fold stratified protocol):
  --baseline : TF-IDF + Logistic Regression (class-weight balanced)
  (default)  : DistilBERT with class-weighted cross-entropy (student of the LLM
               speaker labels)

Reports Macro-F1 + per-speaker classification report + confusion matrix.

Usage:
  python src/train_speaker.py --baseline        # TF-IDF baseline, 5-fold
  python src/train_speaker.py --cv              # DistilBERT, 5-fold CV
  python src/train_speaker.py --min-count 8     # only speakers with >=8 segments
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
SEED = 42


def load_data(source="labeled", min_count=5):
    """Return (texts, y_ids, labels) for named speakers with >= min_count segs."""
    src = GOLD_DIR if source == "gold" else LABELED_DIR
    texts, speakers = [], []
    for f in sorted(src.glob("*.json")):
        for seg in json.loads(f.read_text(encoding="utf-8")):
            sp = (seg.get("speaker") or "").strip()
            txt = (seg.get("text") or "").strip()
            if not txt or not sp or sp.lower() == "unknown":
                continue
            texts.append(txt)
            speakers.append(sp)
    keep = {s for s, c in Counter(speakers).items() if c >= min_count}
    texts2, y2 = [], []
    for t, s in zip(texts, speakers):
        if s in keep:
            texts2.append(t)
            y2.append(s)
    labels = sorted(set(y2))
    lab2id = {l: i for i, l in enumerate(labels)}
    y = np.array([lab2id[s] for s in y2])
    return texts2, y, labels


def show(y_true, y_pred, labels):
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\nMacro-F1: {macro:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=labels,
                                digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):", labels)
    print(confusion_matrix(y_true, y_pred))


def run_baseline(texts, y, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    X = np.array(texts, dtype=object)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        y_pred[te] = clf.predict(X[te])
    print("=== TF-IDF + LogReg speaker id (5-fold) ===")
    show(y, y_pred, labels)


# --- DistilBERT with class-weighted CE (reused pattern from train_bert_stance) ---
def train_bert_fold(model_name, tokenizer, X_tr, y_tr, X_te, y_te,
                    n_labels, labels, epochs, batch, max_len, out_dir):
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, set_seed)
    set_seed(SEED)
    enc_tr = tokenizer(list(X_tr), truncation=True, padding=True, max_length=max_len)
    enc_te = tokenizer(list(X_te), truncation=True, padding=True, max_length=max_len)

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, lab):
            self.enc, self.lab = enc, lab

        def __len__(self):
            return len(self.lab)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.lab[i], dtype=torch.long)
            return item

    cw = compute_class_weight("balanced", classes=np.arange(n_labels), y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            lab = inputs.pop("labels")
            out = model(**inputs)
            w = class_weights.to(out.logits.device)
            loss = torch.nn.functional.cross_entropy(out.logits, lab, weight=w)
            return (loss, out) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels,
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)})
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        logging_strategy="no", save_strategy="no", report_to="none",
        seed=SEED, fp16=torch.cuda.is_available())
    trainer = WeightedTrainer(model=model, args=args, train_dataset=DS(enc_tr, y_tr))
    trainer.train()
    logits = trainer.predict(DS(enc_te, y_te)).predictions
    return np.argmax(logits, axis=-1)


def run_bert(texts, y, labels, args):
    import torch
    from transformers import AutoTokenizer
    X = np.array(texts, dtype=object)
    n_labels = len(labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out_dir = str(DATA_DIR / "bert_out")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"DistilBERT speaker id | device: {device} | {len(texts)} segs, {n_labels} speakers")

    if args.cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred = np.empty_like(y)
        for k, (tr, te) in enumerate(skf.split(X, y), 1):
            print(f"--- fold {k}/5 (train={len(tr)}, test={len(te)}) ---")
            y_pred[te] = train_bert_fold(args.model, tokenizer, X[tr], y[tr],
                                         X[te], y[te], n_labels, labels,
                                         args.epochs, args.batch, args.max_len, out_dir)
        print("=== DistilBERT speaker id (5-fold CV) ===")
        show(y, y_pred, labels)
    else:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2,
                                  stratify=y, random_state=SEED)
        y_pred = train_bert_fold(args.model, tokenizer, X[tr], y[tr], X[te], y[te],
                                 n_labels, labels, args.epochs, args.batch,
                                 args.max_len, out_dir)
        print("=== DistilBERT speaker id (80/20 split) ===")
        show(y[te], y_pred, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--min-count", type=int, default=5,
                    help="drop speakers with fewer than this many segments")
    ap.add_argument("--cv", action="store_true", help="5-fold CV (BERT)")
    ap.add_argument("--baseline", action="store_true", help="TF-IDF baseline instead of BERT")
    ap.add_argument("--source", choices=["labeled", "gold"], default="labeled")
    args = ap.parse_args()

    texts, y, labels = load_data(args.source, args.min_count)
    if len(texts) == 0:
        print("No named-speaker segments. Run label_segments.py first.")
        return
    print(f"Speakers ({len(labels)}): { {labels[i]: int((y == i).sum()) for i in range(len(labels))} }")

    if args.baseline:
        run_baseline(texts, y, labels)
    else:
        run_bert(texts, y, labels, args)


if __name__ == "__main__":
    main()
