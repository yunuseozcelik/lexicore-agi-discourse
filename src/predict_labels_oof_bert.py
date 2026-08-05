"""BERT variant of predict_labels_oof.py — 5-fold OOF claim/stance predictions
with DistilBERT instead of TF-IDF, for the inductive retrieval evaluation.

Same output shape and file convention as predict_labels_oof.py, so
eval_retrieval.py --inductive consumes it unchanged. The point: the TF-IDF OOF
gives the conservative (lower-bound) inductive retrieval number; a stronger
labeller should raise it. This script produces the stronger labels so we can
measure exactly how much.

Writes data/predicted/oof_labels_bert_<taxonomy>.json (and, when --install is
passed, also overwrites oof_labels_<taxonomy>.json so eval_retrieval picks it up).

Usage:
  python src/predict_labels_oof_bert.py --install
  python src/eval_retrieval.py --inductive
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, set_seed,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
OUT_DIR = DATA_DIR / "predicted"
STANCES = ["Support", "Refute", "Neutral"]
SEED = 42


def load_segments(source="labeled", taxonomy="v1"):
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
            segs.append({"text": txt, "claim_labels": labels, "stance": s.get("stance")})
    return segs, list(CLAIM_IDS)


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels, dtype):
        self.enc = enc; self.labels = labels; self.dtype = dtype
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=self.dtype)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw); self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = self.class_weights.to(out.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w)
        return (loss, out) if return_outputs else loss


def _args(out_dir, epochs, batch):
    return TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        logging_strategy="no", save_strategy="no", report_to="none",
        seed=SEED, fp16=torch.cuda.is_available())


def predict_stance_oof(model_name, tok, texts, y, epochs, batch, max_len, out_dir, n_splits=5):
    X = np.array(texts, dtype=object); y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"  [stance] fold {k}/{n_splits}", flush=True)
        set_seed(SEED)
        enc_tr = tok(list(X[tr]), truncation=True, padding=True, max_length=max_len)
        enc_te = tok(list(X[te]), truncation=True, padding=True, max_length=max_len)
        cw = compute_class_weight("balanced", classes=np.arange(3), y=y[tr])
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        tr_ = WeightedTrainer(class_weights=torch.tensor(cw, dtype=torch.float),
                              model=model, args=_args(out_dir, epochs, batch),
                              train_dataset=DS(enc_tr, y[tr], torch.long))
        tr_.train()
        logits = tr_.predict(DS(enc_te, y[te], torch.long)).predictions
        pred[te] = [STANCES[i] for i in np.argmax(logits, axis=-1)]
        del model, tr_; torch.cuda.empty_cache()
    return pred


def predict_claims_oof(model_name, tok, texts, Y, epochs, batch, max_len, out_dir, thr=0.5, n_splits=5):
    X = np.array(texts, dtype=object)
    pred = np.zeros_like(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(kf.split(X), 1):
        print(f"  [claim ] fold {k}/{n_splits}", flush=True)
        set_seed(SEED)
        enc_tr = tok(list(X[tr]), truncation=True, padding=True, max_length=max_len)
        enc_te = tok(list(X[te]), truncation=True, padding=True, max_length=max_len)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=Y.shape[1], problem_type="multi_label_classification")
        tr_ = Trainer(model=model, args=_args(out_dir, epochs, batch),
                      train_dataset=DS(enc_tr, Y[tr].astype(np.float32), torch.float))
        tr_.train()
        logits = tr_.predict(DS(enc_te, Y[te].astype(np.float32), torch.float)).predictions
        probs = 1 / (1 + np.exp(-logits))
        pred[te] = (probs >= thr).astype(Y.dtype)
        del model, tr_; torch.cuda.empty_cache()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["labeled", "gold"], default="labeled")
    ap.add_argument("--taxonomy", choices=["v1", "v2"], default="v1")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--install", action="store_true",
                    help="also copy over oof_labels_<taxonomy>.json for eval_retrieval")
    args = ap.parse_args()

    dev = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Model {args.model} | device {dev} | taxonomy {args.taxonomy}", flush=True)
    segs, claim_ids = load_segments(args.source, args.taxonomy)
    id2col = {c: i for i, c in enumerate(claim_ids)}
    texts = [s["text"] for s in segs]
    print(f"{len(segs)} segments, {len(claim_ids)} claim categories", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    out_dir = str(DATA_DIR / "bert_out_oof")

    # ---- claims (multi-label) ----
    Y = np.zeros((len(segs), len(claim_ids)), dtype=np.int64)
    for i, s in enumerate(segs):
        for lab in s["claim_labels"]:
            if lab in id2col:
                Y[i, id2col[lab]] = 1
    print("Training claim OOF (DistilBERT, 5-fold)...", flush=True)
    P_claim = predict_claims_oof(args.model, tok, texts, Y, args.epochs, args.batch, args.max_len, out_dir)

    # ---- stance (single-label) ----
    idx = [i for i, s in enumerate(segs) if s["stance"] in STANCES]
    y_st = [STANCES.index(segs[i]["stance"]) for i in idx]
    print("Training stance OOF (DistilBERT, 5-fold)...", flush=True)
    P_stance_sub = predict_stance_oof(args.model, tok, [texts[i] for i in idx], y_st,
                                      args.epochs, args.batch, args.max_len, out_dir)
    P_stance = np.empty(len(segs), dtype=object)
    for j, i in enumerate(idx):
        P_stance[i] = P_stance_sub[j]

    for i, s in enumerate(segs):
        s["pred_claim_labels"] = [claim_ids[c] for c in np.where(P_claim[i] > 0)[0]]
        s["pred_stance"] = P_stance[i]

    stats = {
        "model": args.model,
        "claim_micro_f1": float(f1_score(Y, P_claim, average="micro", zero_division=0)),
        "claim_macro_f1": float(f1_score(Y, P_claim, average="macro", zero_division=0)),
        "stance_accuracy": float(accuracy_score([segs[i]["stance"] for i in idx],
                                                [segs[i]["pred_stance"] for i in idx])),
        "stance_macro_f1": float(f1_score([segs[i]["stance"] for i in idx],
                                          [segs[i]["pred_stance"] for i in idx],
                                          average="macro", zero_division=0)),
        "n_segments": len(segs), "taxonomy": args.taxonomy,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bert_path = OUT_DIR / f"oof_labels_bert_{args.taxonomy}.json"
    bert_path.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / f"oof_stats_bert_{args.taxonomy}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BERT OOF prediction quality ===", flush=True)
    print(f"  claim   micro-F1 {stats['claim_micro_f1']:.3f}  macro-F1 {stats['claim_macro_f1']:.3f}")
    print(f"  stance  accuracy {stats['stance_accuracy']:.3f}  macro-F1 {stats['stance_macro_f1']:.3f}")
    print(f"Wrote {bert_path}")

    if args.install:
        dest = OUT_DIR / f"oof_labels_{args.taxonomy}.json"
        shutil.copyfile(bert_path, dest)
        print(f"Installed -> {dest}  (eval_retrieval.py --inductive will use BERT labels)")


if __name__ == "__main__":
    main()
