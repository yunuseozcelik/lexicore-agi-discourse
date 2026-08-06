"""Honest / leakage-controlled evaluation of the BERT stance model.

The TF-IDF honest_eval.py reports stance Macro-F1 under three protocols; the
project guide (section 3.7) notes the same protocols were never reported for
BERT. This script closes that gap by running the BERT stance fine-tuner under:

  (a) segment 5-fold    : StratifiedKFold over segments (silver-vs-silver).
  (b) video GroupKFold  : a whole video held out each fold (no same-video leak).
  (c) gold held-out     : train on silver EXCLUDING human-reviewed segments,
                          test against the human gold stance.

Claim-bearing segments only (is_claim=True), matching the corrected TF-IDF eval:
non-claims were all forced to Neutral and would inflate the score.

Usage:
  python src/honest_eval_bert.py                       # distilbert, 4 epochs
  python src/honest_eval_bert.py --model bert-base-uncased --epochs 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bert_stance import train_fold, LABELS, LABEL2ID

DATA = Path(__file__).resolve().parent.parent / "data"


def rnd(x):
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return x


def load_rows():
    rows = []
    for f in sorted((DATA / "labeled").glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            st, txt = s.get("stance"), (s.get("text") or "").strip()
            if st in LABEL2ID and txt and s.get("is_claim"):
                rows.append({"text": txt, "y": LABEL2ID[st],
                             "vid": f.stem, "key": (f.stem, rnd(s.get("start")))})
    return rows


def load_gold():
    g = {}
    for f in sorted((DATA / "gold").glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("reviewed") and s.get("is_claim"):
                st, txt = s.get("stance"), (s.get("text") or "").strip()
                if st in LABEL2ID and txt:
                    g[(f.stem, rnd(s.get("start")))] = {"text": txt, "y": LABEL2ID[st]}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    rows = load_rows()
    gold = load_gold()
    X = np.array([r["text"] for r in rows], dtype=object)
    y = np.array([r["y"] for r in rows])
    g = np.array([r["vid"] for r in rows])
    keys = [r["key"] for r in rows]
    tok = AutoTokenizer.from_pretrained(args.model)
    out = str(DATA / "bert_out")
    print(f"{len(rows)} claim-bearing stance segments | {len(gold)} gold | model {args.model}\n")

    def run_cv(splitter, groups=None):
        yp = np.empty_like(y)
        it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
        for k, (tr, te) in enumerate(it, 1):
            print(f"    fold {k} (train={len(tr)}, test={len(te)})")
            yp[te] = train_fold(args.model, tok, X[tr], y[tr], X[te], y[te],
                                args.epochs, args.batch, args.max_len, out)
        return f1_score(y, yp, average="macro")

    print("(a) segment 5-fold"); a = run_cv(StratifiedKFold(5, shuffle=True, random_state=42))
    print("(b) video GroupKFold"); b = run_cv(GroupKFold(5), groups=g)

    print("(c) gold held-out")
    tr = [i for i, k in enumerate(keys) if k not in gold]
    Xg = np.array([v["text"] for v in gold.values()], dtype=object)
    yg = np.array([v["y"] for v in gold.values()])
    pg = train_fold(args.model, tok, X[tr], y[tr], Xg, yg,
                    args.epochs, args.batch, args.max_len, out)
    c = f1_score(yg, pg, average="macro")

    print("\n" + "=" * 60)
    print(f"BERT STANCE ({args.model}) Macro-F1, claim-bearing only")
    print(f"  (a) segment 5-fold  : {a:.3f}")
    print(f"  (b) video GroupKFold: {b:.3f}    <- sizinti kontrollu")
    print(f"  (c) gold test       : {c:.3f}    <- insan gerçegine karsi")
    print("=" * 60)
    print("krsl. TF-IDF (honest_eval.py): 0.522 / 0.472 / 0.369")


if __name__ == "__main__":
    main()
