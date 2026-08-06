"""Measure stance-labeling agreement against the human gold labels.

Compares how well each automatic stance source reproduces the human-reviewed
gold stance, on claim-bearing gold segments (is_claim=True):

  - old silver : original zero-shot single-anchor gpt-4o-mini stance (data/labeled)
  - new v2      : few-shot category-conditional stance (a re-labeled dir), read
                  from the `stance_v2` field.

Reports accuracy, Macro-F1, and Cohen's kappa vs gold. This isolates the effect
of the labeling improvement, since both are scored against the same human gold.

Usage:
  python src/eval_stance_agreement.py                         # old silver only
  python src/eval_stance_agreement.py --new-dir data/gold_stance_v2
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

DATA = Path(__file__).resolve().parent.parent / "data"
STANCES = ["Support", "Refute", "Neutral"]


def rnd(x):
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return x


def load_gold():
    g = {}
    for f in sorted((DATA / "gold").glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("reviewed"):
                g[(f.stem, rnd(s.get("start")))] = {
                    "stance": s.get("stance"), "is_claim": s.get("is_claim")}
    return g


def load_stance(dirname, field):
    d = {}
    for f in sorted((DATA / dirname).glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            d[(f.stem, rnd(s.get("start")))] = s.get(field)
    return d


def score(name, pred, gold):
    pairs = [(gold[k]["stance"], pred.get(k)) for k in gold
             if gold[k]["is_claim"] and gold[k]["stance"] in STANCES
             and pred.get(k) in STANCES]
    y = [a for a, _ in pairs]
    p = [b for _, b in pairs]
    print(f"{name:16} n={len(pairs):4}  acc={accuracy_score(y, p):.3f}  "
          f"macroF1={f1_score(y, p, average='macro', labels=STANCES, zero_division=0):.3f}  "
          f"kappa={cohen_kappa_score(y, p, labels=STANCES):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", default=None,
                    help="Dir with re-labeled segments carrying a stance_v2 field.")
    args = ap.parse_args()

    gold = load_gold()
    print(f"gold reviewed claim-bearing segments: "
          f"{sum(1 for v in gold.values() if v['is_claim'] and v['stance'] in STANCES)}\n")
    print("stance source vs human gold (claim-bearing only):")
    score("old silver", load_stance("labeled", "stance"), gold)
    if args.new_dir:
        score("new few-shot", load_stance(args.new_dir.replace("data/", ""), "stance_v2"), gold)


if __name__ == "__main__":
    main()
