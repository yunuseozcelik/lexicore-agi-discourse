"""
Step 3b — LLM-vs-human agreement (Cohen's kappa) on the gold set.

For every human-reviewed segment in data/gold/, compares the LLM "silver" label
(snapshotted as *_silver by label_manual.py) against the human "gold" label and
reports, per field:
  - raw accuracy (how often the LLM already agreed with the human),
  - Cohen's kappa (agreement corrected for chance),
  - a confusion matrix (for stance).

Kappa interpretation (Landis & Koch): <0.20 poor, 0.21-0.40 fair,
0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.0 almost perfect. This is the
headline number for "how trustworthy are our silver labels", and it justifies
using the LLM to label the remaining (unreviewed) segments for BERT training.

Usage:
  python src/compute_agreement.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
STANCES = ["Support", "Refute", "Neutral"]


def load_reviewed():
    """Return list of reviewed segments across all gold files."""
    rows = []
    for f in sorted(GOLD_DIR.glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("reviewed"):
                rows.append(s)
    return rows


def report_field(rows, field, labels=None):
    """Print accuracy + kappa for one field (silver vs gold)."""
    pairs = [(s.get(f"{field}_silver"), s.get(field)) for s in rows
             if s.get(f"{field}_silver") is not None and s.get(field) is not None]
    if not pairs:
        print(f"  {field:9}: no comparable values")
        return
    silver, gold = zip(*pairs)
    acc = accuracy_score(gold, silver)
    try:
        kappa = cohen_kappa_score(gold, silver, labels=labels)
    except Exception:
        kappa = float("nan")
    print(f"  {field:9}: acc={acc:.3f}  kappa={kappa:.3f}  (n={len(pairs)})")
    return silver, gold


def main():
    if not GOLD_DIR.exists():
        print("No data/gold/ yet. Review some segments with label_manual.py first.")
        return
    rows = load_reviewed()
    if not rows:
        print("No reviewed segments found in data/gold/.")
        return

    print(f"Reviewed (gold) segments: {len(rows)}\n")
    print("=== LLM silver vs human gold ===")
    report_field(rows, "speaker")
    report_field(rows, "is_claim", labels=[True, False])
    report_field(rows, "topic")
    stance_pair = report_field(rows, "stance", labels=STANCES)

    if stance_pair:
        silver, gold = stance_pair
        print("\nStance confusion (rows=human gold, cols=LLM silver) order", STANCES)
        print(confusion_matrix(gold, silver, labels=STANCES))

    print("\nKappa guide: <0.20 poor · 0.21-0.40 fair · 0.41-0.60 moderate · "
          "0.61-0.80 substantial · >0.80 almost perfect")


if __name__ == "__main__":
    main()
