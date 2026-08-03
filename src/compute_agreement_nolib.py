"""Step 3b (no-deps variant) — LLM silver vs human/AI-verified gold agreement.

Same output as compute_agreement.py but implements Cohen's kappa, accuracy and
the confusion matrix in pure Python, so it runs even where scipy/sklearn fail to
load. Counts only reviewed==True segments in data/gold/.

Usage:  python src/compute_agreement_nolib.py
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
STANCES = ["Support", "Refute", "Neutral"]


def load_reviewed():
    rows = []
    for f in sorted(GOLD_DIR.glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("reviewed"):
                rows.append(s)
    return rows


def cohen_kappa(pairs):
    """pairs = list of (a, b). Returns (accuracy, kappa, n)."""
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    labels = set()
    for a, b in pairs:
        labels.add(a)
        labels.add(b)
    ca = Counter(a for a, b in pairs)
    cb = Counter(b for a, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return po, kappa, n


def report_field(rows, field):
    pairs = [(s.get(f"{field}_silver"), s.get(field)) for s in rows
             if s.get(f"{field}_silver") is not None and s.get(field) is not None]
    r = cohen_kappa(pairs)
    if r is None:
        print(f"  {field:9}: no comparable values")
        return None
    po, kappa, n = r
    print(f"  {field:9}: acc={po:.3f}  kappa={kappa:.3f}  (n={n})")
    return pairs


def confusion(pairs, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for silver, gold in pairs:
        if silver in idx and gold in idx:
            m[idx[gold]][idx[silver]] += 1  # rows=gold, cols=silver
    return m


def main():
    rows = load_reviewed()
    if not rows:
        print("No reviewed segments found in data/gold/.")
        return
    print(f"Reviewed (gold) segments: {len(rows)}\n")
    print("=== LLM silver vs gold ===")
    report_field(rows, "speaker")
    report_field(rows, "is_claim")
    report_field(rows, "topic")
    stance_pairs = report_field(rows, "stance")

    if stance_pairs:
        m = confusion(stance_pairs, STANCES)
        print("\nStance confusion (rows=gold, cols=LLM silver) order", STANCES)
        for i, l in enumerate(STANCES):
            print(f"  {l:8}", m[i])
        # gold stance distribution
        gold_dist = Counter(g for _, g in stance_pairs)
        print("\nGold stance distribution:", {s: gold_dist[s] for s in STANCES})

    print("\nKappa guide: <0.20 poor · 0.21-0.40 fair · 0.41-0.60 moderate · "
          "0.61-0.80 substantial · >0.80 almost perfect")


if __name__ == "__main__":
    main()
