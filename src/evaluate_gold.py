"""Step 3c — full evaluation of the LLM silver pipeline against the human/AI gold.

Treats the reviewed gold labels as ground truth and scores the LLM "silver"
predictions as a classifier: per-class precision / recall / F1, macro-F1,
accuracy and Cohen's kappa, for both `stance` and `is_claim`. Also emits the
stance confusion matrix and a handful of concrete error examples.

Pure Python (no sklearn/scipy). Reads data/gold/*.json, reviewed==True only.
Writes a machine-readable summary to data/eval_report.json for the demo.

Usage:  python src/evaluate_gold.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
OUT = DATA_DIR / "eval_report.json"
STANCES = ["Support", "Refute", "Neutral"]


def load_reviewed():
    rows = []
    for f in sorted(GOLD_DIR.glob("*.json")):
        vid = f.stem
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("reviewed"):
                s = dict(s)
                s["_video"] = vid
                rows.append(s)
    return rows


def cohen_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    labels = set(a for a, _ in pairs) | set(b for _, b in pairs)
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return po, kappa, n


def prf_per_class(pairs, labels):
    """pairs = (pred_silver, true_gold). Returns per-class p/r/f1 + support."""
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()
    for pred, true in pairs:
        support[true] += 1
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    out = {}
    for l in labels:
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        r = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[l] = {"precision": p, "recall": r, "f1": f1, "support": support[l]}
    macro_f1 = sum(out[l]["f1"] for l in labels) / len(labels)
    return out, macro_f1


def confusion(pairs, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for silver, gold in pairs:
        if silver in idx and gold in idx:
            m[idx[gold]][idx[silver]] += 1  # rows=gold, cols=silver
    return m


def field_pairs(rows, field):
    return [(s.get(f"{field}_silver"), s.get(field)) for s in rows
            if s.get(f"{field}_silver") is not None and s.get(field) is not None]


def error_examples(rows, field, k=6):
    ex = []
    for s in rows:
        sv, gd = s.get(f"{field}_silver"), s.get(field)
        if sv is not None and gd is not None and sv != gd:
            ex.append({
                "video": s.get("_video"),
                "start": s.get("start"),
                "silver": sv,
                "gold": gd,
                "text": (s.get("text_tr") or s.get("text") or "")[:240],
            })
    return ex[:k]


def main():
    rows = load_reviewed()
    print(f"Reviewed gold segments: {len(rows)}\n")
    report = {"n": len(rows), "fields": {}}

    for field in ["is_claim", "stance"]:
        pairs = field_pairs(rows, field)
        # normalize bool/str for stable labels
        labels = STANCES if field == "stance" else sorted(
            {str(a) for a, _ in pairs} | {str(b) for _, b in pairs})
        norm = [(str(a), str(b)) for a, b in pairs]
        po, kappa, n = cohen_kappa(norm)
        prf, macro = prf_per_class(norm, labels)
        print(f"=== {field} ===  acc={po:.3f}  kappa={kappa:.3f}  macroF1={macro:.3f}  (n={n})")
        for l in labels:
            d = prf[l]
            print(f"   {l:8}  P={d['precision']:.2f}  R={d['recall']:.2f}  "
                  f"F1={d['f1']:.2f}  (n={d['support']})")
        report["fields"][field] = {
            "accuracy": po, "kappa": kappa, "macro_f1": macro, "n": n,
            "labels": labels, "per_class": prf,
        }
        if field == "stance":
            m = confusion(norm, STANCES)
            report["fields"][field]["confusion"] = m
            report["fields"][field]["confusion_labels"] = STANCES
            print("   confusion (rows=gold, cols=silver):")
            for i, l in enumerate(STANCES):
                print(f"     {l:8}", m[i])
        report["fields"][field]["errors"] = error_examples(rows, field)
        print()

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
