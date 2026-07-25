"""
Bootstrap silver multi-label canonical claim labels via embedding similarity.

The LLM labeler (label_segments.py) fills `claim_labels` going forward, but the
already-labeled 5-video set predates the taxonomy. This script assigns a
`claim_labels` set to every segment WITHOUT one, so the multi-label claim model
(train_claim_multilabel.py) can be trained immediately, and it also serves as a
cheap fallback when re-labeling is not desired.

Method (zero-shot, no API): embed each segment's claim text (fallback: segment
text) and each taxonomy category's description+keywords with the same
sentence-transformer used elsewhere (all-MiniLM-L6-v2), then assign every
category whose cosine similarity exceeds a threshold. Non-claim segments get
["other"]. The auto labels are also stored under `claim_labels_auto` for
transparency, so a later human/LLM pass can be compared against them.

Usage:
  pip install sentence-transformers
  python src/assign_claim_labels.py                 # fill only missing claim_labels
  python src/assign_claim_labels.py --overwrite     # recompute for all segments
  python src/assign_claim_labels.py --threshold 0.28 --topk 3
"""

import argparse
import json
from pathlib import Path

import numpy as np

from claim_taxonomy import CLAIM_TAXONOMY, CLAIM_IDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
MODEL_NAME = "all-MiniLM-L6-v2"
OTHER = "other"


def taxonomy_texts():
    """One descriptive string per category (description + keywords) to embed."""
    texts = []
    for c in CLAIM_TAXONOMY:
        texts.append(f"{c['name']}. {c['description']} " + ", ".join(c["keywords"]))
    return texts


def cosine_matrix(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="min cosine similarity for a category to be assigned")
    ap.add_argument("--topk", type=int, default=2,
                    help="always keep at least the top-k categories for a claim")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute claim_labels even if already present")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Run:  pip install sentence-transformers")

    files = sorted(LABELED_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No data/labeled/*.json — run label_segments.py first.")

    print(f"Loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    cat_emb = model.encode(taxonomy_texts(), show_progress_bar=False)
    # index of the 'other' category so we can special-case non-claims
    other_idx = CLAIM_IDS.index(OTHER)

    total, assigned = 0, 0
    for f in files:
        segs = json.loads(f.read_text(encoding="utf-8"))
        # collect the texts we need to embed (claim if present, else segment text)
        idxs, texts = [], []
        for i, s in enumerate(segs):
            if s.get("claim_labels") and not args.overwrite:
                continue
            if not s.get("is_claim"):
                # non-claims are trivially "other"; no need to embed
                s["claim_labels"] = [OTHER]
                s["claim_labels_auto"] = [OTHER]
                assigned += 1
                continue
            idxs.append(i)
            texts.append((s.get("claim") or s.get("text") or "").strip())
        total += len(segs)

        if texts:
            emb = model.encode(texts, show_progress_bar=False)
            sim = cosine_matrix(np.asarray(emb), np.asarray(cat_emb))  # (n, C)
            for row, i in zip(sim, idxs):
                order = np.argsort(-row)
                # drop 'other' from the ranked candidates for real claims
                cand = [j for j in order if j != other_idx]
                chosen = [j for j in cand if row[j] >= args.threshold]
                if len(chosen) < args.topk:  # guarantee at least top-k
                    chosen = cand[:args.topk]
                labels = [CLAIM_IDS[j] for j in chosen]
                segs[i]["claim_labels"] = labels
                segs[i]["claim_labels_auto"] = labels
                assigned += 1

        f.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{f.stem}] {len(segs)} segments processed")

    print(f"\nAssigned claim_labels to {assigned} segments (threshold={args.threshold}, "
          f"topk={args.topk}). Non-claims -> ['{OTHER}'].")


if __name__ == "__main__":
    main()
