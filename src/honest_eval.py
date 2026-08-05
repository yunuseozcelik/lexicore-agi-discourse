"""Honest / leakage-controlled evaluation of the TF-IDF baselines.

Additive and CPU-only: it does NOT touch the training scripts. For each task it
reports Macro-F1 under three settings, so the reported numbers can be stated
honestly rather than as "model accuracy":

  (a) segment 5-fold   : the current method (StratifiedKFold over segments) —
                         this is silver-vs-silver cross-validation.
  (b) video GroupKFold : same classifier, but a whole video is held out each
                         fold, so segments from a video seen in training never
                         appear in test — controls the same-video adjacency leak.
  (c) gold held-out    : train on silver labels EXCLUDING the human-reviewed
                         segments, then test against the HUMAN gold labels —
                         real generalization to ground truth.

Usage:  python src/honest_eval.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_taxonomy import CLAIM_IDS

DATA = Path(__file__).resolve().parent.parent / "data"
STANCES = ["Support", "Refute", "Neutral"]
CLAIM_IDS = list(CLAIM_IDS)


def rnd(x):
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return x


def load_labeled():
    rows = []
    for f in sorted((DATA / "labeled").glob("*.json")):
        vid = f.stem
        for s in json.loads(f.read_text(encoding="utf-8")):
            rows.append({
                "vid": vid, "key": (vid, rnd(s.get("start"))),
                "text": (s.get("text") or "").strip(),
                "speaker": (s.get("speaker") or "").strip(),
                "stance": s.get("stance"),
                "claims": s.get("claim_labels") or [],
            })
    return rows


def load_gold():
    g = {}
    for f in sorted((DATA / "gold").glob("*.json")):
        vid = f.stem
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("reviewed"):
                g[(vid, rnd(s.get("start")))] = {
                    "text": (s.get("text") or "").strip(),
                    "speaker": (s.get("speaker") or "").strip(),
                    "stance": s.get("stance"),
                    "claims": s.get("claim_labels") or [],
                }
    return g


def clf():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
        LogisticRegression(max_iter=1000, class_weight="balanced"))


def ovr():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
        OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")))


def cv_single(X, y, g, labels, splitter, groups=None):
    yp = np.empty_like(y)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        yp[te] = clf().fit(X[tr], y[tr]).predict(X[te])
    return f1_score(y, yp, average="macro", labels=labels, zero_division=0)


def cv_multi(X, Y, splitter, groups=None):
    P = np.zeros_like(Y)
    it = splitter.split(X, Y, groups) if groups is not None else splitter.split(X)
    for tr, te in it:
        P[te] = ovr().fit(X[tr], Y[tr]).predict(X[te])
    return (f1_score(Y, P, average="micro", zero_division=0),
            f1_score(Y, P, average="macro", zero_division=0))


# ---------------------------------------------------------------- STANCE
def eval_stance(rows, gold):
    d = [(r["text"], r["stance"], r["vid"], r["key"]) for r in rows
         if r["stance"] in STANCES and r["text"]]
    X = np.array([a for a, *_ in d], dtype=object)
    y = np.array([b for _, b, *_ in d])
    g = np.array([c for *_, c, _ in d])
    keys = [k for *_, k in d]

    a = cv_single(X, y, g, STANCES, StratifiedKFold(5, shuffle=True, random_state=42))
    b = cv_single(X, y, g, STANCES, GroupKFold(5), groups=g)

    gkeys = {k for k, v in gold.items() if v["stance"] in STANCES}
    tr = [i for i, k in enumerate(keys) if k not in gold]
    gt = [(v["text"], v["stance"]) for k, v in gold.items()
          if v["stance"] in STANCES and v["text"]]
    m = clf().fit(X[tr], y[tr])
    Xg = np.array([t for t, _ in gt], dtype=object)
    yg = np.array([s for _, s in gt])
    c = f1_score(yg, m.predict(Xg), average="macro", labels=STANCES, zero_division=0)
    return a, b, c, len(d), len(gt)


# ---------------------------------------------------------------- CLAIM (multi-label)
def to_Y(claimlists):
    idx = {c: i for i, c in enumerate(CLAIM_IDS)}
    Y = np.zeros((len(claimlists), len(CLAIM_IDS)), dtype=np.int64)
    for i, cl in enumerate(claimlists):
        for c in cl:
            if c in idx:
                Y[i, idx[c]] = 1
    return Y


def eval_claim(rows, gold):
    d = [(r["text"], r["claims"], r["vid"], r["key"]) for r in rows
         if r["claims"] and r["text"]]
    X = np.array([a for a, *_ in d], dtype=object)
    Y = to_Y([b for _, b, *_ in d])
    g = np.array([c for *_, c, _ in d])
    keys = [k for *_, k in d]

    a = cv_multi(X, Y, KFold(5, shuffle=True, random_state=42))
    b = cv_multi(X, Y, GroupKFold(5), groups=g)

    tr = [i for i, k in enumerate(keys) if k not in gold]
    gt = [(v["text"], v["claims"]) for k, v in gold.items() if v["claims"] and v["text"]]
    m = ovr().fit(X[tr], Y[tr])
    Xg = np.array([t for t, _ in gt], dtype=object)
    Yg = to_Y([cl for _, cl in gt])
    Pg = m.predict(Xg)
    c = (f1_score(Yg, Pg, average="micro", zero_division=0),
         f1_score(Yg, Pg, average="macro", zero_division=0))
    return a, b, c, len(d), len(gt)


# ---------------------------------------------------------------- SPEAKER
def eval_speaker(rows, gold, min_count=5):
    d = [(r["text"], r["speaker"], r["vid"], r["key"]) for r in rows
         if r["text"] and r["speaker"] and r["speaker"].lower() != "unknown"]
    keep = {s for s, c in Counter(x[1] for x in d).items() if c >= min_count}
    d = [x for x in d if x[1] in keep]
    X = np.array([a for a, *_ in d], dtype=object)
    y = np.array([b for _, b, *_ in d])
    g = np.array([c for *_, c, _ in d])
    keys = [k for *_, k in d]
    labels = sorted(set(y))

    a = cv_single(X, y, g, labels, StratifiedKFold(5, shuffle=True, random_state=42))
    b = cv_single(X, y, g, labels, GroupKFold(5), groups=g)

    tr = [i for i, k in enumerate(keys) if k not in gold]
    gt = [(v["text"], v["speaker"]) for k, v in gold.items()
          if v["speaker"] and v["speaker"].lower() != "unknown" and v["speaker"] in keep and v["text"]]
    m = clf().fit(X[tr], y[tr])
    Xg = np.array([t for t, _ in gt], dtype=object)
    yg = np.array([s for _, s in gt])
    c = f1_score(yg, m.predict(Xg), average="macro",
                 labels=sorted(set(yg)), zero_division=0)
    return a, b, c, len(d), len(gt)


def main():
    rows = load_labeled()
    gold = load_gold()
    print(f"labeled segments: {len(rows)} | gold reviewed: {len(gold)}\n")

    st = eval_stance(rows, gold)
    cl = eval_claim(rows, gold)
    sp = eval_speaker(rows, gold)

    print("=" * 68)
    print("STANCE  (Macro-F1)                              n_silver / n_gold")
    print(f"  (a) segment 5-fold  : {st[0]:.3f}")
    print(f"  (b) video GroupKFold: {st[1]:.3f}    <- sizinti kontrollu")
    print(f"  (c) gold test       : {st[2]:.3f}    <- insan gerçegine karsi")
    print(f"      n = {st[3]} silver / {st[4]} gold")
    print("-" * 68)
    print("CLAIM   (micro / macro)")
    print(f"  (a) segment 5-fold  : {cl[0][0]:.3f} / {cl[0][1]:.3f}")
    print(f"  (b) video GroupKFold: {cl[1][0]:.3f} / {cl[1][1]:.3f}    <- sizinti kontrollu")
    print(f"  (c) gold test       : {cl[2][0]:.3f} / {cl[2][1]:.3f}    <- insan gerçegine karsi")
    print(f"      n = {cl[3]} silver / {cl[4]} gold")
    print("-" * 68)
    print("SPEAKER (Macro-F1)")
    print(f"  (a) segment 5-fold  : {sp[0]:.3f}")
    print(f"  (b) video GroupKFold: {sp[1]:.3f}    <- ayni kisi tek videodaysa ogrenilemez")
    print(f"  (c) gold test       : {sp[2]:.3f}")
    print(f"      n = {sp[3]} silver / {sp[4]} gold")
    print("=" * 68)


if __name__ == "__main__":
    main()
