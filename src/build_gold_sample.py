"""
Step 3a — build the stratified GOLD sample to review.

Hand-labeling all 2782 silver segments is infeasible, and label_manual.py by
itself just walks a video top-to-bottom, which would over-sample the dominant
Neutral class and the biggest videos. For a gold TEST set we instead want a
*stratified* sample so that:

  - every stance (Support / Refute / Neutral) is represented roughly equally,
    so the gold set can actually measure per-class F1 (Support is the rarest
    silver class, so proportional sampling would leave too few), and
  - the picks are spread across all 23 videos / many speakers, not concentrated
    in one long video.

This script selects ~N segments (default 450, i.e. ~150 per stance) with a fixed
seed and SEEDS data/gold/<video>.json with ONLY those segments, each marked
reviewed=False and with its silver value snapshotted as <field>_silver. Because
label_manual.py loads the gold file when it exists, the annotator then reviews
exactly this stratified subset (per video) and nothing else.

Idempotent-ish: if a gold file already has reviewed segments, this refuses to
overwrite it (so you never lose human work). Use --force to rebuild from scratch.

Usage:
  python src/build_gold_sample.py            # 450 segments, seed 42
  python src/build_gold_sample.py --n 300    # smaller sample
  python src/build_gold_sample.py --force    # rebuild (drops any prior seeding)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SILVER_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
STANCES = ["Support", "Refute", "Neutral"]
# same set label_manual.py snapshots, so agreement can be computed on all of them
EDITABLE = ["speaker", "is_claim", "claim", "claim_labels", "stance", "topic"]
SEED = 42


def load_pool():
    """Return {video_id: [segment, ...]} for every silver file."""
    pool = {}
    for f in sorted(SILVER_DIR.glob("*.json")):
        pool[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return pool


def has_human_work():
    """True if any existing gold file already contains a reviewed segment."""
    if not GOLD_DIR.exists():
        return False
    for f in GOLD_DIR.glob("*.json"):
        for s in json.loads(f.read_text(encoding="utf-8")):
            if s.get("reviewed"):
                return True
    return False


def stratified_pick(pool, n):
    """Pick ~n segments, balanced across stance, spread across videos.

    Returns a set of (video_id, segment_index) keys.
    """
    rng = random.Random(SEED)
    # bucket every segment by stance, remembering where it came from
    by_stance = defaultdict(list)          # stance -> [(video, idx), ...]
    for vid, segs in pool.items():
        for i, s in enumerate(segs):
            st = s.get("stance") if s.get("stance") in STANCES else "Neutral"
            by_stance[st].append((vid, i))

    per_stance = max(1, n // len(STANCES))
    chosen = set()
    for st in STANCES:
        items = by_stance[st][:]
        rng.shuffle(items)
        # spread within a stance across videos: sort a shuffled list by a
        # round-robin rank so consecutive picks come from different videos
        seen = defaultdict(int)
        ranked = []
        for vid, idx in items:
            ranked.append((seen[vid], rng.random(), vid, idx))
            seen[vid] += 1
        ranked.sort()
        for _, _, vid, idx in ranked[:per_stance]:
            chosen.add((vid, idx))
    return chosen


def snapshot(seg):
    """Return a copy of seg with silver snapshots + reviewed=False."""
    s = dict(seg)
    for field in EDITABLE:
        s.setdefault(f"{field}_silver", s.get(field))
    s["reviewed"] = False
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450, help="target sample size (~150/stance)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if gold files already exist (drops prior seeding)")
    args = ap.parse_args()

    if not SILVER_DIR.exists() or not any(SILVER_DIR.glob("*.json")):
        print("No silver files in data/labeled/. Run label_segments.py first.")
        return
    if has_human_work() and not args.force:
        print("data/gold/ already has REVIEWED segments — refusing to overwrite.\n"
              "Run compute_agreement.py to see progress, or --force to rebuild "
              "(this discards existing gold work).")
        return

    pool = load_pool()
    chosen = stratified_pick(pool, args.n)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    if args.force:
        for f in GOLD_DIR.glob("*.json"):
            f.unlink()

    # group chosen keys by video and write one gold file per video that has picks
    by_video = defaultdict(list)
    for vid, idx in chosen:
        by_video[vid].append(idx)

    stance_count = defaultdict(int)
    total = 0
    for vid in sorted(by_video):
        idxs = sorted(by_video[vid])
        segs = [snapshot(pool[vid][i]) for i in idxs]
        for s in segs:
            stance_count[s.get("stance") if s.get("stance") in STANCES else "Neutral"] += 1
        (GOLD_DIR / f"{vid}.json").write_text(
            json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        total += len(segs)

    print(f"Seeded {total} segments across {len(by_video)} videos into data/gold/.")
    print("Stance balance:", {s: stance_count[s] for s in STANCES})
    print("\nNext: streamlit run src/label_manual.py")
    print("Then:  python src/compute_agreement.py")


if __name__ == "__main__":
    main()
