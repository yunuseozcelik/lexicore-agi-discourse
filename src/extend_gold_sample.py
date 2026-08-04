"""
Step 3b — grow the GOLD sample without touching existing human work.

build_gold_sample.py seeds data/gold/ from scratch and refuses to run once any
segment is reviewed (with --force it DELETES the gold files). That is the right
guard, but it leaves no way to continue: after the first batch is fully
reviewed, label_manual.py has an empty queue and the only documented option is
destructive.

This script adds N *new* unreviewed segments to the existing gold files:
  - segments already present in data/gold/ are never re-picked and never
    modified (matched on video + start/end), so reviewed work is preserved,
  - the new picks are stratified by silver stance and spread across videos,
    the same balance build_gold_sample.py aims for, and
  - each pick gets its silver snapshot + reviewed=False, so compute_agreement.py
    and label_manual.py treat them exactly like first-batch segments.

Usage:
  python src/extend_gold_sample.py --n 200        # add ~200 (~67 per stance)
  python src/extend_gold_sample.py --n 100 --dry-run
Then:
  python src/translate_gold.py                    # translate only the new ones
  streamlit run src/label_manual.py
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
# same set build_gold_sample.py / label_manual.py snapshot
EDITABLE = ["speaker", "is_claim", "claim", "claim_labels", "stance", "topic"]
SEED = 42


def seg_key(seg):
    """Identity of a segment within its video (start/end are stable across files)."""
    return (seg.get("start"), seg.get("end"))


def load_dir(path):
    return {f.stem: json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(path.glob("*.json"))}


def stance_of(seg):
    st = seg.get("stance")
    return st if st in STANCES else "Neutral"


def stratified_pick(pool, taken, n, seed):
    """Pick ~n segments not already in `taken`, balanced by stance, spread by video.

    pool  -> {video_id: [segment, ...]}   (silver)
    taken -> {(video_id, seg_key), ...}   (already in gold)
    Returns {(video_id, index), ...}.
    """
    rng = random.Random(seed)
    by_stance = defaultdict(list)
    for vid, segs in pool.items():
        for i, s in enumerate(segs):
            if (vid, seg_key(s)) in taken:
                continue
            by_stance[stance_of(s)].append((vid, i))

    per_stance = max(1, n // len(STANCES))
    chosen = set()
    for st in STANCES:
        items = by_stance[st][:]
        rng.shuffle(items)
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
    s = dict(seg)
    for field in EDITABLE:
        s.setdefault(f"{field}_silver", s.get(field))
    s["reviewed"] = False
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="how many NEW segments to add (default 200)")
    ap.add_argument("--seed", type=int, default=SEED, help="sampling seed")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be added, write nothing")
    args = ap.parse_args()

    if not SILVER_DIR.exists() or not any(SILVER_DIR.glob("*.json")):
        print("data/labeled/ içinde silver dosya yok. Önce label_segments.py çalıştır.")
        return

    pool = load_dir(SILVER_DIR)
    gold = load_dir(GOLD_DIR) if GOLD_DIR.exists() else {}

    taken = {(vid, seg_key(s)) for vid, segs in gold.items() for s in segs}
    reviewed_before = sum(1 for segs in gold.values() for s in segs if s.get("reviewed"))
    print(f"Mevcut gold: {len(taken)} segment ({reviewed_before} incelenmiş).")
    print(f"Kalan havuz: {sum(len(v) for v in pool.values()) - len(taken)} segment.")

    chosen = stratified_pick(pool, taken, args.n, args.seed)
    if not chosen:
        print("Eklenecek yeni segment bulunamadı — havuz tükenmiş olabilir.")
        return

    by_video = defaultdict(list)
    for vid, idx in chosen:
        by_video[vid].append(idx)

    stance_count = defaultdict(int)
    for vid, idxs in by_video.items():
        for i in idxs:
            stance_count[stance_of(pool[vid][i])] += 1

    print(f"\nEklenecek: {len(chosen)} segment / {len(by_video)} video")
    print("Stance dengesi:", {s: stance_count[s] for s in STANCES})

    if args.dry_run:
        print("\n--dry-run: hiçbir dosya yazılmadı.")
        return

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for vid in sorted(by_video):
        existing = gold.get(vid, [])
        new = [snapshot(pool[vid][i]) for i in sorted(by_video[vid])]
        merged = sorted(existing + new, key=lambda s: (s.get("start") or 0))
        (GOLD_DIR / f"{vid}.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(taken) + len(chosen)
    print(f"\nGold artık {total} segment ({reviewed_before} incelenmiş, "
          f"{total - reviewed_before} sırada).")
    print("\nSonraki: python src/translate_gold.py")
    print("Sonra:   streamlit run src/label_manual.py")


if __name__ == "__main__":
    main()
