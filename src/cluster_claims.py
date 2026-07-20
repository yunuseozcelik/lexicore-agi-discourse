"""
Step 5 — canonical claim clustering.

  1. Loads every raw claim string from data/labeled/*.json.
  2. Vectorizes them with TF-IDF and clusters with KMeans.
  3. For each cluster, picks the claim closest to the centroid as the
     canonical (representative) claim.
  4. Writes data/graph/canonical_claims.json: cluster id -> canonical claim,
     member claims, and how many segments map to it.

Usage:
  python cluster_claims.py [--k 20]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
OUT_PATH = DATA_DIR / "graph" / "canonical_claims.json"


def load_claims():
    """Return list of raw claim strings (one per claim-bearing segment)."""
    claims = []
    for f in sorted(LABELED_DIR.glob("*.json")):
        for seg in json.loads(f.read_text(encoding="utf-8")):
            c = (seg.get("claim") or "").strip()
            if seg.get("is_claim") and c:
                claims.append(c)
    return claims


def cluster_claims(claims, k):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vectorizer.fit_transform(claims)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    clusters = {}
    for cluster_id in range(k):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        members = [claims[i] for i in idx]

        # pick the member closest to the cluster centroid as the canonical claim
        centroid = km.cluster_centers_[cluster_id]
        member_vecs = X[idx].toarray()
        dists = np.linalg.norm(member_vecs - centroid, axis=1)
        canonical = members[int(np.argmin(dists))]

        clusters[str(cluster_id)] = {
            "canonical_claim": canonical,
            "size": len(members),
            "members": members,
        }
    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="number of canonical claim clusters")
    args = parser.parse_args()

    claims = load_claims()
    n = len(claims)
    if n == 0:
        print("No claims found. Run label_segments.py first.")
        return

    k = min(args.k, n)
    print(f"Clustering {n} raw claims into {k} canonical claims...")

    clusters = cluster_claims(claims, k)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(clusters)} clusters to {OUT_PATH}")

    sizes = Counter({cid: c["size"] for cid, c in clusters.items()})
    print("\nCluster sizes (largest first):")
    for cid, size in sizes.most_common():
        print(f"  [{cid}] size={size}: {clusters[cid]['canonical_claim'][:80]}")


if __name__ == "__main__":
    main()
