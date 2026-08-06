"""
Task 5 — unified person-claim-stance knowledge graph.

Upgrades the mini person->anchor graph (build_graph.py) into the tripartite graph
the project asks for:

  - Person nodes : named speakers.
  - Claim nodes  : the canonical claim CATEGORIES (claim_taxonomy.py), not a
                   single anchor.
  - Edges        : person --> claim category, carrying the stance distribution
                   (Support / Refute / Neutral counts) the speaker took while
                   discussing that category, plus a dominant stance and weight.

Because each segment can express several canonical claims (multi-label), one
segment contributes to several person-claim edges. This is the structure that
enables stance-aware retrieval (eval_retrieval.py) and LLM-as-a-judge validation
(graph_judge.py).

Input : ../data/labeled/*.json (or --source gold), needs speaker + stance +
        claim_labels (run assign_claim_labels.py / label_segments.py first).
Output: ../data/graph/person_claim_stance_graph.json
        ../data/graph/graph_full.png (optional, needs networkx+matplotlib)

Usage:
  python src/build_graph_full.py
  python src/build_graph_full.py --source gold
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from claim_taxonomy import CLAIM_IDS, ID2NAME

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
STANCE_V2_DIR = DATA_DIR / "labeled_stance_v2"
GOLD_DIR = DATA_DIR / "gold"
GRAPH_DIR = DATA_DIR / "graph"
STANCES = ["Support", "Refute", "Neutral"]
STANCE_COLORS = {"Support": "#2ca02c", "Refute": "#d62728", "Neutral": "#7f7f7f"}

# Active taxonomy; `--taxonomy v2` swaps in the collapsed 8-category version and
# projects the stored v1 claim_labels onto it (see claim_taxonomy_v2.py).
CLAIM_IDS = list(CLAIM_IDS)
_COLLAPSE = None


def use_taxonomy(version):
    global CLAIM_IDS, ID2NAME, _COLLAPSE
    if version == "v2":
        from claim_taxonomy_v2 import CLAIM_IDS_V2, ID2NAME_V2, collapse_labels
        CLAIM_IDS, ID2NAME, _COLLAPSE = list(CLAIM_IDS_V2), ID2NAME_V2, collapse_labels
    else:
        from claim_taxonomy import CLAIM_IDS as V1, ID2NAME as N1
        CLAIM_IDS, ID2NAME, _COLLAPSE = list(V1), N1, None


def load_segments(source, conditional=False, predicted=False, taxonomy="v1"):
    if predicted:
        # graph from the classifiers' own out-of-fold predictions, not silver.
        # pred labels are already in the target taxonomy, so no collapse here.
        f = DATA_DIR / "predicted" / f"oof_labels_{taxonomy}.json"
        segs = json.loads(f.read_text(encoding="utf-8"))
        for s in segs:
            s["claim_labels"] = s.get("pred_claim_labels") or []
            s["stance"] = s.get("pred_stance")
        return segs

    # conditional stance lives in labeled_stance_v2 (has per-category claim_stances)
    src = GOLD_DIR if source == "gold" else (STANCE_V2_DIR if conditional else LABELED_DIR)
    segs = []
    for f in sorted(src.glob("*.json")):
        segs.extend(json.loads(f.read_text(encoding="utf-8")))
    if _COLLAPSE is not None:
        for s in segs:
            if s.get("claim_labels"):
                s["claim_labels"] = _COLLAPSE(s["claim_labels"])
    return segs


def build(segments, claims_only=False, conditional=False):
    """Build the tripartite graph.

    conditional=True uses the category-conditional stance (claim_stances): each
    person->category edge carries the stance the speaker took toward THAT
    category's proposition, instead of one global-anchor stance reused for every
    category the segment expressed. This is the fix for the single-anchor
    ambiguity documented in the guide (section 2.6).

    claims_only=True drops segments the labeller marked `is_claim=False`. All
    439 such segments that reach the graph are labelled Neutral -- not because
    the speaker was neutral, but because the labelling schema had no "no stance"
    option, so non-claims were funnelled into Neutral. Removing them takes out
    that artefact.

    Neutral is NOT removed: Support/Neutral/Refute is the stance scheme the
    project is built on, and "this speaker discusses the topic without taking a
    side" is a finding the graph is supposed to represent. An earlier version
    dropped Neutral edges entirely, which cut ~1500 segment-mentions and forced
    every edge into a binary it was never meant to carry.
    """
    # (person, claim_id) -> {stance: count}
    edge = defaultdict(lambda: {s: 0 for s in STANCES})
    person_segs = defaultdict(int)
    claim_segs = defaultdict(int)

    for s in segments:
        spk = s.get("speaker")
        labels = s.get("claim_labels") or []
        if not spk or spk == "Unknown" or not labels:
            continue
        if claims_only and not s.get("is_claim"):
            continue

        if conditional:
            # one edge per (category, its own stance); map v1 category -> active
            # taxonomy so v2 collapsing still works.
            person_segs[spk] += 1
            for c in (s.get("claim_stances") or []):
                cid, stc = c.get("category"), c.get("stance")
                if stc not in STANCES or not cid:
                    continue
                if _COLLAPSE is not None:
                    m = _COLLAPSE([cid])
                    cid = m[0] if m else None
                if not cid or cid not in CLAIM_IDS or (claims_only and cid == "other"):
                    continue
                edge[(spk, cid)][stc] += 1
                claim_segs[cid] += 1
            continue

        st = s.get("stance")
        if st not in STANCES:
            continue
        person_segs[spk] += 1
        for cid in labels:
            if cid not in CLAIM_IDS:
                continue
            # `other` means "no substantive AGI claim", so a stance edge on it
            # carries no discourse content and the audit scored it 4.2%. It is
            # dropped from the graph but kept in the taxonomy for the classifier.
            if claims_only and cid == "other":
                continue
            edge[(spk, cid)][st] += 1
            claim_segs[cid] += 1

    nodes = []
    for spk, tot in sorted(person_segs.items(), key=lambda kv: -kv[1]):
        nodes.append({"id": f"PERSON::{spk}", "type": "person",
                      "label": spk, "segments": tot})
    for cid in CLAIM_IDS:
        if claim_segs[cid]:
            nodes.append({"id": f"CLAIM::{cid}", "type": "claim",
                          "label": ID2NAME[cid], "segments": claim_segs[cid]})

    edges = []
    for (spk, cid), dist in edge.items():
        w = sum(dist.values())
        dom = max(STANCES, key=lambda s: dist[s])
        edges.append({
            "source": f"PERSON::{spk}", "target": f"CLAIM::{cid}",
            "support": dist["Support"], "refute": dist["Refute"],
            "neutral": dist["Neutral"], "weight": w, "dominant_stance": dom,
        })
    edges.sort(key=lambda e: -e["weight"])
    return {"nodes": nodes, "edges": edges}


def draw_heatmap(graph, out_path, top_n=20, subtitle=None):
    """Person x claim-category matrix: colour = dominant stance, alpha = weight.

    The bipartite line drawing does not survive 28 people x 8 categories -- ~200
    edges all funnel through the same corridor and nothing is readable. A matrix
    has no crossings by construction: every (person, category) pair gets its own
    cell, so both "who talks about what" and "which way they lean" are legible
    at a glance.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("\n[viz] skipped heatmap (install: pip install matplotlib)")
        return

    persons = [n for n in graph["nodes"] if n["type"] == "person"][:top_n]
    claims = [n for n in graph["nodes"] if n["type"] == "claim"]
    if not persons or not claims:
        return
    pidx = {p["id"]: i for i, p in enumerate(persons)}
    cidx = {c["id"]: i for i, c in enumerate(claims)}

    cell = {}
    for e in graph["edges"]:
        if e["source"] in pidx and e["target"] in cidx:
            cell[(pidx[e["source"]], cidx[e["target"]])] = e
    if not cell:
        return
    wmax = max(e["weight"] for e in cell.values())

    fig, ax = plt.subplots(figsize=(1.35 * len(claims) + 4, 0.42 * len(persons) + 2.5))
    for (r, c), e in cell.items():
        # alpha encodes evidence volume; sqrt so mid-weight cells stay visible
        a = 0.20 + 0.80 * (e["weight"] / wmax) ** 0.5
        ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                   facecolor=STANCE_COLORS[e["dominant_stance"]],
                                   alpha=a, edgecolor="white", linewidth=1.5))
        if e["weight"] >= max(3, 0.06 * wmax):
            ax.text(c + 0.5, r + 0.5, str(e["weight"]), ha="center", va="center",
                    fontsize=7, color="white" if a > 0.6 else "#333", fontweight="bold")

    ax.set_xlim(0, len(claims))
    ax.set_ylim(len(persons), 0)
    ax.set_xticks([i + 0.5 for i in range(len(claims))])
    ax.set_xticklabels([c["label"] for c in claims], rotation=35, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(persons))])
    ax.set_yticklabels([p["label"] for p in persons], fontsize=8)
    ax.set_xticks(range(len(claims) + 1), minor=True)
    ax.set_yticks(range(len(persons) + 1), minor=True)
    ax.grid(which="minor", color="#e8e8e8", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(facecolor=STANCE_COLORS[s], label=s) for s in STANCES
                       if any(e["dominant_stance"] == s for e in cell.values())],
              title="Dominant stance  (number = segments, shade = volume)",
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8)
    ax.set_title(f"Who takes which position — top {len(persons)} speakers", pad=12)
    if subtitle:
        ax.text(0.0, 1.0, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved heatmap -> {out_path}")


def draw(graph, out_path, min_weight=1, max_persons=None, title=None, subtitle=None):
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("\n[viz] skipped PNG (install: pip install networkx matplotlib)")
        return

    # Drawing every edge is unreadable (~200 lines through one corridor), and
    # most of them are 1-3 segment edges competing visually with 100+ segment
    # ones. Keep the strong edges and the people who carry them.
    edges = [e for e in graph["edges"] if e["weight"] >= min_weight]
    if max_persons:
        keep = [n["id"] for n in graph["nodes"] if n["type"] == "person"][:max_persons]
        edges = [e for e in edges if e["source"] in set(keep)]
    if not edges:
        edges = graph["edges"]
    live = {e["source"] for e in edges} | {e["target"] for e in edges}

    persons = [n["id"] for n in graph["nodes"] if n["type"] == "person" and n["id"] in live]
    claims = [n["id"] for n in graph["nodes"] if n["type"] == "claim" and n["id"] in live]
    G = nx.Graph()
    G.add_nodes_from(persons + claims)
    # bipartite layout: persons on the left column, claim categories on the right
    pos = {}
    for i, p in enumerate(persons):
        pos[p] = (-1.0, 1 - 2 * i / max(1, len(persons) - 1))
    for i, c in enumerate(claims):
        pos[c] = (1.0, 1 - 2 * i / max(1, len(claims) - 1))

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(-1.75, 1.85)
    wmax = max(e["weight"] for e in edges)
    # Straight lines all cross near the middle; a slight arc per edge separates
    # them so individual connections stay traceable.
    import numpy as _np
    for i, e in enumerate(sorted(edges, key=lambda x: x["weight"])):
        p, c = pos[e["source"]], pos[e["target"]]
        t = _np.linspace(0, 1, 60)
        bow = 0.22 * _np.sin(_np.pi * t) * (1 if i % 2 else -1)
        ax.plot(p[0] + (c[0] - p[0]) * t,
                p[1] + (c[1] - p[1]) * t + bow,
                color=STANCE_COLORS[e["dominant_stance"]],
                linewidth=0.5 + 5.0 * (e["weight"] / wmax) ** 0.7,
                alpha=0.45, zorder=1, solid_capstyle="round")

    nx.draw_networkx_nodes(G, pos, nodelist=persons, node_color="#ffcc66",
                           node_size=1500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=claims, node_color="#9ecae1",
                           node_size=1200, node_shape="s", ax=ax)
    # Labels outside the columns, so they never sit on top of the edges.
    for n in graph["nodes"]:
        if n["id"] not in pos:
            continue
        x, y = pos[n["id"]]
        left = n["type"] == "person"
        ax.text(x - 0.14 if left else x + 0.14, y, n["label"],
                ha="right" if left else "left", va="center",
                fontsize=9, zorder=3)
    # Only show stances that actually appear -- with --stance-only there are no
    # Neutral edges, and a legend entry for an unused colour is misleading.
    present = {e["dominant_stance"] for e in edges}
    legend = [Line2D([0], [0], color=col, lw=3, label=s)
              for s, col in STANCE_COLORS.items() if s in present]
    ax.legend(handles=legend, title="Dominant stance (edge width = #segments)",
              loc="lower center", ncol=len(legend))
    ax.set_title(title or "Person–Claim–Stance Knowledge Graph")
    if subtitle:
        ax.text(0.5, -0.03, subtitle, transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="#555")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[viz] saved PNG -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["labeled", "gold"], default="labeled")
    ap.add_argument("--taxonomy", choices=["v1", "v2"], default="v1")
    ap.add_argument("--claims-only", action="store_true",
                    help="drop segments marked is_claim=False (all of them are "
                         "Neutral-by-default artefacts) and the `other` category; "
                         "genuine Support/Neutral/Refute edges are all kept")
    ap.add_argument("--conditional-stance", action="store_true",
                    help="build edges from category-conditional claim_stances "
                         "(data/labeled_stance_v2) instead of one global stance")
    ap.add_argument("--predicted", action="store_true",
                    help="build from the classifiers' OOF predictions "
                         "(data/predicted/oof_labels_*.json) instead of silver labels")
    args = ap.parse_args()

    use_taxonomy(args.taxonomy)
    suffix = "" if args.taxonomy == "v1" else "_v2"
    if args.claims_only:
        suffix += "_claims"
    if args.conditional_stance:
        suffix += "_cond"
    if args.predicted:
        suffix += "_pred"

    segs = load_segments(args.source, conditional=args.conditional_stance,
                         predicted=args.predicted, taxonomy=args.taxonomy)
    if not segs:
        print("No segments. Run the pipeline first "
              "(label_stance_v2.py for --conditional-stance).")
        return
    graph = build(segs, claims_only=args.claims_only, conditional=args.conditional_stance)
    n_person = sum(1 for n in graph["nodes"] if n["type"] == "person")
    n_claim = sum(1 for n in graph["nodes"] if n["type"] == "claim")

    if not graph["edges"]:
        print("No person-claim edges — segments lack claim_labels. "
              "Run assign_claim_labels.py or label_segments.py first.")
        return

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    out = GRAPH_DIR / f"person_claim_stance_graph{suffix}.json"
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== PERSON-CLAIM-STANCE KNOWLEDGE GRAPH (taxonomy {args.taxonomy}) ===")
    print(f"Person nodes : {n_person}")
    print(f"Claim nodes  : {n_claim}  (canonical categories)")
    print(f"Edges        : {len(graph['edges'])}")
    print(f"Saved -> {out}\n")
    print("Top person-claim edges (by #segments):")
    print(f"{'Speaker':18} {'Claim category':32} {'#':>3} {'dominant':>8}")
    print("-" * 66)
    for e in graph["edges"][:15]:
        spk = e["source"].split("::", 1)[1]
        cl = e["target"].split("::", 1)[1]
        print(f"{spk[:18]:18} {ID2NAME.get(cl, cl)[:32]:32} {e['weight']:>3} "
              f"{e['dominant_stance']:>8}")

    # Say what the picture leaves out -- with --stance-only a large share of the
    # corpus is deliberately excluded, and a figure that hides that reads as if
    # it covered everything.
    kept = sum(e["support"] + e["refute"] + e["neutral"] for e in graph["edges"])
    if args.claims_only:
        sub = (f"Claim-bearing segments only (is_claim=True); the `other` category "
               f"is excluded. All three stances are shown. "
               f"{len(graph['edges'])} edges over {kept} segment-mentions.")
    else:
        sub = (f"All labelled segments. {len(graph['edges'])} edges over "
               f"{kept} segment-mentions.")

    draw_heatmap(graph, GRAPH_DIR / f"graph_heatmap{suffix}.png", subtitle=sub)
    draw(graph, GRAPH_DIR / f"graph_full{suffix}.png", min_weight=25, max_persons=8,
         subtitle=sub + "  Showing edges with >=25 segments for the 8 most active speakers.")


if __name__ == "__main__":
    main()
