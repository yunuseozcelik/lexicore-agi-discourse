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
GOLD_DIR = DATA_DIR / "gold"
GRAPH_DIR = DATA_DIR / "graph"
STANCES = ["Support", "Refute", "Neutral"]
STANCE_COLORS = {"Support": "#2ca02c", "Refute": "#d62728", "Neutral": "#7f7f7f"}


def load_segments(source):
    src = GOLD_DIR if source == "gold" else LABELED_DIR
    segs = []
    for f in sorted(src.glob("*.json")):
        segs.extend(json.loads(f.read_text(encoding="utf-8")))
    return segs


def build(segments):
    # (person, claim_id) -> {stance: count}
    edge = defaultdict(lambda: {s: 0 for s in STANCES})
    person_segs = defaultdict(int)
    claim_segs = defaultdict(int)

    for s in segments:
        spk = s.get("speaker")
        st = s.get("stance")
        labels = s.get("claim_labels") or []
        if not spk or spk == "Unknown" or st not in STANCES or not labels:
            continue
        person_segs[spk] += 1
        for cid in labels:
            if cid not in CLAIM_IDS:
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


def draw(graph, out_path):
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("\n[viz] skipped PNG (install: pip install networkx matplotlib)")
        return

    persons = [n["id"] for n in graph["nodes"] if n["type"] == "person"]
    claims = [n["id"] for n in graph["nodes"] if n["type"] == "claim"]
    G = nx.Graph()
    G.add_nodes_from(persons + claims)
    # bipartite layout: persons on the left column, claim categories on the right
    pos = {}
    for i, p in enumerate(persons):
        pos[p] = (-1.0, 1 - 2 * i / max(1, len(persons) - 1))
    for i, c in enumerate(claims):
        pos[c] = (1.0, 1 - 2 * i / max(1, len(claims) - 1))

    fig, ax = plt.subplots(figsize=(13, 9))
    for e in graph["edges"]:
        p, c = pos[e["source"]], pos[e["target"]]
        ax.plot([p[0], c[0]], [p[1], c[1]],
                color=STANCE_COLORS[e["dominant_stance"]],
                linewidth=0.4 + 0.5 * e["weight"], alpha=0.6, zorder=1)

    nx.draw_networkx_nodes(G, pos, nodelist=persons, node_color="#ffcc66",
                           node_size=1500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=claims, node_color="#9ecae1",
                           node_size=1200, node_shape="s", ax=ax)
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    legend = [Line2D([0], [0], color=col, lw=3, label=s)
              for s, col in STANCE_COLORS.items()]
    ax.legend(handles=legend, title="Dominant stance (edge width = #segments)",
              loc="lower center", ncol=3)
    ax.set_title("Person–Claim–Stance Knowledge Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[viz] saved PNG -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["labeled", "gold"], default="labeled")
    args = ap.parse_args()

    segs = load_segments(args.source)
    if not segs:
        print("No segments. Run the pipeline first.")
        return
    graph = build(segs)
    n_person = sum(1 for n in graph["nodes"] if n["type"] == "person")
    n_claim = sum(1 for n in graph["nodes"] if n["type"] == "claim")

    if not graph["edges"]:
        print("No person-claim edges — segments lack claim_labels. "
              "Run assign_claim_labels.py or label_segments.py first.")
        return

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    out = GRAPH_DIR / "person_claim_stance_graph.json"
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== PERSON-CLAIM-STANCE KNOWLEDGE GRAPH ===")
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

    draw(graph, GRAPH_DIR / "graph_full.png")


if __name__ == "__main__":
    main()
