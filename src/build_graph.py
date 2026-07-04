"""
Step 4 — mini person-stance knowledge graph (first working version).

Reads the labeled segments from ../data/labeled/*.json and builds a small
person -> stance -> anchor-claim graph:

  - Person nodes  : each identified speaker (Unknown segments are dropped).
  - Claim node    : a single anchor claim (the same ANCHOR_CLAIM stance was
                    scored against in label_segments.py).
  - Stance edges  : one edge per (speaker, stance) pair, weighted by how many
                    segments that speaker took that stance in. Each speaker can
                    therefore have up to three edges (Support / Refute / Neutral).

This is intentionally a *mini* skeleton: it does not yet cluster claims into
canonical nodes or add the time dimension — those are planned for the semantic
graph. What it does give is the first concrete, queryable graph object plus a
per-speaker stance summary.

Output:
  ../data/graph/person_stance_graph.json   (nodes + edges)
  ../data/graph/graph.png                  (visualization, if matplotlib+networkx
                                            are installed; skipped otherwise)
  prints a per-speaker dominant-stance summary table.

Usage:
  python build_graph.py
  # optional, for the PNG figure:
  pip install networkx matplotlib
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
GRAPH_DIR = DATA_DIR / "graph"

STANCES = ["Support", "Refute", "Neutral"]

# Same anchor every stance was labeled against (see label_segments.py). Kept
# here so the graph is self-contained.
ANCHOR_CLAIM = (
    "AGI will arrive in the near term and/or is something to be optimistic about."
)


def load_segments():
    """Return all labeled segments that have both a speaker and a stance."""
    segs = []
    for f in sorted(LABELED_DIR.glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            segs.append(s)
    return segs


def build_graph(segments):
    """Build nodes + weighted stance edges from labeled segments.

    Skips segments whose speaker is unknown/missing or whose stance is not one
    of the three canonical labels, so the graph stays clean.
    """
    # (speaker, stance) -> count
    edge_counts = Counter()
    speaker_totals = Counter()

    for s in segments:
        spk = s.get("speaker")
        st = s.get("stance")
        if not spk or spk == "Unknown":
            continue
        if st not in STANCES:
            continue
        edge_counts[(spk, st)] += 1
        speaker_totals[spk] += 1

    anchor_id = "CLAIM::anchor"
    nodes = [{
        "id": anchor_id,
        "type": "claim",
        "label": ANCHOR_CLAIM,
    }]
    for spk, total in sorted(speaker_totals.items(), key=lambda kv: -kv[1]):
        nodes.append({
            "id": f"PERSON::{spk}",
            "type": "person",
            "label": spk,
            "segments": total,
        })

    edges = []
    for (spk, st), w in sorted(edge_counts.items(), key=lambda kv: -kv[1]):
        edges.append({
            "source": f"PERSON::{spk}",
            "target": anchor_id,
            "stance": st,
            "weight": w,
        })

    return {"nodes": nodes, "edges": edges}, edge_counts, speaker_totals


def dominant_stance(edge_counts, speaker):
    """Return (stance, count) the speaker most often took."""
    per = {st: edge_counts.get((speaker, st), 0) for st in STANCES}
    st = max(per, key=per.get)
    return st, per[st], per


# Colors for each stance edge (also used in the PNG legend).
STANCE_COLORS = {"Support": "#2ca02c", "Refute": "#d62728", "Neutral": "#7f7f7f"}


def draw_graph(graph, out_path):
    """Render the graph to a PNG. No-op (prints a hint) if libs are missing.

    Person nodes are drawn around a central anchor-claim node; each stance edge
    is colored (green=Support, red=Refute, grey=Neutral) and its width scales
    with the number of segments (weight).
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")  # no display needed
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("\n[viz] skipped PNG (install: pip install networkx matplotlib)")
        return

    G = nx.MultiDiGraph()
    for n in graph["nodes"]:
        G.add_node(n["id"], **n)
    for e in graph["edges"]:
        G.add_edge(e["source"], e["target"], stance=e["stance"], weight=e["weight"])

    anchor = next(n["id"] for n in graph["nodes"] if n["type"] == "claim")
    persons = [n["id"] for n in graph["nodes"] if n["type"] == "person"]

    # anchor in the centre, people on a circle around it
    pos = nx.circular_layout(persons, scale=1.0)
    pos[anchor] = (0.0, 0.0)

    fig, ax = plt.subplots(figsize=(11, 8))

    nx.draw_networkx_nodes(G, pos, nodelist=[anchor], node_color="#1f77b4",
                           node_size=2600, node_shape="s", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=persons, node_color="#ffcc66",
                           node_size=1600, ax=ax)

    # each (person, stance) edge, colored + width by weight
    for e in graph["edges"]:
        nx.draw_networkx_edges(
            G, pos, edgelist=[(e["source"], e["target"])],
            edge_color=STANCE_COLORS.get(e["stance"], "#000000"),
            width=1.0 + 0.25 * e["weight"], alpha=0.7,
            connectionstyle="arc3,rad=0.08", arrows=True, ax=ax)

    labels = {n["id"]: n["label"] if n["type"] == "person" else "ANCHOR CLAIM\n(AGI near-term)"
              for n in graph["nodes"]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    legend = [Line2D([0], [0], color=c, lw=3, label=s)
              for s, c in STANCE_COLORS.items()]
    ax.legend(handles=legend, title="Stance (edge width = #segments)", loc="upper left")
    ax.set_title("Person–Stance Knowledge Graph (mini)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[viz] saved PNG -> {out_path}")


def main():
    segments = load_segments()
    if not segments:
        print("No labeled segments. Run label_segments.py first.")
        return

    graph, edge_counts, speaker_totals = build_graph(segments)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRAPH_DIR / "person_stance_graph.json"
    out_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_person = sum(1 for n in graph["nodes"] if n["type"] == "person")
    print("=== PERSON-STANCE KNOWLEDGE GRAPH (mini) ===")
    print(f"Person nodes : {n_person}")
    print(f"Claim nodes  : 1  (anchor)")
    print(f"Stance edges : {len(graph['edges'])}")
    print(f"Saved -> {out_path}\n")

    # per-speaker dominant stance summary (sorted by involvement)
    print("Per-speaker stance summary (Unknown speakers excluded):")
    header = f"{'Speaker':20} {'Segs':>4} | {'Support':>7} {'Refute':>6} {'Neutral':>7} | Dominant"
    print(header)
    print("-" * len(header))
    for spk, total in sorted(speaker_totals.items(), key=lambda kv: -kv[1]):
        st, cnt, per = dominant_stance(edge_counts, spk)
        print(f"{spk[:20]:20} {total:>4} | {per['Support']:>7} {per['Refute']:>6} "
              f"{per['Neutral']:>7} | {st} ({cnt})")

    # dataset-level edge stats
    total_edge_w = sum(e["weight"] for e in graph["edges"])
    by_stance = Counter()
    for e in graph["edges"]:
        by_stance[e["stance"]] += e["weight"]
    print("\nStance edge weight totals (identified speakers only):")
    for st in STANCES:
        print(f"  {st:8}: {by_stance[st]}")
    print(f"  TOTAL   : {total_edge_w}")

    # optional PNG visualization
    draw_graph(graph, GRAPH_DIR / "graph.png")


if __name__ == "__main__":
    main()
