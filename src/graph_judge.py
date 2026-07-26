"""
Task 9 — LLM-as-a-judge validation of the knowledge graph.

Samples edges from the person-claim-stance graph (build_graph_full.py) and, for
each, gives an LLM the actual evidence segments and asks whether the edge is
faithful: does this speaker really hold this stance on this claim category, given
their own quotes? This estimates the graph's structural integrity (what fraction
of edges are supported by the underlying text) — a required validation step
alongside the manual gold annotation.

Reuses the provider clients from label_segments.py (OpenAI / Gemini).

Output: prints a validity rate + per-edge verdicts, and writes
        ../data/graph/graph_judge_report.json.

Usage:
  export OPENAI_API_KEY="..."
  python src/graph_judge.py --sample 20
  python src/graph_judge.py --provider gemini --sample 30
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from claim_taxonomy import ID2NAME, ID2DESC
from label_segments import build_client, DEFAULT_MODELS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
GRAPH_DIR = DATA_DIR / "graph"
STANCES = ["Support", "Refute", "Neutral"]
SEED = 42

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["valid", "invalid", "unclear"],
                    "description": "Is the claimed stance supported by the quotes?"},
        "reason": {"type": "string", "description": "One-sentence justification."},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

PROMPT = """You are auditing an edge of a person-claim-stance knowledge graph built from YouTube AGI discussions.

EDGE UNDER REVIEW:
- Speaker: {speaker}
- Claim category: {claim_name} ({claim_desc})
- Asserted stance toward the anchor ("AGI is near-term / something to be optimistic about"): {stance}
  (Support = optimistic/near-term, Refute = skeptical/far-off, Neutral = no clear position)

EVIDENCE — actual quotes from this speaker on this claim category:
{evidence}

Question: Given ONLY these quotes, is the asserted stance a faithful summary of what the speaker says about this claim category? Answer "valid" if the quotes support it, "invalid" if they contradict it, "unclear" if there is not enough signal."""


def load_segments():
    segs = []
    for f in sorted(LABELED_DIR.glob("*.json")):
        segs.extend(json.loads(f.read_text(encoding="utf-8")))
    return segs


def evidence_for(segments, speaker, claim_id, stance, max_q=3):
    """Return up to max_q quotes by `speaker` on `claim_id` with that stance."""
    quotes = []
    for s in segments:
        if (s.get("speaker") == speaker and stance == s.get("stance")
                and claim_id in (s.get("claim_labels") or [])):
            t = (s.get("text") or "").strip().replace("\n", " ")
            if t:
                quotes.append(t[:400])
        if len(quotes) >= max_q:
            break
    return quotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--sample", type=int, default=20, help="edges to audit")
    args = ap.parse_args()

    graph_path = GRAPH_DIR / "person_claim_stance_graph.json"
    if not graph_path.exists():
        raise SystemExit("Run build_graph_full.py first (missing person_claim_stance_graph.json).")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    segments = load_segments()

    edges = graph["edges"]
    random.seed(SEED)
    random.shuffle(edges)

    model = args.model or DEFAULT_MODELS[args.provider]
    client = build_client(args.provider)
    print(f"Auditing up to {args.sample} edges with {args.provider}:{model}\n")

    report, counts = [], defaultdict(int)
    audited = 0
    for e in edges:
        if audited >= args.sample:
            break
        speaker = e["source"].split("::", 1)[1]
        claim_id = e["target"].split("::", 1)[1]
        stance = e["dominant_stance"]
        quotes = evidence_for(segments, speaker, claim_id, stance)
        if not quotes:
            continue
        ev = "\n".join(f"  {i+1}. \"{q}\"" for i, q in enumerate(quotes))
        prompt = PROMPT.format(speaker=speaker, claim_name=ID2NAME.get(claim_id, claim_id),
                               claim_desc=ID2DESC.get(claim_id, ""), stance=stance, evidence=ev)
        try:
            if args.provider == "openai":
                res = _call_openai_schema(client, model, prompt)
            else:
                res = _call_gemini_schema(client, model, prompt)
        except Exception as ex:
            print(f"  [skip] {speaker} / {claim_id}: {str(ex)[:80]}")
            continue
        verdict = res.get("verdict", "unclear")
        counts[verdict] += 1
        audited += 1
        report.append({"speaker": speaker, "claim": claim_id, "stance": stance,
                       "verdict": verdict, "reason": res.get("reason", ""),
                       "n_quotes": len(quotes)})
        print(f"  [{verdict:7}] {speaker[:16]:16} · {ID2NAME.get(claim_id, claim_id)[:26]:26} · {stance}")

    valid = counts["valid"]
    rate = valid / audited if audited else 0.0
    print(f"\n=== GRAPH VALIDATION (LLM-as-a-judge) ===")
    print(f"Edges audited : {audited}")
    print(f"valid={counts['valid']}  invalid={counts['invalid']}  unclear={counts['unclear']}")
    print(f"Structural validity rate: {rate:.1%}")

    out = GRAPH_DIR / "graph_judge_report.json"
    out.write_text(json.dumps(
        {"audited": audited, "counts": dict(counts), "validity_rate": rate,
         "edges": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out}")


# --- structured-output wrappers (JSON schema), mirroring label_segments.py ---
def _call_openai_schema(client, model, prompt):
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "edge_verdict", "strict": True,
                                         "schema": JUDGE_SCHEMA}})
    return json.loads(resp.choices[0].message.content)


def _call_gemini_schema(client, model, prompt):
    from google.genai import types
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JUDGE_SCHEMA, temperature=0.0))
    return json.loads(resp.text)


if __name__ == "__main__":
    main()
