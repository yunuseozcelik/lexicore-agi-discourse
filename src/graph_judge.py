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

# Active taxonomy; `--taxonomy v2` audits the collapsed 8-category graph. The
# stored claim_labels are v1, so they are projected onto v2 when picking
# evidence — otherwise a v2 category id would never match a segment.
_COLLAPSE = None


def use_taxonomy(version):
    global ID2NAME, ID2DESC, _COLLAPSE
    if version == "v2":
        from claim_taxonomy_v2 import ID2NAME_V2, ID2DESC_V2, collapse_labels
        ID2NAME, ID2DESC, _COLLAPSE = ID2NAME_V2, ID2DESC_V2, collapse_labels
    else:
        from claim_taxonomy import ID2NAME as N1, ID2DESC as D1
        ID2NAME, ID2DESC, _COLLAPSE = N1, D1, None

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

# The original prompt defined the stance against one fixed anchor ("AGI is
# near-term / something to be optimistic about") but then asked whether it
# summarised what the speaker says about THIS CATEGORY. The judge resolved the
# contradiction by falling back on the anchor, so an edge about Scaling was
# graded on whether the quotes discussed AGI timelines -- and returned "unclear"
# whenever they did not. That is a measurement artefact, not a bad edge: it hit
# the descriptive categories hardest (Scaling 11.5%, Timeline 33.3%). The stance
# is now evaluated relative to the category itself.
PROMPT = """You are auditing an edge of a person-claim-stance knowledge graph built from YouTube AGI discussions.

EDGE UNDER REVIEW:
- Speaker: {speaker}
- Claim category: {claim_name} ({claim_desc})
- Asserted stance ON THIS CATEGORY: {stance}
  Interpret the stance relative to the category above, NOT to any other topic.
  The stance describes the speaker's OUTLOOK on the topic, not whether they
  think the topic is important or real:
    Support = optimistic / affirming — progress is happening, it will work out,
              the positive case holds
    Refute  = pessimistic / concerned / skeptical — they doubt it, warn about
              it, or emphasise problems and risks
    Neutral = they discuss it without leaning either way

  Careful: for a topic like "AI Safety & Risk", a speaker who stresses that the
  risks are serious is expressing CONCERN, which is Refute — not a rejection of
  the topic's importance. Do not mark such an edge invalid.

EVIDENCE — actual quotes from this speaker on this claim category:
{evidence}

Question: Given ONLY these quotes, is the asserted stance a faithful summary of the speaker's position ON THIS CATEGORY? Do not require the quotes to mention AGI timelines or overall optimism about AI — judge only the category named above. Answer "valid" if the quotes support the asserted stance, "invalid" if they contradict it, "unclear" only if the quotes genuinely take no position on this category."""


def load_segments():
    segs = []
    for f in sorted(LABELED_DIR.glob("*.json")):
        segs.extend(json.loads(f.read_text(encoding="utf-8")))
    return segs


def evidence_for(segments, speaker, claim_id, stance, max_q=3, claims_only=False):
    """Return up to max_q quotes by `speaker` on `claim_id` with that stance.

    claims_only must match how the graph was built: if the edge only counts
    claim-bearing segments, the evidence shown to the judge must come from the
    same pool, otherwise the judge rates the edge against quotes that did not
    contribute to it.
    """
    quotes = []
    for s in segments:
        if claims_only and not s.get("is_claim"):
            continue
        labels = s.get("claim_labels") or []
        if _COLLAPSE is not None:
            labels = _COLLAPSE(labels)
        if (s.get("speaker") == speaker and stance == s.get("stance")
                and claim_id in labels):
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
    ap.add_argument("--taxonomy", choices=["v1", "v2"], default="v1")
    ap.add_argument("--claims-only", action="store_true",
                    help="audit the claim-filtered graph (build_graph_full.py --claims-only)")
    ap.add_argument("--tag", default="",
                    help="suffix for the report filename, to keep runs side by side")
    args = ap.parse_args()

    use_taxonomy(args.taxonomy)
    suffix = "" if args.taxonomy == "v1" else "_v2"
    if args.claims_only:
        suffix += "_claims"

    graph_path = GRAPH_DIR / f"person_claim_stance_graph{suffix}.json"
    if not graph_path.exists():
        raise SystemExit(f"Run build_graph_full.py --taxonomy {args.taxonomy} first "
                         f"(missing {graph_path.name}).")
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
        quotes = evidence_for(segments, speaker, claim_id, stance,
                              claims_only=args.claims_only)
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
    print(f"\n=== GRAPH VALIDATION (LLM-as-a-judge, taxonomy {args.taxonomy}) ===")
    print(f"Edges audited : {audited}")
    print(f"valid={counts['valid']}  invalid={counts['invalid']}  unclear={counts['unclear']}")
    print(f"Structural validity rate: {rate:.1%}")

    # Per-category breakdown — this is what exposed the v1 problem (abstract
    # categories scored 0-8% while distinctive ones scored 46-55%).
    per_cat = defaultdict(lambda: defaultdict(int))
    for r in report:
        per_cat[r["claim"]][r["verdict"]] += 1
    print(f"\n{'Category':34} {'n':>4} {'valid':>6} {'rate':>7}")
    print("-" * 54)
    cat_rows = {}
    for cid, c in sorted(per_cat.items(),
                         key=lambda kv: -(kv[1]["valid"] / max(1, sum(kv[1].values())))):
        n = sum(c.values())
        r = c["valid"] / n if n else 0.0
        cat_rows[cid] = {"n": n, "valid": c["valid"], "invalid": c["invalid"],
                         "unclear": c["unclear"], "rate": r}
        print(f"{ID2NAME.get(cid, cid)[:34]:34} {n:>4} {c['valid']:>6} {r:>6.1%}")

    out = GRAPH_DIR / f"graph_judge_report{suffix}{args.tag}.json"
    out.write_text(json.dumps(
        {"taxonomy": args.taxonomy, "audited": audited, "counts": dict(counts),
         "validity_rate": rate, "per_category": cat_rows, "edges": report},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


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
