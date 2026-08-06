"""Judge the category-conditional graph: are edges backed by the speaker's own
quotes, where stance is read against each category's own proposition."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from claim_taxonomy import ID2NAME, CLAIM_PROPOSITIONS
from label_stance_v2 import env_key

DATA = Path(__file__).resolve().parent.parent / "data"
STANCES = ["Support", "Refute", "Neutral"]

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"verdict": {"type": "string", "enum": ["valid", "invalid", "unclear"]},
                         "reason": {"type": "string"}},
          "required": ["verdict", "reason"]}

PROMPT = """Auditing one edge of a person-claim-stance graph from AGI discussions.

PROPOSITION: "{prop}"
The speaker's asserted stance toward this proposition: {stance}
  Support = agrees with it, Refute = disagrees with it, Neutral = no clear side.

The speaker's own quotes on this topic:
{ev}

Given only these quotes, is "{stance}" a faithful summary of the speaker's
position on the proposition? Answer valid, invalid, or unclear."""


def load_v2segs():
    segs = []
    for f in sorted((DATA / "labeled_stance_v2").glob("*.json")):
        segs.extend(json.loads(f.read_text(encoding="utf-8")))
    return segs


def quotes_for(segs, spk, cid, stance, k=3):
    out = []
    for s in segs:
        if s.get("speaker") != spk or not s.get("is_claim"):
            continue
        cs = {c["category"]: c["stance"] for c in (s.get("claim_stances") or [])}
        if cs.get(cid) != stance:
            continue
        t = (s.get("text") or "").strip().replace("\n", " ")
        if t:
            out.append(t[:400])
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--model", default="gpt-5.6-luna")
    args = ap.parse_args()

    graph = json.loads((DATA / "graph" / "person_claim_stance_graph_claims_cond.json")
                       .read_text(encoding="utf-8"))
    segs = load_v2segs()
    from openai import OpenAI
    client = OpenAI(api_key=env_key("OPENAI_API_KEY"))

    edges = graph["edges"]
    random.seed(42)
    random.shuffle(edges)

    counts = defaultdict(int)
    per_cat = defaultdict(lambda: defaultdict(int))
    report = []
    n = 0
    for e in edges:
        if n >= args.sample:
            break
        spk = e["source"].split("::", 1)[1]
        cid = e["target"].split("::", 1)[1]
        stance = e["dominant_stance"]
        prop = CLAIM_PROPOSITIONS.get(cid)
        if not prop:
            continue
        qs = quotes_for(segs, spk, cid, stance)
        if not qs:
            continue
        ev = "\n".join(f'  {i+1}. "{q}"' for i, q in enumerate(qs))
        prompt = PROMPT.format(prop=prop, stance=stance, ev=ev)
        r = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "v", "strict": True, "schema": SCHEMA}})
        v = json.loads(r.choices[0].message.content)["verdict"]
        counts[v] += 1
        per_cat[cid][v] += 1
        report.append({"speaker": spk, "claim": cid, "stance": stance, "verdict": v})
        n += 1
        print(f"  [{v:7}] {spk[:16]:16} · {ID2NAME.get(cid, cid)[:24]:24} · {stance}")

    rate = counts["valid"] / n if n else 0
    print(f"\nedges={n}  valid={counts['valid']} invalid={counts['invalid']} unclear={counts['unclear']}")
    print(f"validity rate: {rate:.1%}")
    out = DATA / "graph" / "judge_conditional_report.json"
    out.write_text(json.dumps({"audited": n, "counts": dict(counts),
                               "validity_rate": rate, "edges": report}, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
