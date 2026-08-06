"""Improved stance labeling — few-shot and category-conditional (Gemini).

This fixes two weaknesses of the original zero-shot single-anchor stance pass
(see PROGRESS.md and the project guide, sections 2.5-2.6):

  1. Zero-shot -> few-shot. The prompt now shows worked examples so the model
     has a concrete rubric instead of guessing from a bare instruction.
  2. Single global anchor -> category-conditional. The old pass scored every
     segment against one fixed proposition ("AGI is near-term / optimistic"),
     which is ill-defined when the speaker is discussing a different topic. Here
     each claim CATEGORY carries its own proposition (claim_taxonomy.py), and the
     speaker's stance is scored per category they actually express -> claim_stances.

It reuses the existing per-segment claim_labels (already assigned): it only
re-derives stance, so it is cheaper and does not disturb speaker/claim labels.
A backward-compatible scalar `stance` (toward the original global anchor) is
also produced so the honest_eval gold comparison keeps working.

Usage:
  python src/label_stance_v2.py --indir data/labeled --outdir data/labeled_stance_v2
  python src/label_stance_v2.py --indir data/gold --outdir data/gold_stance_v2 --limit 120
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_taxonomy import CLAIM_IDS, CLAIM_PROPOSITIONS, propositions_prompt_block
from label_segments import ANCHOR_CLAIM

STANCES = ["Support", "Refute", "Neutral"]


def gemini_client():
    """Build a Gemini client, forcing the key from the project .env so a stale
    GEMINI_API_KEY / GOOGLE_API_KEY already in the shell can't shadow it."""
    import os
    from google import genai
    key = None
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    key = key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY in .env or environment.")
    return genai.Client(api_key=key)

FEW_SHOT = """EXAMPLES

Segment: "Honestly I think advanced AI could pose a real existential danger if we
don't get alignment right — this is not something to be casual about."
claim_labels: [ai_safety_risk]
-> claim_stances: [{"category": "ai_safety_risk", "stance": "Support"}]
   (agrees AI poses serious risk)
-> stance (global optimism anchor): Refute

Segment: "People keep saying we need some big new idea, but I really don't buy
that — just scaling what we already have keeps working better than anyone
expected."
claim_labels: [scaling_compute, breakthroughs_needed]
-> claim_stances: [{"category": "scaling_compute", "stance": "Support"},
                   {"category": "breakthroughs_needed", "stance": "Refute"}]
   (backs scaling; rejects that new breakthroughs are needed)
-> stance (global optimism anchor): Support

Segment: "So today we're going to talk a bit about your background before we get
into the technical stuff."
claim_labels: [other]
-> claim_stances: []
-> stance (global optimism anchor): Neutral
"""

PROMPT_TEMPLATE = """You label the STANCE a speaker takes in a segment of a long-form AGI / AI-safety
discussion. You are given the segment and the claim categories it already
expresses. For EACH category, decide the speaker's stance toward that category's
proposition:
- Support : the speaker agrees with / argues for the proposition.
- Refute  : the speaker disagrees with / argues against it.
- Neutral : the speaker mentions the topic but takes no clear side.

CATEGORY PROPOSITIONS (score each expressed category against its own proposition):
{propositions}

Also give one backward-compatible overall `stance` toward this global anchor:
"{anchor}"
(Support = optimistic / AGI near-term, Refute = skeptical / far off, Neutral = no clear side.)

{few_shot}

Now label this segment.
claim_labels: [{labels}]
SEGMENT:
\"\"\"{text}\"\"\"
"""


def build_schema():
    return {
        "type": "object",
        "properties": {
            "claim_stances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": CLAIM_IDS},
                        "stance": {"type": "string", "enum": STANCES},
                    },
                    "required": ["category", "stance"],
                },
            },
            "stance": {"type": "string", "enum": STANCES},
        },
        "required": ["claim_stances", "stance"],
    }


def build_prompt(text, labels):
    expressed = [c for c in labels if c in CLAIM_PROPOSITIONS]
    return PROMPT_TEMPLATE.format(
        propositions=propositions_prompt_block(expressed or CLAIM_IDS),
        anchor=ANCHOR_CLAIM,
        few_shot=FEW_SHOT,
        labels=", ".join(labels),
        text=text.strip(),
    )


def call_gemini(client, model, prompt, schema, retries=5):
    from google.genai import types
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            msg = str(e)
            wait = 15 * attempt if ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                                    or "quota" in msg.lower()) else 2 * attempt
            print(f"    [retry {attempt}/{retries}] {msg[:120]} (waiting {wait}s)")
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser(description="Few-shot category-conditional stance labeling.")
    ap.add_argument("--indir", default="data/labeled")
    ap.add_argument("--outdir", default="data/labeled_stance_v2")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--limit", type=int, default=None, help="Label first N claim segments total.")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--only-gold", action="store_true",
                    help="Input dir is gold: only re-label human-reviewed segments.")
    args = ap.parse_args()

    client = gemini_client()
    schema = build_schema()
    root = Path(__file__).resolve().parent.parent
    indir, outdir = root / args.indir, root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model}  |  {indir} -> {outdir}\n")

    n = 0
    for f in sorted(indir.glob("*.json")):
        segs = json.loads(f.read_text(encoding="utf-8"))
        for seg in segs:
            if args.limit is not None and n >= args.limit:
                break
            if args.only_gold and not seg.get("reviewed"):
                continue
            labels = seg.get("claim_labels") or []
            text = (seg.get("text") or "").strip()
            # non-claims have no stance target
            if not text or not seg.get("is_claim") or not labels or labels == ["other"]:
                seg["claim_stances"] = []
                seg["stance_v2"] = "Neutral"
                continue
            out = call_gemini(client, args.model, build_prompt(text, labels), schema)
            if out is None:
                print(f"  [{f.stem}] FAILED — left unchanged")
                continue
            seg["claim_stances"] = [cs for cs in out.get("claim_stances", [])
                                    if cs.get("category") in CLAIM_IDS]
            seg["stance_v2"] = out.get("stance")
            n += 1
            print(f"  [{f.stem}] {n:4} | {out.get('stance'):7} | "
                  f"{[(c['category'], c['stance']) for c in seg['claim_stances']]}")
            time.sleep(args.sleep)
        (outdir / f.name).write_text(
            json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.limit is not None and n >= args.limit:
            break

    print(f"\nRe-labeled {n} claim segments -> {outdir}")


if __name__ == "__main__":
    main()
