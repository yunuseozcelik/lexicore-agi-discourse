"""Improved stance labeling — few-shot and category-conditional.

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

Providers: openai (default, gpt-5.6-luna) or gemini. Keys are read from .env.
OpenAI calls run concurrently (--workers) so a whole file finishes fast.

Usage:
  python src/label_stance_v2.py --indir data/gold --outdir data/gold_stance_v2 --only-gold
  python src/label_stance_v2.py --provider gemini --model gemini-3-flash-preview ...
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_taxonomy import CLAIM_IDS, CLAIM_PROPOSITIONS, propositions_prompt_block
from label_segments import ANCHOR_CLAIM

STANCES = ["Support", "Refute", "Neutral"]


def env_key(name):
    """Read a key straight from the project .env, ignoring any stale value in the
    shell environment (which repeatedly shadowed the right one)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("'\"")
    import os
    return os.environ.get(name)


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


def build_prompt(text, labels):
    expressed = [c for c in labels if c in CLAIM_PROPOSITIONS]
    return PROMPT_TEMPLATE.format(
        propositions=propositions_prompt_block(expressed or CLAIM_IDS),
        anchor=ANCHOR_CLAIM,
        few_shot=FEW_SHOT,
        labels=", ".join(labels),
        text=text.strip(),
    )


# ------------------------------------------------------------------ schema
_ITEM = {"type": "object", "additionalProperties": False,
         "properties": {"category": {"type": "string", "enum": CLAIM_IDS},
                        "stance": {"type": "string", "enum": STANCES}},
         "required": ["category", "stance"]}
_PROPS = {"claim_stances": {"type": "array", "items": _ITEM},
          "stance": {"type": "string", "enum": STANCES}}
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": _PROPS, "required": ["claim_stances", "stance"]}


# ------------------------------------------------------------------ providers
def make_caller(provider, model):
    """Return a function prompt -> parsed dict (with retries), for the provider."""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=env_key("OPENAI_API_KEY"))
        rf = {"type": "json_schema",
              "json_schema": {"name": "stance", "strict": True, "schema": SCHEMA}}

        def call(prompt):
            for attempt in range(1, 6):
                try:
                    r = client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}],
                        response_format=rf)
                    return json.loads(r.choices[0].message.content)
                except Exception as e:
                    msg = str(e)
                    wait = 15 * attempt if ("429" in msg or "rate" in msg.lower()) else 2 * attempt
                    print(f"    [retry {attempt}] {msg[:100]} ({wait}s)", flush=True)
                    time.sleep(wait)
            return None
        return call

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=env_key("GEMINI_API_KEY"))
    # Gemini's schema dialect does not take additionalProperties.
    gschema = {"type": "object", "properties": {
        "claim_stances": {"type": "array", "items": {"type": "object", "properties": {
            "category": {"type": "string", "enum": CLAIM_IDS},
            "stance": {"type": "string", "enum": STANCES}}, "required": ["category", "stance"]}},
        "stance": {"type": "string", "enum": STANCES}}, "required": ["claim_stances", "stance"]}

    def call(prompt):
        for attempt in range(1, 6):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=gschema, temperature=0.0))
                return json.loads(r.text)
            except Exception as e:
                msg = str(e)
                wait = 15 * attempt if ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                                        or "503" in msg) else 2 * attempt
                print(f"    [retry {attempt}] {msg[:100]} ({wait}s)", flush=True)
                time.sleep(wait)
        return None
    return call


def label_one(call, seg):
    """Return (claim_stances, stance_v2) for a segment, or None to skip API."""
    labels = seg.get("claim_labels") or []
    text = (seg.get("text") or "").strip()
    if not text or not seg.get("is_claim") or not labels or labels == ["other"]:
        return [], "Neutral"  # non-claim: no stance target
    out = call(build_prompt(text, labels))
    if out is None:
        return None
    cs = [c for c in out.get("claim_stances", []) if c.get("category") in CLAIM_IDS]
    return cs, out.get("stance")


def main():
    ap = argparse.ArgumentParser(description="Few-shot category-conditional stance labeling.")
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini"])
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--indir", default="data/labeled")
    ap.add_argument("--outdir", default="data/labeled_stance_v2")
    ap.add_argument("--limit", type=int, default=None, help="Label first N claim segments total.")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent API calls (openai).")
    ap.add_argument("--only-gold", action="store_true",
                    help="Input dir is gold: only re-label human-reviewed segments.")
    args = ap.parse_args()

    call = make_caller(args.provider, args.model)
    root = Path(__file__).resolve().parent.parent
    indir, outdir = root / args.indir, root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{args.provider}:{args.model} | {indir} -> {outdir} | workers={args.workers}\n", flush=True)

    n = 0
    for f in sorted(indir.glob("*.json")):
        segs = json.loads(f.read_text(encoding="utf-8"))
        prev_path = outdir / f.name
        if prev_path.exists():  # resume
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            for i, p in enumerate(prev):
                if i < len(segs) and "stance_v2" in p:
                    segs[i]["stance_v2"] = p["stance_v2"]
                    segs[i]["claim_stances"] = p.get("claim_stances", [])

        todo = [s for s in segs if "stance_v2" not in s
                and (not args.only_gold or s.get("reviewed"))]
        if args.limit is not None:
            todo = todo[:max(0, args.limit - n)]
        if not todo:
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda s: label_one(call, s), todo))
        for seg, res in zip(todo, results):
            if res is None:
                continue
            seg["claim_stances"], seg["stance_v2"] = res
            if seg.get("is_claim") and (seg.get("claim_labels") or []) not in ([], ["other"]):
                n += 1
        prev_path.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{f.stem}] {len(todo)} labeled | running total {n}", flush=True)
        if args.limit is not None and n >= args.limit:
            break

    print(f"\nRe-labeled {n} claim segments this run -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
