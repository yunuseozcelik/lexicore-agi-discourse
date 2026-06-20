"""
Step 2 — zero-shot LLM labeling of segments (speaker + claim + stance).

Reads baseline segments from ../data/segments/*.json and, for each segment,
asks an LLM to fill in:
  - speaker : who is talking (from the per-video roster, or "Unknown")
  - claim   : a one-sentence paraphrase of the segment's main assertion
  - stance  : Support / Refute / Neutral toward a fixed anchor claim
  - topic / confidence : extra fields for clustering and manual QA
Labeled segments are written to ../data/labeled/<video>.json.

Stance needs a target, so each segment is scored against one reference
proposition (ANCHOR_CLAIM): Support = agrees AGI is near-term / optimistic,
Refute = skeptical / thinks it's far off, Neutral = no clear position.

Providers:
  - openai (default) : gpt-4o-mini
  - gemini           : gemini-2.5-flash

Usage:
  export OPENAI_API_KEY="..."
  python label_segments.py            # all segments, resumes if interrupted
  python label_segments.py --limit 80 # quick sample
"""

import argparse
import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEG_DIR = DATA_DIR / "segments"
OUT_DIR = DATA_DIR / "labeled"

DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
DEFAULT_MODELS = {
    "openai": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
}

# The single proposition every segment's stance is scored against.
ANCHOR_CLAIM = (
    "Artificial General Intelligence (AGI) will arrive in the near term "
    "(within roughly the next several years) and/or is broadly something to "
    "be optimistic about."
)

# Shared with fetch_transcripts.py — people we expect to appear, given as a
# hint so the model prefers real names over generic guesses.
TRACKED_PEOPLE = [
    "Jensen Huang", "Demis Hassabis", "Yann LeCun", "Jeff Dean",
    "Mustafa Suleyman", "Lex Fridman", "Dwarkesh Patel",
]

# ---------------------------------------------------------------------------
# Per-video participant roster. Captions have no speaker tags, so we read each
# video's intro ("Today I'm talking with X", "welcome Y") and record who is in
# it. The roster is passed to the model as a candidate set so it picks the
# speaker from conversational cues instead of guessing freely. This can be
# auto-extracted from video titles/descriptions or intro lines when scaling up.
# Leave host=None / guests=[] for unknown rosters (model falls back to context).
# ---------------------------------------------------------------------------
VIDEO_META = {
    "vif8NQcjVf0": {"host": "Lex Fridman",    "guests": ["Jensen Huang"],
                    "source": "Lex Fridman Podcast"},
    "b1TeeIG6Uaw": {"host": "Dwarkesh Patel", "guests": ["Victor Shih"],
                    "source": "Dwarkesh Patel"},
    "WLBsUarvWTw": {"host": "Dwarkesh Patel", "guests": ["Tamay Besiroglu", "Ege Erdil"],
                    "source": "Dwarkesh Patel"},
    "F_7M4Hc-usM": {"host": "Moderator",      "guests": ["Sam Altman"],
                    "source": "Stanford (How to Start a Startup)"},
}


def candidates_for(video_id):
    """Return the known speaker candidate list for a video (may be empty)."""
    meta = VIDEO_META.get(video_id, {})
    names = []
    if meta.get("host"):
        names.append(meta["host"])
    names.extend(meta.get("guests", []))
    return names

# Structured-output schema — forces the model to return clean JSON.
SCHEMA_PROPERTIES = {
    "speaker": {
        "type": "string",
        "description": "Name of the person speaking if identifiable from "
                       "context, otherwise 'Unknown'.",
    },
    "is_claim": {
        "type": "boolean",
        "description": "True if the segment contains a substantive "
                       "claim/assertion (not just a question or filler).",
    },
    "claim": {
        "type": "string",
        "description": "One concise sentence paraphrasing the main claim. "
                       "Empty string if is_claim is false.",
    },
    "topic": {
        "type": "string",
        "description": "Short 2-4 word topic tag, e.g. 'AGI timeline', "
                       "'AI safety', 'compute/chips', 'jobs', 'off-topic'.",
    },
    "stance": {
        "type": "string",
        "enum": ["Support", "Refute", "Neutral"],
        "description": "Stance of the speaker toward the ANCHOR claim.",
    },
    "confidence": {
        "type": "string",
        "enum": ["high", "medium", "low"],
    },
}
REQUIRED = list(SCHEMA_PROPERTIES.keys())

PROMPT_TEMPLATE = """You are labeling segments of long-form YouTube discussions about AGI and AI safety (2022-2026) to build a person-claim-stance dataset.

ANCHOR CLAIM (score stance against THIS):
"{anchor}"

Stance definitions (toward the anchor claim):
- Support : the speaker agrees / is optimistic AGI is near-term / argues in favor.
- Refute  : the speaker disagrees / is skeptical / thinks AGI is far off / argues against.
- Neutral : no clear position on the anchor, off-topic, or just a question/filler.

SPEAKERS IN THIS VIDEO (assign the speaker to one of these based on conversational cues such as who is asking vs. answering, self-references, and who was introduced; use "Unknown" ONLY if genuinely impossible):
{candidates}

Other people who may be mentioned across the dataset (do NOT assign unless clearly speaking): {people}

Label the following transcript segment. Extract the speaker, whether it states a claim, a one-sentence paraphrase of that claim, a short topic tag, the stance toward the anchor, and your confidence.

SEGMENT (t={start}s-{end}s):
\"\"\"{text}\"\"\"
"""


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------
def build_client(provider):
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("Run:  pip install openai")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set it first:\n  export OPENAI_API_KEY='your-key'")
        return OpenAI(api_key=key)

    if provider == "gemini":
        try:
            from google import genai
        except ImportError:
            raise SystemExit("Run:  pip install google-genai")
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("Set it first:\n  export GEMINI_API_KEY='your-key'")
        return genai.Client(api_key=key)

    raise SystemExit(f"Unknown provider: {provider}")


def _build_prompt(seg, candidates):
    cand_str = ", ".join(candidates) if candidates else (
        "(roster unknown — infer the speaker from introductions/context)")
    return PROMPT_TEMPLATE.format(
        anchor=ANCHOR_CLAIM,
        candidates=cand_str,
        people=", ".join(TRACKED_PEOPLE),
        start=seg.get("start"),
        end=seg.get("end"),
        text=seg.get("text", "").strip(),
    )


def _call_openai(client, model, prompt):
    schema = {
        "type": "object",
        "properties": SCHEMA_PROPERTIES,
        "required": REQUIRED,
        "additionalProperties": False,
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "segment_label", "strict": True, "schema": schema},
        },
    )
    return json.loads(resp.choices[0].message.content)


def _call_gemini(client, model, prompt):
    from google.genai import types
    schema = {"type": "object", "properties": SCHEMA_PROPERTIES, "required": REQUIRED}
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


def label_one(provider, client, model, seg, candidates, retries=3):
    """Return a dict of labels for one segment, or None on failure."""
    prompt = _build_prompt(seg, candidates)
    call = _call_openai if provider == "openai" else _call_gemini
    for attempt in range(1, retries + 1):
        try:
            return call(client, model, prompt)
        except Exception as e:  # rate limit / transient / parse
            wait = 2 * attempt
            print(f"    [retry {attempt}/{retries}] {e} (waiting {wait}s)")
            time.sleep(wait)
    return None


def merge_labels(seg, labels):
    """Write the model's labels back onto the segment dict (in place)."""
    seg["speaker"] = labels.get("speaker") or "Unknown"
    seg["claim"] = labels.get("claim") or None
    seg["stance"] = labels.get("stance")  # Support / Refute / Neutral
    # extra fields kept for clustering + manual QA
    seg["is_claim"] = labels.get("is_claim")
    seg["topic"] = labels.get("topic")
    seg["confidence"] = labels.get("confidence")
    return seg


def main():
    ap = argparse.ArgumentParser(description="LLM-label transcript segments.")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER,
                    choices=["openai", "gemini"], help="LLM provider.")
    ap.add_argument("--model", default=None,
                    help="Override model name (default depends on provider).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Label only the first N segments TOTAL (sample run). "
                         "Default: all.")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Only label these video IDs. Default: all in segments/.")
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="Seconds between API calls (rate limiting).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-label segments that already have a stance.")
    args = ap.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    print(f"Provider: {args.provider}  |  Model: {model}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = build_client(args.provider)

    seg_files = sorted(SEG_DIR.glob("*.json"))
    if args.videos:
        seg_files = [f for f in seg_files if f.stem in args.videos]
    if not seg_files:
        print("No segment files in data/segments/. Run fetch_transcripts.py first.")
        return

    labeled_count = 0
    dist = {"Support": 0, "Refute": 0, "Neutral": 0}
    claim_count = 0

    for f in seg_files:
        segments = json.loads(f.read_text(encoding="utf-8"))

        # resume: carry over already-labeled segments from a previous run
        out_path = OUT_DIR / f.name
        if out_path.exists() and not args.overwrite:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for i, p in enumerate(prev):
                if i < len(segments) and p.get("stance"):
                    segments[i] = p

        candidates = candidates_for(f.stem)
        print(f"[{f.stem}] {len(segments)} segments  | speakers: "
              f"{', '.join(candidates) if candidates else '(unknown roster)'}")
        for i, seg in enumerate(segments):
            if args.limit is not None and labeled_count >= args.limit:
                break
            if seg.get("stance") and not args.overwrite:
                continue  # already labeled (resume)

            labels = label_one(args.provider, client, model, seg, candidates)
            if labels is None:
                print(f"  [{i}] FAILED — left unlabeled")
                continue
            merge_labels(seg, labels)
            labeled_count += 1
            st = seg.get("stance")
            if st in dist:
                dist[st] += 1
            if seg.get("is_claim"):
                claim_count += 1
            print(f"  [{i}] {str(seg['speaker'])[:18]:18} | {str(st):7} | {seg.get('topic')}")
            time.sleep(args.sleep)

        # save after each video (so a crash doesn't lose work)
        out_path.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if args.limit is not None and labeled_count >= args.limit:
            print(f"\nReached --limit {args.limit}.")
            break

    print("\n=== LABELING STATS ===")
    print(f"Segments labeled this run : {labeled_count}")
    print(f"  contain a claim         : {claim_count}")
    print("Stance distribution:")
    total = sum(dist.values()) or 1
    for k, v in dist.items():
        print(f"  {k:8}: {v:4}  ({100*v/total:.0f}%)")
    print(f"\nLabeled files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
