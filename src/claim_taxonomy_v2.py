"""
Canonical claim taxonomy, v2 — 14 categories collapsed to 8.

WHY THIS EXISTS
---------------
Two independent measurements pointed at the same problem with the v1 taxonomy
(see PROGRESS.md sections 12 and 14):

  1. LLM-as-a-judge marked 239/307 graph edges "unclear" (only 4 "invalid").
     Per-category validity split sharply: categories with distinctive vocabulary
     scored 46-55%, abstract/broad ones scored 0-8% (`intelligence_nature` 0%,
     `open_source` 8%).
  2. The multi-label classifier's worst categories are the same ones —
     `intelligence_nature` F1 0.283 (43 examples), `geopolitics` 0.310 (38),
     `hardware_infra` 0.311 (140) — and BERT-base (110M) did not fix them,
     so this is not a model-capacity problem.

Diagnosis: several v1 categories are too abstract or too rare to be assigned
reliably, and the rare ones also starve the classifier. v2 merges them into
8 categories that are each (a) semantically distinct and (b) frequent enough
to train on.

NO RE-LABELING NEEDED
---------------------
V1_TO_V2 maps every v1 id onto a v2 id, so existing `claim_labels` can be
projected forward with `collapse_labels()`. That makes v1 vs v2 a controlled
comparison on identical data — the only thing that changes is label granularity.
A future LLM re-labeling pass can target v2 directly via taxonomy_prompt_block().

MERGE RATIONALE
---------------
  scaling_compute + hardware_infra      -> scaling_and_hardware
      Both are "does more compute get us there" — hardware_infra is largely the
      physical-substrate half of the same argument, and is too rare alone (140).
  ai_race_power + geopolitics           -> race_and_geopolitics
      Corporate race and national (US/China) race are the same competitive
      dynamic at different scales; geopolitics alone has only 38 examples.
  intelligence_nature -> llm_capabilities
      v1's 0%-validity category. "What is intelligence" and "what can these
      systems actually do" were never separable in practice on this corpus.
  breakthroughs_needed -> llm_capabilities
      "Current methods are not enough" is a claim about current-model limits.
  execution_engineering -> ai_economics  (renamed industry_and_economics)
      Both are about the practical/industrial reality of building and shipping
      AI rather than about the technology's trajectory.
  open_source -> ai_race_power branch    -> race_and_geopolitics
      In this corpus open-vs-closed is argued as a power-distribution question.

Unchanged: agi_timeline, ai_safety_risk, ai_optimism_benefit, other.
"""

CLAIM_TAXONOMY_V2 = [
    {
        "id": "agi_timeline",
        "name": "AGI Timeline",
        "description": "When AGI / human-level AI will arrive — whether it is near-term or far off.",
        "keywords": ["AGI", "timeline", "near-term", "decades", "imminent", "soon", "arrive", "human-level"],
    },
    {
        "id": "llm_capabilities",
        "name": "Capabilities, Limits & What's Missing",
        "description": "What current AI systems can and cannot do — reasoning, planning, world models, the nature of machine vs. human intelligence, and whether fundamentally new ideas are still needed.",
        "keywords": ["LLM", "reasoning", "planning", "cannot", "limitation", "world model", "agency",
                     "breakthrough", "new architecture", "paradigm", "missing piece", "beyond transformers",
                     "intelligence", "learning", "cognition", "brain"],
    },
    {
        "id": "scaling_and_hardware",
        "name": "Scaling, Compute & Hardware",
        "description": "Whether scaling models, data, and compute is the main path to progress — including the chips, GPUs, and infrastructure that make it possible.",
        "keywords": ["scaling", "scale", "compute", "emergent", "bigger models", "training", "data",
                     "hardware", "chip", "GPU", "NVIDIA", "infrastructure", "datacenter", "semiconductor"],
    },
    {
        "id": "ai_safety_risk",
        "name": "AI Safety & Risk",
        "description": "AI risk, safety, alignment, danger, or society over/under-reacting to it.",
        "keywords": ["safety", "risk", "danger", "alignment", "control", "overreact", "fear", "existential"],
    },
    {
        "id": "ai_optimism_benefit",
        "name": "AI Optimism & Human Benefit",
        "description": "Optimism that AI will amplify humanity, solve problems, and improve the future.",
        "keywords": ["optimism", "benefit", "amplify", "hope", "solve problems", "positive", "abundance"],
    },
    {
        "id": "industry_and_economics",
        "name": "Industry, Economics & Execution",
        "description": "The economic and practical reality of AI — markets, jobs, productivity, and the engineering work needed to actually build and ship systems.",
        "keywords": ["economy", "economic", "jobs", "labor", "market", "industry", "productivity", "GDP",
                     "execution", "engineering", "practical", "hard work", "build", "product", "ship"],
    },
    {
        "id": "race_and_geopolitics",
        "name": "Race, Power & Geopolitics",
        "description": "Competitive dynamics and concentration of power between AI actors — companies, open vs. proprietary platforms, and nations such as the US and China.",
        "keywords": ["race", "competition", "dominate", "power", "concentration", "monopoly",
                     "China", "US", "geopolitics", "national", "export", "chips ban", "government",
                     "open source", "open-source", "proprietary", "closed", "weights"],
    },
    {
        "id": "other",
        "name": "Other / Off-topic",
        "description": "No substantive AGI-related claim, off-topic, or small talk.",
        "keywords": ["off-topic", "intro", "welcome", "small talk", "question", "filler"],
    },
]

# v1 id -> v2 id. Every v1 category maps somewhere, so existing labels project
# forward without any re-labeling.
V1_TO_V2 = {
    "agi_timeline":          "agi_timeline",
    "llm_capabilities":      "llm_capabilities",
    "intelligence_nature":   "llm_capabilities",
    "breakthroughs_needed":  "llm_capabilities",
    "scaling_compute":       "scaling_and_hardware",
    "hardware_infra":        "scaling_and_hardware",
    "ai_safety_risk":        "ai_safety_risk",
    "ai_optimism_benefit":   "ai_optimism_benefit",
    "ai_economics":          "industry_and_economics",
    "execution_engineering": "industry_and_economics",
    "ai_race_power":         "race_and_geopolitics",
    "geopolitics":           "race_and_geopolitics",
    "open_source":           "race_and_geopolitics",
    "other":                 "other",
}

# Per-category proposition for category-conditional stance (see claim_taxonomy.py).
CLAIM_PROPOSITIONS_V2 = {
    "agi_timeline":           "AGI / human-level AI will arrive in the near term (within roughly the next several years).",
    "llm_capabilities":       "Current AI systems are already broadly capable — they can genuinely reason, plan, and model the world.",
    "scaling_and_hardware":   "Scaling models, data, compute, and hardware is the main path to further AI progress.",
    "ai_safety_risk":         "AI poses a serious safety / existential risk that society should take seriously.",
    "ai_optimism_benefit":    "AI will, on balance, greatly benefit humanity and the future is something to be optimistic about.",
    "industry_and_economics": "AI will drive large economic and industrial transformation — major effects on jobs, markets, and how systems get built.",
    "race_and_geopolitics":   "AI development is a high-stakes competitive race for power between companies and nations.",
    "other":                  None,
}

CLAIM_IDS_V2 = [c["id"] for c in CLAIM_TAXONOMY_V2]
ID2NAME_V2 = {c["id"]: c["name"] for c in CLAIM_TAXONOMY_V2}
ID2DESC_V2 = {c["id"]: c["description"] for c in CLAIM_TAXONOMY_V2}


def collapse_labels(labels):
    """Project a list of v1 claim ids onto v2 ids (deduplicated, v2 order).

    Unknown ids are dropped rather than raising, so partially-labeled or
    hand-edited data still loads.
    """
    mapped = {V1_TO_V2[l] for l in labels if l in V1_TO_V2}
    return [cid for cid in CLAIM_IDS_V2 if cid in mapped]


def taxonomy_prompt_block():
    """A compact bullet list of the v2 taxonomy for LLM labeling prompts."""
    return "\n".join(f"- {c['id']}: {c['description']}" for c in CLAIM_TAXONOMY_V2)


if __name__ == "__main__":
    import json
    from collections import Counter
    from pathlib import Path

    print(f"{len(CLAIM_TAXONOMY_V2)} canonical claim categories (v2):\n")
    print(taxonomy_prompt_block())

    # Report what the collapse does to the real label distribution.
    labeled = Path(__file__).resolve().parent.parent / "data" / "labeled"
    v1_counts, v2_counts, n = Counter(), Counter(), 0
    for f in sorted(labeled.glob("*.json")):
        for seg in json.loads(f.read_text(encoding="utf-8")):
            labs = seg.get("claim_labels")
            if not labs:
                continue
            n += 1
            v1_counts.update(labs)
            v2_counts.update(collapse_labels(labs))

    if n:
        print(f"\n--- label frequency over {n} segments ---")
        print("\nv1 (14 categories):")
        for cid, c in sorted(v1_counts.items(), key=lambda x: -x[1]):
            print(f"  {cid:<24} {c:>5}")
        print("\nv2 (8 categories):")
        for cid in CLAIM_IDS_V2:
            print(f"  {cid:<24} {v2_counts[cid]:>5}")
        rare1 = sum(1 for c in v1_counts.values() if c < 150)
        rare2 = sum(1 for cid in CLAIM_IDS_V2 if v2_counts[cid] < 150)
        print(f"\ncategories with <150 examples:  v1={rare1}  ->  v2={rare2}")

    out = Path(__file__).resolve().parent.parent / "data" / "claim_taxonomy_v2.json"
    out.write_text(json.dumps(CLAIM_TAXONOMY_V2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
