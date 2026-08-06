"""
Canonical claim taxonomy for multi-label claim classification.

The official project asks for *multi-label canonical claim classification*: each
segment can express zero, one, or several canonical claims. Rather than an open
set of raw claim paraphrases, we fix a small taxonomy of canonical claim
CATEGORIES that the AGI-discourse corpus revolves around. This taxonomy was
derived by inspecting the 20 embedding-based claim clusters
(data/graph/canonical_claims_embed.json) and merging them into ~14 coherent,
mutually-distinguishable themes.

Each category has:
  - id          : stable snake_case key used in the data + models
  - name        : short human-readable label (for the labeling UI / plots)
  - description : one sentence the annotator / LLM / embedding model scores against
  - keywords    : hint terms (used by the embedding bootstrap in assign_claim_labels.py)

Edit this list ONCE before the big labeling push — changing it afterwards means
re-labeling. Keep ids stable even if you rename `name`/`description`.
"""

CLAIM_TAXONOMY = [
    {
        "id": "agi_timeline",
        "name": "AGI Timeline",
        "description": "When AGI / human-level AI will arrive — whether it is near-term or far off.",
        "keywords": ["AGI", "timeline", "near-term", "decades", "imminent", "soon", "arrive", "human-level"],
    },
    {
        "id": "llm_capabilities",
        "name": "LLM Capabilities & Limits",
        "description": "What current LLMs/AI systems can and cannot do — reasoning, planning, world models, agency.",
        "keywords": ["LLM", "reasoning", "planning", "current models", "cannot", "limitation", "world model", "agency"],
    },
    {
        "id": "scaling_compute",
        "name": "Scaling & Compute",
        "description": "Whether scaling models and compute is the main path to progress in AI.",
        "keywords": ["scaling", "scale", "compute", "emergent", "bigger models", "training", "data"],
    },
    {
        "id": "breakthroughs_needed",
        "name": "New Breakthroughs Needed",
        "description": "Whether fundamentally new ideas or architectures are required beyond current methods.",
        "keywords": ["breakthrough", "new architecture", "paradigm", "missing piece", "not enough", "beyond transformers"],
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
        "id": "ai_economics",
        "name": "AI Economics & Jobs",
        "description": "Economic impact of AI — markets, industry structure, productivity, and jobs.",
        "keywords": ["economy", "economic", "jobs", "labor", "market", "industry", "productivity", "GDP"],
    },
    {
        "id": "ai_race_power",
        "name": "AI Race & Power Concentration",
        "description": "Competitive dynamics between AI actors and concentration of power / whether one AI dominates.",
        "keywords": ["race", "competition", "dominate", "power", "concentration", "monopoly", "countermeasure"],
    },
    {
        "id": "geopolitics",
        "name": "Geopolitics (US / China)",
        "description": "National / geopolitical AI competition, especially the US versus China.",
        "keywords": ["China", "US", "geopolitics", "national", "export", "chips ban", "ahead", "government"],
    },
    {
        "id": "open_source",
        "name": "Open Source vs Proprietary",
        "description": "Whether AI development should rely on open-source or closed/proprietary platforms.",
        "keywords": ["open source", "open-source", "proprietary", "closed", "weights", "platform"],
    },
    {
        "id": "intelligence_nature",
        "name": "Nature of Intelligence & Learning",
        "description": "The nature of intelligence and learning — human vs. animal vs. machine, culture, language.",
        "keywords": ["intelligence", "learning", "human", "animal", "culture", "language", "brain", "cognition"],
    },
    {
        "id": "execution_engineering",
        "name": "Execution & Engineering Reality",
        "description": "The practical engineering work and commitment needed to actually build AI, beyond ideas.",
        "keywords": ["execution", "engineering", "practical", "hard work", "plan", "commitment", "build", "product"],
    },
    {
        "id": "hardware_infra",
        "name": "Hardware & Infrastructure",
        "description": "AI hardware, chips, GPUs, NVIDIA, and compute infrastructure growth.",
        "keywords": ["hardware", "chip", "GPU", "NVIDIA", "infrastructure", "datacenter", "semiconductor"],
    },
    {
        "id": "other",
        "name": "Other / Off-topic",
        "description": "No substantive AGI-related claim, off-topic, or small talk.",
        "keywords": ["off-topic", "intro", "welcome", "small talk", "question", "filler"],
    },
]

# Convenience lookups
CLAIM_IDS = [c["id"] for c in CLAIM_TAXONOMY]
ID2NAME = {c["id"]: c["name"] for c in CLAIM_TAXONOMY}
ID2DESC = {c["id"]: c["description"] for c in CLAIM_TAXONOMY}

# Category-conditional stance targets. A single global anchor makes stance
# ambiguous ("advanced AI is risky" supports the safety-risk claim but refutes
# blunt optimism). Instead each category carries its own proposition, and a
# speaker's stance is scored against the proposition of the category they are
# actually discussing: Support = agrees with it, Refute = disagrees, Neutral =
# mentions it without taking a side. `other` has no proposition (always Neutral).
CLAIM_PROPOSITIONS = {
    "agi_timeline":         "AGI / human-level AI will arrive in the near term (within roughly the next several years).",
    "llm_capabilities":     "Current AI systems are already broadly capable — they can genuinely reason, plan, and model the world.",
    "scaling_compute":      "Scaling models, data, and compute is the main path to further AI progress.",
    "breakthroughs_needed": "Fundamentally new ideas or architectures are still needed; today's methods are not enough to reach AGI.",
    "ai_safety_risk":       "AI poses a serious safety / existential risk that society should take seriously.",
    "ai_optimism_benefit":  "AI will, on balance, greatly benefit humanity and the future is something to be optimistic about.",
    "ai_economics":         "AI will drive large economic transformation — major effects on jobs, industry, and productivity.",
    "ai_race_power":        "AI development is a dangerous race that concentrates power in a few actors.",
    "geopolitics":          "National AI competition (especially the US vs. China) is decisive for how AI unfolds.",
    "open_source":          "AI should be developed openly (open-source / open weights) rather than closed and proprietary.",
    "intelligence_nature":  "Machine intelligence is fundamentally the same kind of thing as human / biological intelligence.",
    "execution_engineering":"Practical engineering and execution — not new ideas — is the real bottleneck to building AI.",
    "hardware_infra":       "Hardware and compute infrastructure (chips, GPUs, datacenters) is a critical driver of AI progress.",
    "other":                None,
}


def taxonomy_prompt_block():
    """A compact bullet list of the taxonomy for LLM labeling prompts."""
    return "\n".join(f"- {c['id']}: {c['description']}" for c in CLAIM_TAXONOMY)


def propositions_prompt_block(ids=None):
    """Bullet list of `id: proposition` for the given categories (default: all
    with a proposition), used by the category-conditional stance labeler."""
    ids = ids if ids is not None else CLAIM_IDS
    return "\n".join(f"- {cid}: {CLAIM_PROPOSITIONS[cid]}"
                     for cid in ids if CLAIM_PROPOSITIONS.get(cid))


if __name__ == "__main__":
    import json
    print(f"{len(CLAIM_TAXONOMY)} canonical claim categories:\n")
    print(taxonomy_prompt_block())
    # also dump a JSON copy next to the other graph artifacts for reference
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "data" / "claim_taxonomy.json"
    out.write_text(json.dumps(CLAIM_TAXONOMY, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
