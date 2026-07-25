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


def taxonomy_prompt_block():
    """A compact bullet list of the taxonomy for LLM labeling prompts."""
    return "\n".join(f"- {c['id']}: {c['description']}" for c in CLAIM_TAXONOMY)


if __name__ == "__main__":
    import json
    print(f"{len(CLAIM_TAXONOMY)} canonical claim categories:\n")
    print(taxonomy_prompt_block())
    # also dump a JSON copy next to the other graph artifacts for reference
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "data" / "claim_taxonomy.json"
    out.write_text(json.dumps(CLAIM_TAXONOMY, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
