"""
Step 3 — manual gold labeling (human-in-the-loop over silver labels).

A Streamlit app for building a GOLD (human-verified) label set. Instead of
labeling from scratch, it loads the LLM "silver" labels from data/labeled/ and
shows one segment at a time with those labels PRE-FILLED. The annotator just
confirms or corrects speaker / is_claim / claim / stance / topic, which is
3-5x faster than blank labeling and gives us:
  - a gold test set (real evaluation, not silver-on-silver), and
  - LLM-vs-human agreement (Cohen's kappa, see compute_agreement.py).

Corrected segments get "reviewed": true and are written to data/gold/<video>.json,
preserving the original silver value under "*_silver" so we can measure agreement.

Recommended workflow:
  1. fetch_transcripts.py   -> data/segments/
  2. label_segments.py      -> data/labeled/   (silver, LLM)
  3. label_manual.py (this) -> data/gold/      (human-verified subset)

Because hand-labeling all ~2200 segments is too much, review a stratified
SAMPLE (~400-500) as the gold set; the rest stay silver for BERT training.

Usage:
  pip install streamlit
  streamlit run src/label_manual.py
"""

import json
from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SILVER_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
STANCES = ["Support", "Refute", "Neutral"]

# Fields the annotator can edit, with the silver value snapshotted alongside.
EDITABLE = ["speaker", "is_claim", "claim", "stance", "topic"]


def list_videos():
    return sorted(f.stem for f in SILVER_DIR.glob("*.json"))


def load_segments(video_id):
    """Load gold if it exists (resume), else start from the silver copy."""
    gold_path = GOLD_DIR / f"{video_id}.json"
    if gold_path.exists():
        return json.loads(gold_path.read_text(encoding="utf-8"))
    segs = json.loads((SILVER_DIR / f"{video_id}.json").read_text(encoding="utf-8"))
    # snapshot the silver value so agreement can be computed later
    for s in segs:
        for field in EDITABLE:
            s.setdefault(f"{field}_silver", s.get(field))
        s.setdefault("reviewed", False)
    return segs


def save_segments(video_id, segs):
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    (GOLD_DIR / f"{video_id}.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")


def first_unreviewed(segs):
    for i, s in enumerate(segs):
        if not s.get("reviewed"):
            return i
    return 0


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LexiCore — Gold Labeling", layout="wide")
st.title("🏷️ LexiCore — Manual Gold Labeling")

videos = list_videos()
if not videos:
    st.error("No silver-labeled files in data/labeled/. Run label_segments.py first.")
    st.stop()

# --- sidebar: pick video, show progress ---
with st.sidebar:
    st.header("Video")
    video_id = st.selectbox("Video ID", videos, key="video_select")

# (re)load segments when the selected video changes
if st.session_state.get("_loaded_video") != video_id:
    st.session_state.segs = load_segments(video_id)
    st.session_state.idx = first_unreviewed(st.session_state.segs)
    st.session_state._loaded_video = video_id

segs = st.session_state.segs
n = len(segs)
idx = st.session_state.idx
reviewed = sum(1 for s in segs if s.get("reviewed"))

with st.sidebar:
    st.metric("Reviewed", f"{reviewed} / {n}")
    st.progress(reviewed / n if n else 0.0)
    jump = st.number_input("Jump to segment #", 1, n, idx + 1)
    if st.button("Go"):
        st.session_state.idx = int(jump) - 1
        st.rerun()
    st.caption("Silver = LLM label (pre-filled). Fix it, then Save & Next.")

# --- main: current segment ---
seg = segs[idx]
st.subheader(f"Segment {idx + 1} / {n}   ·   t={seg.get('start')}s–{seg.get('end')}s"
             f"   {'✅ reviewed' if seg.get('reviewed') else '⬜ not reviewed'}")

st.text_area("Transcript", seg.get("text", ""), height=220, disabled=True,
             key=f"txt_{video_id}_{idx}")

col1, col2 = st.columns(2)
with col1:
    speaker = st.text_input("Speaker", seg.get("speaker") or "",
                            key=f"sp_{video_id}_{idx}")
    is_claim = st.checkbox("Contains a claim (is_claim)",
                           value=bool(seg.get("is_claim")),
                           key=f"ic_{video_id}_{idx}")
    topic = st.text_input("Topic", seg.get("topic") or "",
                          key=f"tp_{video_id}_{idx}")
with col2:
    cur_stance = seg.get("stance") if seg.get("stance") in STANCES else "Neutral"
    stance = st.radio("Stance (toward the anchor claim)", STANCES,
                      index=STANCES.index(cur_stance),
                      key=f"st_{video_id}_{idx}", horizontal=True)
    claim = st.text_area("Claim (one-sentence paraphrase)", seg.get("claim") or "",
                         height=120, key=f"cl_{video_id}_{idx}")

# show what the LLM originally said, so the annotator sees the correction
silver_bits = " · ".join(
    f"{f}={seg.get(f'{f}_silver')}" for f in ["speaker", "stance", "is_claim"])
st.caption(f"🤖 silver: {silver_bits}")


def apply_edits():
    seg["speaker"] = speaker.strip() or "Unknown"
    seg["is_claim"] = is_claim
    seg["claim"] = claim.strip() or None
    seg["stance"] = stance
    seg["topic"] = topic.strip() or None
    seg["reviewed"] = True


nav1, nav2, nav3, _ = st.columns([1, 1, 1, 3])
with nav1:
    if st.button("⬅ Prev", use_container_width=True):
        st.session_state.idx = max(0, idx - 1)
        st.rerun()
with nav2:
    if st.button("💾 Save & Next", type="primary", use_container_width=True):
        apply_edits()
        save_segments(video_id, segs)
        st.session_state.idx = min(n - 1, idx + 1)
        st.rerun()
with nav3:
    if st.button("Next ➡ (no save)", use_container_width=True):
        st.session_state.idx = min(n - 1, idx + 1)
        st.rerun()
