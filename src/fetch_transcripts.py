"""
Step 1 — transcript collection + baseline segmentation.

  1. Takes a list of YouTube video IDs.
  2. Pulls their subtitles/transcripts (no speech-to-text).
  3. Saves a structured JSON per video into ../data/raw/.
  4. Runs a baseline time-window segmentation into ../data/segments/.
  5. Prints dataset stats (videos / segments / hours).

Baseline segmentation is simple time-windowing; semantic (LLM-based)
segmentation is planned and would replace `segment_baseline`. Speaker /
claim / stance labeling happens later in label_segments.py.

Usage:
  pip install youtube-transcript-api
  python fetch_transcripts.py
"""

import json
import os
from pathlib import Path

# youtube-transcript-api pulls captions directly. (yt-dlp can also download
# .vtt subtitle files, but this library is simpler for plain transcripts.)
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except ImportError:
    raise SystemExit("Run:  pip install youtube-transcript-api")

# youtube-transcript-api >= 1.0 changed the API: you instantiate the class and
# call .fetch(); the old static .get_transcript() was removed. We instantiate
# once and reuse it.
_API = YouTubeTranscriptApi()


# ---------------------------------------------------------------------------
# 1. SEED LIST — video IDs to collect. Start with a handful, grow toward ~200.
#    Only the 11-char video ID is needed (the part after v= in the URL),
#    e.g. "dQw4w9WgXcQ" from https://www.youtube.com/watch?v=dQw4w9WgXcQ
# ---------------------------------------------------------------------------
SEED_VIDEOS = [
    "vif8NQcjVf0",
    "b1TeeIG6Uaw",
    "WLBsUarvWTw",
    "F_7M4Hc-usM",
    "5t1vTLU7s40",  # Lex Fridman × Yann LeCun (AGI skeptic; strengthens Refute)
]

# People we care about (used later for speaker identification + the graph).
# Kept here so the whole team shares one canonical list.
TRACKED_PEOPLE = [
    "Jensen Huang",      # Nvidia
    "Demis Hassabis",    # Google DeepMind
    "Yann LeCun",        # Meta
    "Jeff Dean",         # Google
    "Mustafa Suleyman",  # Microsoft AI
    # add more as you find them (snowball from podcast guest lists)
]

# Channels to mine for video IDs.
SOURCE_CHANNELS = [
    "Lex Fridman Podcast",
    "Dwarkesh Patel",
    "Machine Learning Street Talk",
]

# Target time window for the dataset.
YEAR_RANGE = (2022, 2026)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SEG_DIR = DATA_DIR / "segments"


# ---------------------------------------------------------------------------
# 2. Pull one transcript
# ---------------------------------------------------------------------------
def fetch_one(video_id: str, languages=("en",)):
    """Return list of {text, start, duration} or None if unavailable."""
    try:
        # prefers manual captions, falls back to auto-generated.
        # .fetch() returns a FetchedTranscript; .to_raw_data() gives a list of
        # {text, start, duration} dicts (same shape the rest of the code expects).
        fetched = _API.fetch(video_id, languages=list(languages))
        return fetched.to_raw_data()
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"  [skip] no transcript for {video_id}")
        return None
    except VideoUnavailable:
        print(f"  [skip] video unavailable {video_id}")
        return None
    except Exception as e:  # network / misc
        print(f"  [error] {video_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Baseline segmentation (placeholder for semantic segmentation).
#    Groups caption lines into ~`window_sec` chunks. To be replaced later with
#    LLM-based semantic segmentation.
# ---------------------------------------------------------------------------
def segment_baseline(transcript, window_sec=90):
    segments = []
    if not transcript:
        return segments
    cur_text, cur_start = [], transcript[0]["start"]
    for line in transcript:
        cur_text.append(line["text"])
        if line["start"] - cur_start >= window_sec:
            segments.append({
                "start": round(cur_start, 1),
                "end": round(line["start"] + line.get("duration", 0), 1),
                "text": " ".join(cur_text).strip(),
                # filled in later by label_segments.py:
                "speaker": None,
                "claim": None,
                "stance": None,   # one of: Support / Refute / Neutral
            })
            cur_text, cur_start = [], line["start"]
    if cur_text:  # leftover
        segments.append({
            "start": round(cur_start, 1),
            "end": round(transcript[-1]["start"] + transcript[-1].get("duration", 0), 1),
            "text": " ".join(cur_text).strip(),
            "speaker": None, "claim": None, "stance": None,
        })
    return segments


# ---------------------------------------------------------------------------
# 3b. Semantic-ish segmentation (improved baseline; still no LLM).
#     Instead of cutting purely on elapsed time, it cuts at *natural boundaries*
#     — a long pause between caption lines (a strong signal of a speaker turn)
#     or a sentence end (.?!) — but only once a segment is long enough. This
#     avoids slicing through the middle of a sentence or a speaker's turn, which
#     is what makes the fixed-window baseline noisy (mixed speakers/topics in one
#     segment -> more "Unknown" speakers, blurrier stance).
#
#     Params:
#       min_sec    : don't cut before a segment reaches this length.
#       max_sec    : force a cut past this length even without a clean boundary.
#       pause_sec  : a gap >= this between two lines counts as a turn boundary.
# ---------------------------------------------------------------------------
def segment_semantic(transcript, min_sec=60, max_sec=180, pause_sec=1.5):
    segments = []
    if not transcript:
        return segments

    def flush(cur_text, cur_start, end):
        segments.append({
            "start": round(cur_start, 1),
            "end": round(end, 1),
            "text": " ".join(cur_text).strip(),
            # filled in later by label_segments.py:
            "speaker": None, "claim": None, "stance": None,
        })

    cur_text, cur_start = [], transcript[0]["start"]
    for i, line in enumerate(transcript):
        cur_text.append(line["text"])
        line_end = line["start"] + line.get("duration", 0)
        elapsed = line["start"] - cur_start

        # gap to the NEXT line (long gap == likely speaker turn / topic shift)
        nxt = transcript[i + 1] if i + 1 < len(transcript) else None
        gap = (nxt["start"] - line_end) if nxt else 0.0

        ends_sentence = line["text"].rstrip().endswith((".", "?", "!"))

        # cut when: long enough AND (a real pause OR a sentence end),
        # or when we've simply run past the hard max length.
        clean_boundary = elapsed >= min_sec and (gap >= pause_sec or ends_sentence)
        too_long = elapsed >= max_sec
        if clean_boundary or too_long:
            flush(cur_text, cur_start, line_end)
            cur_text = []
            cur_start = nxt["start"] if nxt else line_end

    if cur_text:  # leftover
        flush(cur_text, cur_start,
              transcript[-1]["start"] + transcript[-1].get("duration", 0))
    return segments


# ---------------------------------------------------------------------------
# 4. Run the pipeline and print dataset stats
# ---------------------------------------------------------------------------
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SEG_DIR.mkdir(parents=True, exist_ok=True)

    if not SEED_VIDEOS:
        print("SEED_VIDEOS is empty — add a few YouTube video IDs and re-run.")
        return

    total_videos, total_segments, total_seconds = 0, 0, 0.0

    for vid in SEED_VIDEOS:
        print(f"[fetch] {vid}")
        transcript = fetch_one(vid)
        if transcript is None:
            continue

        # save raw transcript
        (RAW_DIR / f"{vid}.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # semantic-ish segmentation (cuts at natural pauses / sentence ends);
        # segment_baseline() is kept for comparison but no longer the default.
        segs = segment_semantic(transcript)
        (SEG_DIR / f"{vid}.json").write_text(
            json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        dur = transcript[-1]["start"] + transcript[-1].get("duration", 0)
        total_videos += 1
        total_segments += len(segs)
        total_seconds += dur
        print(f"  -> {len(segs)} segments, ~{dur/60:.0f} min")

    print("\n=== DATASET STATS ===")
    print(f"Videos with transcripts : {total_videos}")
    print(f"Total discourse segments: {total_segments}")
    print(f"Total content           : {total_seconds/3600:.1f} hours")
    if total_videos:
        print(f"Avg segments per video  : {total_segments/total_videos:.0f}")


if __name__ == "__main__":
    main()
