"""
AGI-Discourse — Step 1: Transcript collection + baseline segmentation
====================================================================

What this does (the professor's "first job = data collection"):
  1. Takes a list of YouTube videos.
  2. Pulls their subtitles/transcripts (NO speech-to-text).
  3. Saves a clean structured JSON per video into ../data/raw/.
  4. Does a BASELINE segmentation (time-window based) so you immediately
     get numbers for the report (how many videos, how many segments).
  5. Prints dataset stats.

IMPORTANT — this is a STARTING POINT, not the final pipeline:
  - Baseline segmentation here is simple time-windowing. The professor
    wants SEMANTIC segmentation (via an LLM) later. That's the planned
    next step; the function `segment_baseline` is where you'll swap it in.
  - Speaker / claim / stance labeling is NOT here yet (that's the LLM
    labeling stage, next milestone). This script just gets you clean data.

Setup:
  pip install youtube-transcript-api

Run:
  python fetch_transcripts.py
"""

import json
import os
from pathlib import Path

# youtube-transcript-api is the cleanest tool for pulling captions directly.
# (The professor mentioned youtube-dl / yt-dlp — that works too and can
#  download .vtt subtitle files, but this library is simpler for transcripts.)
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except ImportError:
    raise SystemExit("Run:  pip install youtube-transcript-api")


# ---------------------------------------------------------------------------
# 1. SEED LIST — fill this in. These are the channels/people the professor
#    named in the kickoff meeting. Start with a handful, grow toward ~200.
#    You only need the 11-char video ID (the part after v= in the URL).
# ---------------------------------------------------------------------------

# Just put video IDs here. Example format:
#   "dQw4w9WgXcQ"  ->  from https://www.youtube.com/watch?v=dQw4w9WgXcQ
SEED_VIDEOS = [
    # "VIDEO_ID_1",
    # "VIDEO_ID_2",
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

# Channels to mine for video IDs (note for the report / your own reference):
SOURCE_CHANNELS = [
    "Lex Fridman Podcast",
    "Dwarkesh Patel",
    "Machine Learning Street Talk",
]

# Time window for the dataset (professor: 2022-2026, ~200 videos).
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
        # prefers manual captions, falls back to auto-generated
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
        return transcript
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
# 3. Baseline segmentation (PLACEHOLDER for semantic segmentation)
#    Groups caption lines into ~`window_sec` chunks. Good enough to get
#    segment counts for the progress report. Replace later with LLM-based
#    semantic segmentation.
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
                # fields to be filled by the LLM labeling stage (next milestone):
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
# 4. Run the whole thing + print stats for the report
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

        # baseline segmentation
        segs = segment_baseline(transcript)
        (SEG_DIR / f"{vid}.json").write_text(
            json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        dur = transcript[-1]["start"] + transcript[-1].get("duration", 0)
        total_videos += 1
        total_segments += len(segs)
        total_seconds += dur
        print(f"  -> {len(segs)} segments, ~{dur/60:.0f} min")

    # ---- numbers you can paste into the Dataset section of the report ----
    print("\n=== DATASET STATS (for the progress report) ===")
    print(f"Videos with transcripts : {total_videos}")
    print(f"Total discourse segments: {total_segments}")
    print(f"Total content           : {total_seconds/3600:.1f} hours")
    if total_videos:
        print(f"Avg segments per video  : {total_segments/total_videos:.0f}")


if __name__ == "__main__":
    main()
