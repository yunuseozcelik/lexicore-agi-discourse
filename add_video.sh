#!/usr/bin/env bash
# Tek bir videoyu uçtan uca ekler: transcript çek → segmentle → LLM ile etiketle.
#
# Kullanım:
#   ./add_video.sh VIDEO_ID          # çek + segmentle + etiketle
#   ./add_video.sh VIDEO_ID --fetch  # sadece çek + segmentle (etiketleme yok)
#
# Tek tek çalıştırmak YouTube'un IP rate-limit'ine takılmamak için tercih edilir.
# Her adım idempotent: yarıda kalırsa aynı komut kaldığı yerden devam eder.

set -euo pipefail
cd "$(dirname "$0")"

# "--" ayırıcısını yut, böylece tire ile başlayan ID'ler de verilebilir
# (ör. ./add_video.sh -- -HzgcbRXUK8).
if [ "${1:-}" = "--" ]; then shift; fi

VID="${1:-}"
if [ -z "$VID" ]; then
  echo "Kullanım: ./add_video.sh VIDEO_ID [--fetch]"
  echo "Tire ile başlayan ID için: ./add_video.sh -- -VIDEO_ID"
  exit 1
fi
PY=.venv/bin/python

echo "=== [1/2] Transcript + segmentasyon: $VID ==="
$PY - "$VID" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from youtube_transcript_api import YouTubeTranscriptApi
import fetch_transcripts as ft

vid = sys.argv[1]
raw_p = Path("data/raw") / f"{vid}.json"
seg_p = Path("data/segments") / f"{vid}.json"

if raw_p.exists():
    print(f"  transcript zaten var, atlanıyor ({raw_p})")
    tr = json.loads(raw_p.read_text(encoding="utf-8"))
else:
    fetched = YouTubeTranscriptApi().fetch(vid)
    tr = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
    raw_p.write_text(json.dumps(tr, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  çekildi: {len(tr)} altyazı satırı")

# Otomatik (ASR) altyazılarda noktalama ve duraklama yok: segment_semantic'in
# iki kesim sinyali de tutmaz, her segment max_sec'e dayanır (~180sn) ve video
# az sayıda çok uzun segmente bölünür. Böyle transcript'lerde eşikleri sıkıyoruz
# ki segment uzunluğu insan altyazılı videolardaki ~67sn'ye yakın kalsın.
punct_ratio = sum(
    1 for l in tr if l["text"].rstrip().endswith((".", "?", "!"))
) / len(tr)
if punct_ratio < 0.10:
    segs = ft.segment_semantic(tr, min_sec=60, max_sec=90, pause_sec=0.4)
    mode = f"ASR altyazı (noktalama %{punct_ratio*100:.0f}) -> max=90 pause=0.4"
else:
    segs = ft.segment_semantic(tr)  # min=60: PROGRESS.md'deki en iyi ablasyon
    mode = f"normal altyazı (noktalama %{punct_ratio*100:.0f})"

seg_p.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
mins = (tr[-1]["start"] + tr[-1]["duration"]) / 60
avg = sum(s["end"] - s["start"] for s in segs) / len(segs)
print(f"  {mode}")
print(f"  süre {mins:.0f}dk -> {len(segs)} segment (ort {avg:.0f}sn)")

# etiketlemeden önce konuşmacıların kim olduğunu görebilmek için ilk metin
print("\n  --- ilk 300 karakter (konuşmacı tespiti için) ---")
print("  " + " ".join(s["text"] for s in tr[:40])[:300].replace("\n", " "))
PYEOF

if [ "${2:-}" = "--fetch" ]; then
  echo ""
  echo "Sadece çekme istendi. Etiketlemek için:"
  echo "  ./add_video.sh $VID"
  exit 0
fi

echo ""
echo "=== [2/2] LLM etiketleme (gpt-4o-mini): $VID ==="
# --videos=ID biçimi şart: boşluklu biçimde tire ile başlayan ID'yi argparse
# flag sanıp "unrecognized arguments" veriyor.
$PY src/label_segments.py --videos="$VID" --provider openai

echo ""
echo "=== BİTTİ: $VID ==="
$PY - "$VID" <<'PYEOF'
import json, sys, collections
from pathlib import Path
vid = sys.argv[1]
p = Path("data/labeled") / f"{vid}.json"
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    st = collections.Counter(s.get("stance") for s in d)
    sp = collections.Counter(s.get("speaker") for s in d)
    print(f"  {len(d)} segment | stance: {dict(st)}")
    print(f"  konuşmacılar: {', '.join(f'{k}:{v}' for k, v in sp.most_common(5))}")
PYEOF
