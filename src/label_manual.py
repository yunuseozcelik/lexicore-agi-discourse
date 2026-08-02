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
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_taxonomy import CLAIM_IDS, ID2NAME

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SILVER_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
STANCES = ["Support", "Refute", "Neutral"]

# Fields the annotator can edit, with the silver value snapshotted alongside.
EDITABLE = ["speaker", "is_claim", "claim", "claim_labels", "stance", "topic"]


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
# App  (arayüz Türkçe — etiketler İngilizce değerlerle saklanır)
# ---------------------------------------------------------------------------
# Duruş değerleri veri setinde İngilizce saklanır; ekranda Türkçe gösteririz.
STANCE_TR = {
    "Support": "✅ Destekliyor / iyimser",
    "Refute":  "❌ Karşı çıkıyor / endişeli",
    "Neutral": "➖ Tarafsız / net duruş yok",
}

st.set_page_config(page_title="LexiCore — Altın Etiketleme", layout="wide")
st.title("🏷️ LexiCore — Elle Altın Etiketleme")

videos = list_videos()
if not videos:
    st.error("data/labeled/ içinde etiketli dosya yok. Önce label_segments.py çalıştır.")
    st.stop()

# --- kısa kullanım kılavuzu (her zaman görünür) ---
with st.expander("ℹ️ Nasıl yapılır? (bir kez oku, 30 saniye)", expanded=False):
    st.markdown("""
**Amaç:** LLM'in otomatik tahminlerini (silver) insan gözüyle **onaylamak veya düzeltmek**.
Alanlar zaten dolu gelir — çoğu doğrudur, çoğunlukla sadece **Kaydet ve Sonraki**'ye basarsın.

- **Konuşmacı:** Bu metni kim söylüyor? (yanlışsa düzelt, bilinmiyorsa boş bırak)
- **İddia içeriyor mu?:** Bir görüş/sav var mı, yoksa sadece sohbet/soru mu?
- **Duruş:** Konuşmacı AGI'a karşı ne hissediyor?
  - **Destekliyor** = iyimser, olumlu, "yakında/iyi olacak"
  - **Karşı çıkıyor** = endişeli, kötümser, "riskli/tehlikeli"
  - **Tarafsız** = net taraf yok, betimleyici konuşuyor
- **Konu / İddia özeti / Kategoriler:** yanlışsa düzelt, değilse dokunma.

Bitince yeni terminalde `python src/compute_agreement.py` çalıştır → uyum skoru (κ).
""")

# --- kenar çubuğu: video seç, ilerlemeyi göster ---
with st.sidebar:
    st.header("Video")
    video_id = st.selectbox("Video seç", videos, key="video_select")

# (yeniden) yükle: seçilen video değişince
if st.session_state.get("_loaded_video") != video_id:
    st.session_state.segs = load_segments(video_id)
    st.session_state.idx = first_unreviewed(st.session_state.segs)
    st.session_state._loaded_video = video_id

segs = st.session_state.segs
n = len(segs)
idx = st.session_state.idx
reviewed = sum(1 for s in segs if s.get("reviewed"))

with st.sidebar:
    st.metric("İncelenen", f"{reviewed} / {n}")
    st.progress(reviewed / n if n else 0.0)
    jump = st.number_input("Segmente atla (no)", 1, n, idx + 1)
    if st.button("Git"):
        st.session_state.idx = int(jump) - 1
        st.rerun()
    st.caption("🤖 = LLM'in tahmini (dolu gelir). Yanlışsa düzelt, sonra Kaydet ve Sonraki.")

# --- ana ekran: mevcut segment ---
seg = segs[idx]
durum = "✅ incelendi" if seg.get("reviewed") else "⬜ henüz incelenmedi"
st.subheader(f"Segment {idx + 1} / {n}   ·   t={seg.get('start')}s–{seg.get('end')}s   {durum}")

text_tr = (seg.get("text_tr") or "").strip()
if text_tr:
    st.text_area("📝 Konuşma metni (Türkçe çeviri)", text_tr,
                 height=200, disabled=True, key=f"txttr_{video_id}_{idx}")
    with st.expander("🇬🇧 İngilizce orijinali göster"):
        st.write(seg.get("text", ""))
else:
    st.text_area("📝 Konuşma metni (İngilizce — kaynak video)", seg.get("text", ""),
                 height=220, disabled=True, key=f"txt_{video_id}_{idx}")
    st.caption("💡 Türkçe çeviri için: python src/translate_gold.py çalıştır.")

col1, col2 = st.columns(2)
with col1:
    speaker = st.text_input("👤 Konuşmacı", seg.get("speaker") or "",
                            key=f"sp_{video_id}_{idx}")
    is_claim = st.checkbox("💬 Bir iddia/görüş içeriyor mu?",
                           value=bool(seg.get("is_claim")),
                           key=f"ic_{video_id}_{idx}")
    topic = st.text_input("🏷️ Konu", seg.get("topic") or "",
                          key=f"tp_{video_id}_{idx}")
with col2:
    cur_stance = seg.get("stance") if seg.get("stance") in STANCES else "Neutral"
    stance = st.radio("⚖️ Duruş (AGI iddiasına karşı)", STANCES,
                      index=STANCES.index(cur_stance),
                      format_func=lambda s: STANCE_TR.get(s, s),
                      key=f"st_{video_id}_{idx}")
    claim = st.text_area("✍️ İddia özeti (tek cümle)", seg.get("claim") or "",
                         height=100, key=f"cl_{video_id}_{idx}")

# çoklu-etiket kanonik iddia kategorileri (tam genişlik)
cur_labels = [c for c in (seg.get("claim_labels") or []) if c in CLAIM_IDS]
claim_labels = st.multiselect(
    "📚 Kanonik iddia kategorileri (birden çok seçilebilir)", options=CLAIM_IDS,
    default=cur_labels, format_func=lambda i: f"{ID2NAME[i]} ({i})",
    key=f"cll_{video_id}_{idx}")

# LLM'in ilk tahminini göster ki düzeltme farkı görünsün
sp_s = seg.get("speaker_silver")
st_s = STANCE_TR.get(seg.get("stance_silver"), seg.get("stance_silver"))
ic_s = "evet" if seg.get("is_claim_silver") else "hayır"
st.caption(f"🤖 LLM tahmini →  Konuşmacı: {sp_s}  ·  Duruş: {st_s}  ·  İddia mı: {ic_s}")


def apply_edits():
    seg["speaker"] = speaker.strip() or "Unknown"
    seg["is_claim"] = is_claim
    seg["claim"] = claim.strip() or None
    seg["claim_labels"] = claim_labels
    seg["stance"] = stance
    seg["topic"] = topic.strip() or None
    seg["reviewed"] = True


nav1, nav2, nav3, _ = st.columns([1, 1.4, 1, 3])
with nav1:
    if st.button("⬅ Önceki", use_container_width=True):
        st.session_state.idx = max(0, idx - 1)
        st.rerun()
with nav2:
    if st.button("💾 Kaydet ve Sonraki", type="primary", use_container_width=True):
        apply_edits()
        save_segments(video_id, segs)
        st.session_state.idx = min(n - 1, idx + 1)
        st.rerun()
with nav3:
    if st.button("Atla ➡ (kaydetme)", use_container_width=True):
        st.session_state.idx = min(n - 1, idx + 1)
        st.rerun()
