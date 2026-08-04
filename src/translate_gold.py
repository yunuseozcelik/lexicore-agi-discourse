"""
Yardımcı — altın (gold) örneklemindeki İngilizce metinleri Türkçe'ye çevirir.

Etiketleyici İngilizce konuşma metnini rahat okuyamıyorsa, elle doğrulama zorlaşır.
Bu script data/gold/ içindeki her segmentin `text` alanını bir LLM ile Türkçe'ye
çevirip aynı segmente `text_tr` olarak yazar. label_manual.py bu alanı görünce
metni Türkçe gösterir. Sadece EKLER — orijinal İngilizce `text` korunur.

Ucuz ve hızlı olması için segmentler toplu (varsayılan 10'arlı) gönderilir ve
zaten çevrilmiş olanlar atlanır (tekrar çalıştırılabilir / kaldığı yerden devam).

Sağlayıcı seçilebilir (label_segments.py ile aynı istemciler). Gemini'nin ücretsiz
katmanı bu iş için yeterli ama dakika başına metrelidir, o yüzden --provider gemini
seçilince eşzamanlı istek sayısı düşük tutulur ve 429'da kota penceresi beklenir.

Kullanım:
  export GEMINI_API_KEY="..."          # ya da proje kökünde .env
  python src/translate_gold.py --provider gemini

  export OPENAI_API_KEY="..."
  python src/translate_gold.py          # varsayılan: openai
  python src/translate_gold.py --batch 15
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_segments import build_client  # aynı sağlayıcı kurulumunu paylaş

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
# label_segments.py ile aynı varsayılan modeller
DEFAULT_MODELS = {
    "openai": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
}
# Gemini'nin ücretsiz katmanı dakika başına metreli; paralelliği düşük tut.
DEFAULT_WORKERS = {"openai": 8, "gemini": 2}

SYS = ("Sen profesyonel bir çevirmensin. Sana İngilizce konuşma (transcript) "
       "parçalarından oluşan bir JSON listesi verilecek. Her parçayı akıcı, "
       "doğal Türkçe'ye çevir. Anlamı koru, ekleme/çıkarma yapma. SADECE şu "
       "biçimde JSON döndür: {\"translations\": [\"...\", \"...\"]} — giriş "
       "listesiyle aynı sırada ve aynı uzunlukta.")


def _call_openai(client, model, user):
    """40 sn zaman aşımı; askıda kalırsa hızlıca patlar ve tekrar denenir."""
    resp = client.with_options(timeout=40).chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": user}],
    )
    return json.loads(resp.choices[0].message.content)


def _call_gemini(client, model, user):
    """Gemini'de sistem yönergesi ayrı bir alan; şema ile JSON'a zorluyoruz."""
    from google.genai import types
    schema = {
        "type": "object",
        "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
        "required": ["translations"],
    }
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=SYS,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )
    return json.loads(resp.text)


def translate_batch(provider, client, model, texts, retries=5):
    """texts (list[str]) -> Türkçe list[str], aynı sırada."""
    import time
    call = _call_openai if provider == "openai" else _call_gemini
    user = json.dumps({"texts": texts}, ensure_ascii=False)
    last = None
    for attempt in range(retries):
        try:
            data = call(client, model, user)
            break
        except Exception as e:
            last = e
            msg = str(e)
            # Ücretsiz katmanlar dakika başına metreli — kota penceresi dolsun diye
            # 429'da uzun bekle (label_segments.py ile aynı yaklaşım).
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                time.sleep(15 * (attempt + 1))
            else:
                time.sleep(2 * (attempt + 1))
    else:
        raise last
    out = data.get("translations") or data.get("texts") or []
    if len(out) != len(texts):
        # hizalama bozulursa güvenli tarafta kal: eksikse boş bırak
        out = (out + [""] * len(texts))[:len(texts)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini"])
    ap.add_argument("--model", help="varsayılan: sağlayıcıya göre (DEFAULT_MODELS)")
    ap.add_argument("--batch", type=int, default=10, help="tek istekte segment sayısı")
    ap.add_argument("--workers", type=int,
                    help="eşzamanlı istek sayısı (varsayılan: openai 8, gemini 2)")
    args = ap.parse_args()

    if not GOLD_DIR.exists() or not any(GOLD_DIR.glob("*.json")):
        print("data/gold/ boş. Önce: python src/build_gold_sample.py")
        return

    model = args.model or DEFAULT_MODELS[args.provider]
    workers = args.workers or DEFAULT_WORKERS[args.provider]
    client = build_client(args.provider)

    # çevrilecek her (dosya, index, text) çiftini topla
    files = {}
    pending = []   # (video_id, idx, text)
    for f in sorted(GOLD_DIR.glob("*.json")):
        segs = json.loads(f.read_text(encoding="utf-8"))
        files[f.stem] = (f, segs)
        for i, s in enumerate(segs):
            txt = (s.get("text") or "").strip()
            if txt and not (s.get("text_tr") or "").strip():
                pending.append((f.stem, i, txt))

    if not pending:
        print("Her segment zaten çevrilmiş. Yapılacak iş yok.")
        return

    total = len(pending)
    print(f"Çevrilecek segment: {total} ({args.provider}:{model}, "
          f"batch: {args.batch}, paralel: {workers})", flush=True)

    # işi batch'lere böl, batch'leri paralel çevir (I/O-bound -> thread havuzu)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    chunks = [pending[i:i + args.batch] for i in range(0, total, args.batch)]
    lock = threading.Lock()
    done = {"n": 0}

    def work(chunk):
        texts = [t for (_, _, t) in chunk]
        try:
            trs = translate_batch(args.provider, client, model, texts)
        except Exception as e:
            print(f"  ! batch hata: {e} — atlanıyor", flush=True)
            return set()
        touched = set()
        with lock:
            for (vid, idx, _), tr in zip(chunk, trs):
                files[vid][1][idx]["text_tr"] = (tr or "").strip()
                touched.add(vid)
            done["n"] += len(chunk)
            # her tamamlanan batch'te dokunulan dosyayı kaydet (kesintiye dayanıklı)
            for vid in touched:
                f, segs = files[vid]
                f.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {done['n']}/{total} çevrildi", flush=True)
        return touched

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, c) for c in chunks]
        for _ in as_completed(futures):
            pass

    print("\nBitti. Şimdi arayüzü aç (metin Türkçe görünecek):")
    print("  streamlit run src/label_manual.py")


if __name__ == "__main__":
    main()
