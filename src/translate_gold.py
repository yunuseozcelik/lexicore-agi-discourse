"""
Yardımcı — altın (gold) örneklemindeki İngilizce metinleri Türkçe'ye çevirir.

Etiketleyici İngilizce konuşma metnini rahat okuyamıyorsa, elle doğrulama zorlaşır.
Bu script data/gold/ içindeki her segmentin `text` alanını gpt-4o-mini ile Türkçe'ye
çevirip aynı segmente `text_tr` olarak yazar. label_manual.py bu alanı görünce
metni Türkçe gösterir. Sadece EKLER — orijinal İngilizce `text` korunur.

Ucuz ve hızlı olması için segmentler toplu (varsayılan 10'arlı) gönderilir ve
zaten çevrilmiş olanlar atlanır (tekrar çalıştırılabilir / kaldığı yerden devam).

Kullanım:
  export OPENAI_API_KEY="..."         # ya da proje kökünde .env
  python src/translate_gold.py         # data/gold/ hepsini çevir
  python src/translate_gold.py --batch 15
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_segments import build_client  # aynı OpenAI kurulumunu paylaş

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYS = ("Sen profesyonel bir çevirmensin. Sana İngilizce konuşma (transcript) "
       "parçalarından oluşan bir JSON listesi verilecek. Her parçayı akıcı, "
       "doğal Türkçe'ye çevir. Anlamı koru, ekleme/çıkarma yapma. SADECE şu "
       "biçimde JSON döndür: {\"translations\": [\"...\", \"...\"]} — giriş "
       "listesiyle aynı sırada ve aynı uzunlukta.")


def translate_batch(client, texts, retries=3):
    """texts (list[str]) -> Türkçe list[str], aynı sırada.

    Her istekte 40 sn zaman aşımı; askıda kalırsa hızlıca patlayıp tekrar dener.
    """
    import time
    user = json.dumps({"texts": texts}, ensure_ascii=False)
    last = None
    for attempt in range(retries):
        try:
            resp = client.with_options(timeout=40).chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}],
            )
            break
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    else:
        raise last
    data = json.loads(resp.choices[0].message.content)
    out = data.get("translations") or data.get("texts") or []
    if len(out) != len(texts):
        # hizalama bozulursa güvenli tarafta kal: eksikse boş bırak
        out = (out + [""] * len(texts))[:len(texts)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10, help="tek istekte segment sayısı")
    ap.add_argument("--workers", type=int, default=8, help="eşzamanlı istek sayısı")
    args = ap.parse_args()

    if not GOLD_DIR.exists() or not any(GOLD_DIR.glob("*.json")):
        print("data/gold/ boş. Önce: python src/build_gold_sample.py")
        return

    client = build_client("openai")

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
    print(f"Çevrilecek segment: {total} (model: {MODEL}, batch: {args.batch}, "
          f"paralel: {args.workers})", flush=True)

    # işi batch'lere böl, batch'leri paralel çevir (I/O-bound -> thread havuzu)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    chunks = [pending[i:i + args.batch] for i in range(0, total, args.batch)]
    lock = threading.Lock()
    done = {"n": 0}

    def work(chunk):
        texts = [t for (_, _, t) in chunk]
        try:
            trs = translate_batch(client, texts)
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

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, c) for c in chunks]
        for _ in as_completed(futures):
            pass

    print("\nBitti. Şimdi arayüzü aç (metin Türkçe görünecek):")
    print("  streamlit run src/label_manual.py")


if __name__ == "__main__":
    main()
