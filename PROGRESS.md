# PROGRESS — LexiCore AGI-Discourse

Bu dosya, progress report sonrası bu sprintte yapılan işleri **adım adım** kaydeder.
Amaç: sprint raporunu / final paper'ı yazarken buradaki sıralı geçmişten ve
sonuçlardan faydalanmak. Her adımda "ilk durum → ne yaptık → ne oldu" var.

---

## 0. Başlangıç durumu (progress report anı)

- **4 video**, **302 segment** (90 saniyelik zaman-penceresi segmentasyonu).
- Etiketleme: zero-shot LLM (speaker + claim + stance), tümü "silver" etiket.
- Baseline: TF-IDF + Logistic Regression, **stance Macro-F1 = 0.507**.
- Knowledge graph: sadece **"planned"** (kod yoktu).
- Claim classification: sadece görev olarak tanımlıydı, sonucu yoktu.

---

## 1. Knowledge graph — "planned" → çalışan bileşen

**İlk durum:** Graf sadece plandı, kodu yoktu.

**Ne yaptık:** `src/build_graph.py` yazıldı. Etiketli segmentlerden bir
**person–stance grafı** kuruyor:
- Her konuşmacı bir **person düğümü**, tek bir **anchor claim düğümü**.
- Her `(konuşmacı, stance)` ikilisi, segment sayısıyla ağırlıklı bir **kenar**.
- "Unknown" konuşmacılar grafın dışında bırakılıyor (graf temiz kalsın).
- Teknoloji: Python + `networkx` + `matplotlib`.
- Çıktı: `data/graph/person_stance_graph.json` (node/edge) + `data/graph/graph.png`.

**Ne oldu:** İlk çalışan graf elde edildi. Sam Altman baskın Support,
Yann LeCun / Ege Erdil baskın Refute — kişilerin bilinen görüşleriyle tutarlı.

---

## 2. Claim classification — ikinci NLU görevi için baseline

**İlk durum:** Sadece stance görevinin skoru vardı.

**Ne yaptık:** `src/train_claim_baseline.py` yazıldı. `is_claim` etiketini
kullanarak claim / no-claim ikili sınıflandırması yapıyor (stance baseline'ı ile
aynı TF-IDF + Logistic Regression yapısı).

**Ne oldu:** Claim classification için ilk baseline: **Macro-F1 = 0.749**.

---

## 3. Veri genişletme — 5. video (hedefli)

**İlk durum:** 4 video / 302 segment. En zayıf sınıf **Refute** (58 örnek, F1 0.378).

**Ne yaptık:** En az temsil edilen Refute sınıfını güçlendirmek için **bilinçli
olarak** bir AGI-skeptik konuşmacı seçtik: **Yann LeCun** (Lex Fridman Podcast,
~167 dk). Altyazı `youtube-transcript-api` ile çekilip segmentlere bölündü ve
zero-shot LLM (OpenAI gpt-4o-mini) ile etiketlendi.

**Ne oldu:** Veri seti **5 video / 412 segment**'e çıktı. **Refute 58 → 115**
(neredeyse 2 katı). Refute F1 **0.378 → 0.496** yükseldi. Hedefli genişletme
işe yaradı.

---

## 4. Semantic segmentation — 90sn zaman-penceresi → anlam-tabanlı bölme

**İlk durum:** Segmentler saf zaman-penceresiyle (90sn) kesiliyordu. Sorun:
segmentlerin sadece **%45'i** düzgün cümle sonunda bitiyordu; birçoğu cümle/
konuşmacı ortasından kesiliyordu → karışık segment → yüksek "Unknown" ve
bulanık stance.

**Ne yaptık:** `segment_semantic()` fonksiyonu eklendi (`src/fetch_transcripts.py`).
Mantık: bir segment **min süreye** ulaşınca ve bir **doğal kesim noktası**
(uzun duraklama = konuşmacı/konu değişimi sinyali VEYA cümle sonu `.?!`) gelince
böl. Yani "zaman zemin, anlam kesim noktası". Eski `segment_baseline()`
karşılaştırma için korundu.

**Ne oldu (parametre ablasyonu):**

| Sürüm | Segment | Cümle-sonu temizliği | Stance F1 | Claim F1 | Neutral | Unknown |
|---|---|---|---|---|---|---|
| 90sn baseline | 412 | 45% | 0.519 | 0.749 | 46% | 22% |
| semantic min=30 | 961 | 95% | 0.503 | 0.749 | 65% | 23% |
| **semantic min=60** | **546** | ~90% | **0.562** | **0.782** | 53% | **16%** |

- **min=30 çok kısaydı:** bağlam parçalandı, Neutral %65'e fırladı, skor düştü.
  Ders: "daha temiz her zaman daha iyi değil" — segment uzunluğu kritik parametre.
- **off-topic filtreleme denendi, işe yaramadı:** off-topic'leri baseline'dan
  çıkarınca skor düştü (0.519 → 0.457), çünkü onlar modelin kolay Neutral
  örnekleriydi. (Negatif sonuç da bir bulgu.)
- **min=60 en iyi sürüm:** her metrik iyileşti — en yüksek stance (0.562) ve
  claim (0.782) skoru, en az Unknown (%16), dengeli Neutral.

---

## 5. Ara durum (bölüm 1-4'ün sonu — güncel rakamlar için bölüm 10'a bakınız)

- **5 video**, **546 semantic segment** (min=60, ~67 sn ortalama), tümü etiketli.
- **Stance Macro-F1 = 0.562** | **Claim Macro-F1 = 0.782** (5-fold CV).
- Knowledge graph: **9 kişi düğümü + 1 anchor**, 23 ağırlıklı stance kenarı.
- Kod artık `main`'e merge edildi (eski `eylul` ve `claim-clustering-embedding`
  branch'leri); API anahtarları `.gitignore` ile korunuyor.

### Stance dağılımı (546 segment)

| Sınıf | Sayı | Oran |
|---|---|---|
| Neutral | 292 | 53% |
| Support | 128 | 23% |
| Refute | 126 | 23% |

### Knowledge graph (son hali)

![Person–Stance Knowledge Graph](data/graph/graph.png)

*Yeşil: Support, Kırmızı: Refute, Gri: Neutral; kenar kalınlığı = segment sayısı.*
*Not: kenarlar üst üste bindiği için görselde stance ayrımı her zaman net
okunmuyor (mini graf sınırı); baskın stance için aşağıdaki tabloya bakınız.*

### Konuşmacı bazlı baskın duruş

| Konuşmacı | Segment | Support | Refute | Neutral | Baskın |
|---|---|---|---|---|---|
| Ege Erdil | 130 | 35 | 44 | 51 | Neutral |
| Yann LeCun | 108 | 22 | 44 | 42 | **Refute** |
| Jensen Huang | 84 | 32 | 5 | 47 | Neutral |
| Lex Fridman | 49 | 9 | 11 | 29 | Neutral |
| Victor Shih | 48 | 3 | 8 | 37 | Neutral |
| Sam Altman | 21 | 14 | 2 | 5 | **Support** |
| Tamay Besiroglu | 11 | 5 | 4 | 2 | **Support** |

---

## 6. Kanonik claim clustering — tek anchor → çok-claim'li grafa geçiş

**İlk durum:** Graf tek bir "anchor claim" düğümü kullanıyordu; ham claim'ler
gruplanmamıştı.

**Ne yaptık:** 546 segmentteki **359 ham claim** iki yöntemle kanonik claim'lere
gruplandı ve karşılaştırıldı:
- `src/cluster_claims.py` — **TF-IDF + KMeans** (topic-tabanlı, yüzeysel kelime örtüşmesi).
- `src/cluster_claims_embed.py` — **sentence-transformer embedding** (`all-MiniLM-L6-v2`)
  + KMeans (anlam-tabanlı).
- Her iki çıktı `data/graph/canonical_claims.json` ve `canonical_claims_embed.json`
  (20'şer küme, küme başına kanonik claim + üyeler).

**Ne oldu:** İki yöntem de 359 claim'i 20 kanonik başlığa indirdi. Embedding
yöntemi anlamca yakın ama farklı kelimelerle ifade edilmiş claim'leri daha iyi
topluyor; TF-IDF ise yüzeysel kelime örtüşmesine takılıyor. Bu, tek-anchor
grafından gerçek çok-claim'li person–claim–stance grafına geçişin altyapısı.

---

## 7. BERT student model — silver etiketlerden distillation

**İlk durum:** Sadece TF-IDF + LogReg stance baseline'ı vardı (Macro-F1 0.562).

**Ne yaptık:** `src/train_bert_stance.py` yazıldı. LLM silver stance etiketlerinden
**DistilBERT** (`distilbert-base-uncased`) fine-tune ediliyor — "teacher LLM →
küçük BERT student" distillation. Eğitim RTX 3050 GPU'da. Baseline ile **aynı
5-fold CV** metodolojisi kullanıldı.

- **Dengesizlik tuzağı:** düz cross-entropy ile model çoğunluk sınıfına (Neutral)
  çöktü → Macro-F1 **0.233** (hepsini Neutral tahmin etti). Baseline'daki gibi
  **balanced class-weighted cross-entropy** eklenince model üç sınıfı da öğrendi.

**Ne oldu:**

| Model | Metod | Stance Macro-F1 |
|---|---|---|
| Majority-class | — | ~0.23 |
| DistilBERT (düz CE) | 5-fold CV | 0.233 (çöktü) |
| DistilBERT (weighted CE, 6 epoch) | 5-fold CV | **0.518** |
| **TF-IDF + LogReg (baseline)** | 5-fold CV | **0.562** |

**Bulgu:** DistilBERT (0.518) bu veri setinde tuned TF-IDF baseline'ı (0.562)
**geçemedi**. Neden: sadece **546 örnek** (transformer fine-tune için çok küçük),
**silver** (gürültülü) etiketler ve sınıf dengesizliği. "Transformer küçük +
silver veride otomatik kazanmaz" — rapora değer dürüst bir negatif sonuç.
Sonraki adım: daha fazla veri + gold set ile yeniden ölçüm, veya soft-label
(teacher logit) distillation.

DistilBERT sınıf-bazlı (5-fold CV): Support F1 0.518, Refute 0.437, Neutral 0.597.

---

## 8. Gold etiketleme altyapısı — insan doğrulaması (human-in-the-loop)

**İlk durum:** Tüm etiketler silver'dı (LLM); gerçek (gold) test seti ve
LLM–insan uyum ölçümü yoktu.

**Ne yaptık (altyapı hazır, veri toplanacak):** Sıfırdan elle etiketlemek yerine
**LLM ön-etiketi + insan düzeltmesi** akışı kuruldu:
- `src/label_manual.py` — Streamlit aracı: her segmenti silver etiket **önceden
  dolu** gösterir; insan speaker/is_claim/claim/stance/topic'i onaylar veya
  düzeltir. Düzeltilen segment `reviewed: true` ile `data/gold/`'a yazılır,
  orijinal silver değer `*_silver` altında saklanır.
- `src/compute_agreement.py` — silver vs gold **Cohen's κ** + doğruluk +
  stance confusion matrix.

**Plan:** 20 yeni video eklenecek. ~2200 segmentin hepsi elle yapılmayacak;
**stratified ~400-500 segment** elle doğrulanıp **gold test seti** olacak, kalanı
BERT eğitimi için silver kalacak. κ değeri silver etiketlerin güvenilirliğini
raporlayacak.

---

## 9. Şartname hizalaması — 3 çekirdek görev + graf + retrieval

**İlk durum:** Resmi proje tanımına (Project ID 1) göre eksikler vardı: claim
görevi binary'di (multi-label değil), speaker/graf/retrieval değerlendirmesi
yoktu. Her yeni özellik **ayrı feature branch**'te geliştirilip main'e merge edildi.

**Ne yaptık:**
- **Görev 1 — Speaker identification** (`train_speaker.py`): TF-IDF baseline +
  class-weighted DistilBERT. 5 videoda **TF-IDF 5-fold Macro-F1 = 0.693**
  (accuracy 0.847; Unknown ve seyrek konuşmacılar hariç).
- **Görev 2 — Multi-label canonical claim classification**: 20 embedding
  kümesinden **14 kategorilik kanonik taksonomi** (`claim_taxonomy.py`) türetildi;
  `label_segments.py` artık her segmente multi-label `claim_labels` üretiyor;
  `label_manual.py`'ye çoklu-seçim eklendi; `assign_claim_labels.py` embedding ile
  silver multi-label bootstrap yapıyor; model `train_claim_multilabel.py`
  (DistilBERT sigmoid+BCE ve TF-IDF OvR baseline).
- **Görev 5 — Person–claim–stance grafı** (`build_graph_full.py`): tek anchor
  yerine kanonik claim kategorileri düğüm; person→claim kenarları stance dağılımı
  taşıyor.
- **Görev 6-8 — Stance-aware retrieval** (`eval_retrieval.py`): labels'tan
  (claim × stance) sorgu seti + qrels; **BM25 / dense / hybrid / stance-aware**
  karşılaştırması, **MRR ve nDCG@10**.
- **Görev 9 — Graf doğrulama** (`graph_judge.py`): LLM-as-a-judge kenar denetimi
  (yapısal bütünlük oranı).

**Not:** Multi-label claim, retrieval ve full-graf, `claim_labels` alanının dolu
olmasını ister — `assign_claim_labels.py` (bootstrap) veya güncellenmiş
`label_segments.py` çalıştırıldıktan sonra sonuç üretirler.

---

## 10. Veri genişletme — "AGI DISCOURSE" playlist'i (5 → 23 video)

**İlk durum:** 5 video / 546 segment. Bölüm 8-9'da planlanan "20 video ekle"
adımı bekliyordu. En zayıf sınıf **Refute** (126 örnek, F1 0.496).

**Ne yaptık:** 18 yeni video eklendi. Playlist'i toplu çekmek YouTube'un IP
rate-limit'ine takıldığı için **video-başına** bir akış kuruldu (`add_video.sh`):
transcript çek → segmentle → tek video etiketle, her adım idempotent. Böylece
yarıda kalan iş baştan başlamıyor ve her videonun sonucu tek tek denetlenebiliyor.

**Ne oldu:** **23 video / 2782 segment** (5.1 katı). Kaynak dağılımı bilinçli
olarak iki kutuplu seçildi: AGI-skeptik / x-risk tarafı (Yudkowsky, Yampolskiy ×3,
Tegmark, Leahy, Bengio, MacAskill, Rohin Shah) ve iyimser / geliştirici tarafı
(Amodei ×3, Hassabis, Schulman, Karpathy).

### Skorlar (5-fold CV, aynı metodoloji)

| Görev | 546 segment | 2782 segment |
|---|---|---|
| **Stance Macro-F1** | 0.562 | **0.566** |
| — Refute F1 | 0.496 | **0.594** |
| — Neutral F1 | ~0.75 | 0.678 |
| — Support F1 | — | 0.425 |
| **Claim Macro-F1** | 0.782 | 0.771 |

**Asıl kazanım toplam skorda değil, Refute'ta:** 0.496 → **0.594**. Refute artık
en zayıf değil, en güçlü ikinci sınıf. Sınıf dağılımı da dengelendi:
Neutral %53 → **%45**, Refute %23 → **%33**.

**Ara ölçüm yanıltıcı çıktı (kaydetmeye değer):** 1328 segmentte Refute F1
**0.442'ye düştü** ve "TF-IDF kapasite sınırına vurdu" diye yorumlandı. Veri iki
katına daha çıkınca 0.605'e yükseldi — yani bu bir plato değil, geçici bir çukurdu.
Ders: veri genişletmede tek ara ölçümden model kapasitesi hakkında sonuç çıkarmamak.

**Yeni zayıf halka Support (F1 0.437, 602 örnek)** — en az temsil edilen sınıf.
Sonraki hedefli genişletme buraya yönelmeli. Claim skoru pratikte sabit (0.774);
claim oranı %68 → %76 çıktığı için `no-claim` sınıfı zorlaştı (F1 0.649).

### Konuşmacı tabanı: 9 → 23 kişi (10+ segment)

En zengin düğümler: **Dario Amodei 415** (3 video), **Roman Yampolskiy 268**
(3 video), Will MacAskill 179, Ege Erdil 130, Max Tegmark 123.

**Aynı kişi, farklı bağlamda farklı duruş** — Amodei üç videoda:

| Video | Segment | Support | Refute | Neutral |
|---|---|---|---|---|
| Lex #452 | 249 | %25 | %22 | %53 |
| Dwarkesh (hidden pattern) | 76 | %9 | **%53** | %38 |
| Dwarkesh (exponential) | 90 | %32 | %37 | %31 |

Tek "baskın stance" etiketi bu kişiyi temsil etmiyor. Bu, tek-anchor ölçümünün
sınırını gösteren somut kanıt ve bölüm 6'daki kanonik claim'lere geçişi destekliyor.

---

## 11. Etiketleme kalitesi — süreçte çıkan 4 teknik bulgu

Videolar tek tek eklendiği için her birinin Unknown oranı ayrıca görüldü; bu da
konuşmacı atamasının nelere duyarlı olduğunu ortaya çıkardı.

**1. `VIDEO_META` roster'ı zorunlu.** Roster tanımsız videoda model isim
üretemiyor: ilk denemede 152 segmentin **%90'ı Unknown**. Videonun host/guest'i
`VIDEO_META`'ya eklenince aynı video **%17**'ye indi. Roster'lı videolarda
tipik oran %7-17.

**2. Uydurma placeholder isim zarar veriyor.** Sunucusu bilinmeyen videolarda
host olarak `"Interviewer"` yazılınca Unknown **%78** çıktı — model listedeki
sahte ismi kullanmak yerine Unknown'a düşüyor. Roster'ı **konuk-tek** bırakmak
(host = gerçek konuşmacı, guests = []) aynı videoyu **%10**'a indirdi.

**3. Otomatik (ASR) altyazı segmentasyonu bozuyor.** `segment_semantic()` iki
kesim sinyali kullanıyor: cümle sonu `.?!` veya ≥1.5sn duraklama. ASR
altyazılarda ikisi de yok — Yudkowsky videosunda 5195 satırın **1'i** noktalama
içeriyordu, ≥1.5sn duraklama sadece 16 taneydi. Sonuç: her segment `max_sec`'e
dayanıyor ve 198 dakikalık video **67 segmente** (ort **177sn**) bölünüyor,
diğer videolarda ort 67sn. Parametre taramasıyla `max_sec=90, pause_sec=0.4`
seçildi → **130 segment (ort 93sn)**. Aynı sorun 3 videoda çıktı (Yudkowsky,
Bengio, METR); `add_video.sh` artık noktalama oranını ölçüp %10'un altındaysa
sıkı eşiklere kendisi geçiyor.

**4. Çok konuşmacılı format yapısal sınır.** METR videosunda 3 konuşmacı var ve
altyazıda geçişler yalnızca `>>` ile işaretli, isim geçmiyor (transcript'te
"Beth" 4, "David" 1 kez). Unknown **%80**'de kaldı, roster'la düzelmedi.
3 kişilik Lex #490 panelinde de %38. Karşılaştırma: ikili sohbetlerde %2-15.
Segmentler stance/claim modelleri için tutuldu; Unknown oldukları için grafa
katkı vermiyorlar.

**Genel Unknown oranı %17** (546 segmentte %16'ydı) — veri 5 katına çıkarken
konuşmacı ataması bozulmadı.

---

## 12. Sonraki adımlar (planlanan)

- **DistilBERT'i yeniden ölç:** bölüm 7'deki negatif sonucun gerekçesi "sadece
  546 örnek"ti; artık 2782 var. `train_bert_stance.py --cv` ile TF-IDF'i (0.566)
  geçip geçmediği artık gerçek bir deney.
- **Support sınıfını güçlendir:** yeni en zayıf sınıf (F1 0.425). Refute'ta işe
  yarayan hedefli genişletmenin aynısı, bu kez iyimser konuşmacılarla.
- **Graf + retrieval'ı güncel veriyle çalıştır:** `build_graph_full.py`,
  `eval_retrieval.py`, `graph_judge.py` — 9 değil 23 kişilik graf.
- **Gold set:** ~400-500 segmenti `label_manual.py` ile doğrula, κ'yı raporla.
- **Soft-label distillation:** teacher LLM logit/olasılıklarıyla.
- **Konuşmacı belirsizliği:** bulgu 4'teki çok-konuşmacılı durum için diyalog
  yapısı / ses-tabanlı diarization.

---

## Dosyalar

| Dosya | İş |
|---|---|
| `src/fetch_transcripts.py` | Transcript çekme + segmentasyon (baseline + semantic) |
| `src/label_segments.py` | Zero-shot LLM etiketleme (speaker/claim/stance) |
| `src/train_baseline.py` | Stance baseline (TF-IDF + LogReg) |
| `src/train_claim_baseline.py` | Claim classification baseline |
| `src/build_graph.py` | Person–stance knowledge graph + görsel |
| `src/cluster_claims.py` | Kanonik claim clustering (TF-IDF + KMeans) |
| `src/cluster_claims_embed.py` | Kanonik claim clustering (sentence-embedding + KMeans) |
| `src/train_bert_stance.py` | BERT stance student (DistilBERT, weighted CE, 5-fold CV) |
| `src/label_manual.py` | Elle gold etiketleme aracı (Streamlit; silver'ı düzeltip data/gold/) |
| `src/compute_agreement.py` | LLM silver vs insan gold uyumu (Cohen's κ) |
| `src/claim_taxonomy.py` | 14 kategorilik kanonik claim taksonomisi |
| `src/assign_claim_labels.py` | Embedding ile silver multi-label claim bootstrap |
| `src/train_claim_multilabel.py` | Multi-label claim modeli (DistilBERT BCE + TF-IDF OvR) |
| `src/train_speaker.py` | Speaker identification (TF-IDF + DistilBERT) |
| `src/build_graph_full.py` | Person–claim–stance grafı (kanonik claim düğümleri) |
| `src/eval_retrieval.py` | Stance-aware retrieval eval (BM25/dense/hybrid, MRR/nDCG@10) |
| `src/graph_judge.py` | LLM-as-a-judge graf doğrulama |
| `run_labeling.sh` | Etiketleme + graf yeniden kurma yardımcı script'i |
| `add_video.sh` | Tek video ekleme (çek → segmentle → etiketle; ASR tespiti, idempotent) |
