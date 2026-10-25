# Mimari ve bilinen sınırlamalar

> Bu belge, projenin **ne olduğunu ve ne olmadığını** açıkça yazar.
> 2026-10-25 itibarıyla bu repoda **eğitilmiş model ağırlığı yoktur** ve
> hiçbir doğruluk/benchmark sonucu üretilmemiştir.

## 1. Amaç ve kapsam

GökTürk-Ultra, Türkçe yetenekleri güçlü, talimat izleyen ve gerektiğinde araç
çağıran bir model/ajan prototipi için **altyapı** sağlar:

- öğretmen modellerle sentetik veri üretimi,
- veri doğrulama, tekrar temizliği ve sızıntısız bölme,
- LoRA/QLoRA eğitim giriş noktası ve dürüst donanım ön denetimi,
- sabit bir test seti üzerinde kural tabanlı değerlendirme.

Kapsam dışı: dağıtım/servis, tensor-parallel eğitim, öğretmen model barındırma,
kamuya açık benchmark koşumu.

## 2. Bileşenler

```
src/gokturk/
  schema.py            JSONL kayıt şeması (sürümlü)
  config.py            Katı YAML yükleme + doğrulama; sırlar yalnızca env'den
  teachers.py          OpenAI-uyumlu öğretmen adaptörü (retry/timeout/limit)
  data_generation.py   Taslak (drafter) + yanıt (solver) iki adımlı üretim
  data_quality.py      Şema, kalite filtreleri, tekrar temizliği, gruplu bölme
  env_check.py         Donanım/paket/mimari denetimi + bellek tahmini
  train.py             Eğitim giriş noktası (LoRA/QLoRA/full) + duman testi
  evaluate.py          Biçim kontrolü, kural tabanlı skorlama, opsiyonel yargıç
configs/
  teachers.yaml        Sağlayıcı + öğretmen modeller (env referanslı)
  training.yaml        Eğitim ayarları (açık yorumlarla)
data/
  seeds/               Türkçe tohum görevler
  eval/                Sabit değerlendirme seti (16 öğe, 4 kategori)
  samples/             6 elle yazılmış örnek (eğitim seti DEĞİL)
notebooks/
  kaggle_training.ipynb  Kaggle deneyi (ön denetim → duman testi → dürüst duruş)
tests/                 125 otomatik test
```

Veri akışı:

```
seeds ──> drafter (öğretmen) ──> prompt ──> solver (öğretmen) ──> response
                                                                    │
                                          schema + kalite filtreleri ─┤
                                                                    ▼
                       dedup (exact + near) ──> gruplu bölme ──> train/valid/test
```

## 3. Güvenlik ve sır yönetimi

- API anahtarı hiçbir dosyada tutulmaz. `configs/teachers.yaml` yalnızca
  **değişken adını** (`api_key_env`) içerir; değer `os.environ`'dan okunur.
- `base_url` ve model kimlikleri `env:VAR` biçiminde referanslanabilir.
  Değişken tanımlı değilse üretim **ağa çıkmadan** durur (`--dry-run` yalnızca
  plan yazdırır). Hiçbir endpoint veya model kimliği uydurulmamıştır.
- `.env` `.gitignore` ile hariç tutulur; `.env.example` yalnızca boş anahtarlar
  içerir.
- Kayıt şeması `source` içinde kimlik bilgisi kabul etmez (`data_quality`
  `api_key/authorization/token/password` alanlarını reddeder).
- Hata sınıflandırması: 401/403 → kalıcı hata (tekrar denenmez), 429/5xx/
  timeout/boş yanıt → geri çekilmeli tekrar, art arda N hata → koşu güvenli
  biçimde durur. Boş veya bozuk yanıt **asla** veri olarak yazılmaz.

## 4. Veri kalitesi ve insan incelemesi

Otomatik kontroller (`assess_record`): boş alan, uzunluk bütçesi, reddetme
kalıpları ("as an AI language model" vb.), satır tekrarı, dengesiz kod bloğu,
kontrol karakterleri, `truncated` işareti. Tekrar temizliği: aynı `id`, aynı
normalize edilmiş (prompt+response), ve kelime 3-gram Jaccard ≥ 0.9 yakın
tekrarlar.

Sızıntı azaltma: bölme **grup bazlıdır**. Aynı `source.seed_id` (yoksa prompt'un
ilk 6 kelimesi) her zaman aynı bölünmeye gider, böylece bir istemin
varyasyonları train ve test'e dağılmaz.

**Öğretmen yanıtı doğru kabul edilmez.** Önerilen inceleme süreci:

1. Her üretim koşusundan sonra `data/processed/data_report.json` ve
   `rejects.jsonl` okunur (ret nedenleri dağılımı gözden geçirilir).
   `python -m gokturk.data_quality --data <dosya> --review-sample 30
   --review-out reports/review.md` deterministik bir inceleme örneği üretir.
2. Rastgele 30–50 örnek insan tarafından okunur; hata oranı not edilir.
3. Hata oranı kabul eşiğinin üstündeyse istem şablonu düzeltilir ve o
   `source.template_version` ile üretilmiş veri **atılır**.
4. Yalnızca insan onayından geçen alt küme eğitimde kullanılır; onaylanan
   kayıtların `meta.review` alanına inceleme bilgisi eklenir (şema bunu
   engellemez, `meta` serbesttir).
5. Değerlendirme seti (`data/eval`) öğretmenlerle hiç üretilmez: elle yazılır,
   böylece öğretmenin kendi üslubunu "doğru" sayma riski azalır.

## 5. Eğitim: ne destekleniyor, ne desteklenmiyor

Uygulanan yol: `transformers` + `peft` ile

| Yöntem | Koşul | Durum |
| --- | --- | --- |
| `full` | Ağırlıklar + gradyan + Adam durumları VRAM'e sığmalı | Kod yolu var; 20B'de engellenir |
| `lora` (bf16/fp16 taban) | Taban ağırlıklar tam hassasiyette bellekte | Uygulanmış |
| `qlora` (NF4) | `bitsandbytes` + CUDA | Uygulanmış |
| Unsloth 4-bit | Unsloth çekirdekleri | **Uygulanmadı** (`unsloth_4bit: true` yapılandırma hatası verir) |
| MXFP4 ile yerel eğitim | — | **Desteklenmiyor** (bkz. §6) |

`train.py` ayrıca: prompt token'larını `-100` ile maskeler (kayıp yalnızca
asistan yanıtına aittir), kesme sonrası yanıtı tamamen kaybolan örnekleri
atar, `transformers` 4.x/5.x farklarını (ör. `warmup_ratio` kaldırılması)
uyarlar, `optim=paged_adamw_8bit` CUDA yoksa `adamw_torch`'a düşer ve bunu
loglar.

## 6. Donanım gerçeği: 2×T4 ve gpt-oss-20b

Kaynaklara dayalı özet:

- **gpt-oss-20b**: 21B toplam / 3.6B aktif parametre, 24 katman, 32 uzman,
  top-4 yönlendirme; MoE uzman ağırlıkları **MXFP4** olarak yayınlanmıştır
  (model kartı, OpenAI gpt-oss README).
- Yerel MXFP4 ile **gradyan eğitimi ana akım kütüphanelerde desteklenmez**;
  ağırlıkların bf16/fp16'ya yükseltilmesi gerekir (NVIDIA Model Optimizer
  gpt-oss notları; Unsloth SSS: "no framework supports fp4 or MXFP4 training").
- bf16'ya yükseltilmiş 21B ağırlık ≈42 GB tutar; **bf16 LoRA için yayımlanmış
  gereksinim ≈44 GB VRAM** (Unsloth belgeleri).
- Unsloth'un özel 4-bit QLoRA yolu 20B'yi ≈14 GB'da eğitiyor; bu, transformers
  +bitsandbytes yolunun sağladığı bir şey değildir ve bu repoda uygulanmamıştır.
- **Kaggle ücretsiz katmanı**: 2× Tesla T4 (her biri 16 GB), oturum başına
  12 saat, haftada ~30 GPU saati, `/kaggle/working` için 20 GB kalıcı disk.
- **T4 (compute 7.5) bf16 desteklemez** → fp16'ya düşülür. Unsloth'un hata
  notlarına göre gpt-oss'ta fp16 eğitim kayıp ıraksamasına yol açabiliyor.
  Ayrıca bu repodaki yol model paralelliği/FSDP uygulamaz; model **tek GPU'ya**
  sığmalıdır.

**Sonuç:** `openai/gpt-oss-20b` + 2×T4 üzerinde bu repodaki yöntemlerle eğitim
mümkün değildir. `gokturk.env_check` bunu engelleyici bulgu olarak raporlar ve
`train.py` eğitimi başlatmaz. Gerçekçi yollar:

1. Unsloth tabanlı ayrı bir eğitim yolu (ayrı bağımlılık, ayrı doğrulama;
   fp16 ıraksama riski ölçülmeli).
2. ≥48 GB VRAM'li tek GPU ile bf16 LoRA.
3. 7–8B sınıfı bir taban modelle boru hattını uçtan uca doğrulayıp ölçek büyütmek
   (en düşük riskli sıradaki adım).

Lisans: `openai/gpt-oss-20b` Apache-2.0 (+ kullanım politikası). Bu repo kendi
Apache-2.0 lisansını korur; model ağırlıklarının ve kullanılacak 10.000
örneklik veri setinin lisansı ayrıdır ve **doğrulanmadan devralınmaz**.

## 7. Değerlendirme

Üç mod ayrıdır: `format` (veri biçimi), `score` (sabit test setinde kural
tabanlı geçme oranı), `judge` (öğretmen model yargıç; doğrulanmamış, kapalı
varsayılan). Ölçülmemiş kategoriler `null`/`ölçülmedi` olarak raporlanır;
eksik veri sıfır sayılmaz. Rapordaki hiçbir sayı kamuya açık benchmark yerine
geçmez; kategori başına 4 öğe istatistiksel olarak anlamlı değildir, yalnızca
regresyon yakalamak için bir duman göstergesidir.

## 8. Bilinen sınırlamalar (özet)

1. Gerçek öğretmen uç noktası/model kimlikleri doğrulanmadı; yapılandırma
   `env:` referanslarıyla boş bırakıldı. EVREN'in OpenAI-uyumlu olduğu
   belgelenmiş olsa da, Kimi/MiMo kimlikleri sağlayıcının listesinden
   doğrulanmalı.
2. Üretim tek iş parçacıklı ve sıralıdır; 10.000 örnek için yavaş olabilir
   (oran sınırına uyum için bilinçli). Paralelleştirme eklenmedi.
3. Yakın-tekrar tespiti kelime 3-gram Jaccard ile sınırlıdır; anlamsal tekrarları
   yakalamaz. Büyük veri setlerinde MinHash/LSH gerekir.
4. Token sayımı yoktur; uzunluk bütçesi karakter/kelime cinsindendir.
5. `evaluate.py` anlam doğruluğunu ölçmez; yargıç modu kalibre edilmemiştir.
6. Eğitim betiği yalnızca tek GPU'yu hedefler; FSDP/DeepSpeed yapılandırması yok.
7. Duman testi rastgele bir model kullanır: kayıp değerleri anlamsızdır,
   yalnızca boru hattının çalıştığını gösterir.
8. 10.000 örneklik veri seti bu repoda yoktur; kullanıcı tarafından sağlanır ve
   lisansı/ kalitesi ayrıca doğrulanmalıdır.

## 9. Kaynaklar

- OpenAI gpt-oss model kartı ve README (mimari, MXFP4, Apache-2.0).
- Hugging Face transformers v4.55 sürüm notları (gpt-oss desteği).
- Unsloth gpt-oss eğitim belgeleri (≈44 GB bf16 LoRA, ≈14 GB 4-bit, fp16
  ıraksama notları, "MXFP4 eğitimi desteklenmiyor").
- NVIDIA Model Optimizer gpt-oss örneği (MXFP4 → bf16 yükseltme gerekçesi).
- Kaggle notebook dokümantasyonu (T4×2, 12 saat oturum, ~30 saat/hafta GPU,
  20 GB `/kaggle/working`).
