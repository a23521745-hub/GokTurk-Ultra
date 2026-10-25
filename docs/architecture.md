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
  teachers.py          OpenAI-uyumlu öğretmen adaptörü (retry/timeout/limit, rol yönlendirme)
  task_specs.py        10 görev ailesi: istemler, konu bankaları, katı doğrulayıcılar
  curriculum.py        Deterministik plan (N örnek → aile/konu/seviye/rol) + maliyet
  data_generation.py   Basic (taslak+yanıt) ve PRO (N aday + öz-eleştiri/düzeltme) üretim
  data_quality.py      Şema, kalite filtreleri, tekrar temizliği, gruplu bölme
  env_check.py         Donanım/paket/mimari denetimi + bellek tahmini
  unsloth_backend.py   Unsloth 4-bit QLoRA yolu (≈14 GB VRAM) + API sürüm uyumu
  train.py             Eğitim giriş noktası (transformers | unsloth) + duman testi
  evaluate.py          Biçim kontrolü, kural tabanlı skorlama, opsiyonel yargıç
configs/
  teachers.yaml        Sağlayıcı + öğretmen modeller (env referanslı, rol yönlendirmeli)
  training.yaml        Eğitim ayarları — generic transformers yolu
  training_unsloth_t4.yaml  Unsloth 4-bit QLoRA (Kaggle 2×T4) ayarları
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
- `base_url` ve model kimlikleri `env:VAR` biçiminde de referanslanabilir.
  Değişken tanımlı değilse üretim **ağa çıkmadan** durur (`--dry-run` yalnızca
  plan yazdırır). Hiçbir endpoint veya model kimliği uydurulmamıştır.
- Repoda gelen değerler **doğrulanmıştır** (2026-10-25):
  `https://evren-llmapi.ssyz.org.tr/v1` (kimliksiz GET, sağlayıcının kendi
  `API anahtarı eksik` yanıtını döndürdü), `mimo-v2.6-pro` ve `glm-5.3`
  (EVREN'in model filosu sayfası + bağımsız istemcilerin model listeleri).
- Kimlikler koda gömülü bir varsayım değil, **çalışma anında doğrulanabilir**
  bir iddiadır: `python -m gokturk.data_generation --list-models` sağlayıcının
  `GET /models` yanıtını alır ve `configs/teachers.yaml` içindeki kimliklerle
  karşılaştırır; eksik varsa çıkış kodu 2 ile durur.
- `.env` `.gitignore` ile hariç tutulur; `.env.example` yalnızca boş anahtarlar
  içerir.
- Kayıt şeması `source` içinde kimlik bilgisi kabul etmez (`data_quality`
  `api_key/authorization/token/password` alanlarını reddeder).
- Hata sınıflandırması: 401/403 → kalıcı hata (tekrar denenmez), 429/5xx/
  timeout/boş yanıt → geri çekilmeli tekrar, art arda N hata → koşu güvenli
  biçimde durur. Boş veya bozuk yanıt **asla** veri olarak yazılmaz.
- Öğretmen *model kimlikleri* üretim planında yer almaz; plan yalnızca anlamsal
  rol (`general` / `reasoning` / `code`) konuşur. Kimlik eşlemesi çalışma anında
  doğrulanmış ortam değişkenlerinden yapılır, yani plan dosyası sızsa bile
  sağlayıcı bilgisi taşımaz.

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

## 5. Profesyonel sentetik veri hattı

Hedef: yüksek kalite + yüksek çeşitlilik. Üç katman:

**1) Plan (ağa çıkmaz).** `gokturk.curriculum.build_plan(N, seed=…)` N örneklik
işi ailelere böler (en büyük kalan yöntemi), seviyeleri ve rolleri atar, aynı
seed için birebir aynı planı üretir. 10.000 örneklik varsayılan dağılım:

| Aile | Örnek | Rol | Doğrulayıcılar |
| --- | --- | --- | --- |
| coding | 1348 | code | kod bloğu + dil, sızıntı, Türkçe, uzunluk |
| reasoning | 1348 | reasoning | en az 2 bölüm, sızıntı, Türkçe |
| system_architecture | 1348 | reasoning/general | ≥3 bölüm, uzunluk |
| turkish_advanced | 1124 | general/reasoning | Türkçe, uzunluk |
| instruction_following | 1011 | general | Türkçe, uzunluk |
| coding_debug | 899 | code | kod bloğu, dil |
| tool_calling | 899 | code/general | tek JSON nesnesi + anahtarlar |
| summarization | 787 | general | Türkçe, uzunluk |
| rewrite | 674 | general | Türkçe, uzunluk |
| translation | 562 | general | Türkçe, uzunluk |

Konu bankaları 112 elle yazılmış Türkçe konu içerir; aile başına en az 6.
Plan `--plan-only` ile maliyet tahmini verir (kaba, ≈4 karakter/token).

**2) Üretim (PRO modu).** Her slot için:

1. *Taslak*: öğretmen, aile+konu+seviye için gerçekçi bir Türkçe kullanıcı
   isteği üretir (JSON). Bozuk taslak onarılmaz, atılır.
2. *N aday*: aynı istem için `candidates` (varsayılan 2) yanıt üretilir;
   sıcaklık aday başına hafifçe artırılır (tool_calling/instruction_following
   hariç, determinizm için). En az doğrulayıcı hatası olan, eşitlikte en uzun
   aday seçilir.
3. *Öz-eleştiri + onarım*: farklı roldeki bir öğretmen adayı ölçüt listesine
   göre denetler ve gerekirse **tam** düzeltilmiş yanıt döndürür. Onarım yalnızca
   aynı katı doğrulayıcılardan ve şema/kalite kapısından geçerse kabul edilir;
   aksi halde özgün yanıt korunur ve reddedilen onarım denetim için kaydedilir.
4. *Kapı*: `assess_record` (uzunluk, reddetme, kontrol karakteri, tekrar) + aile
   doğrulayıcıları. Geçmeyen örnek `*.rejects.jsonl`'e nedenleriyle yazılır.
5. *Provenance*: `meta` içinde aile/konu/seviye, aday indeksi, eleştiri
   bulguları, onarım bayrağı, yumuşak notlar, usage; `source` içinde öğretmen
   adı, modeli, rolleri, taslak öğretmeni. `meta.review.status` başlangıçta
   `unreviewed`.

Güvenlik davranışı basic modla aynıdır: anahtar yoksa hiç istek atılmaz; 401/403
ve 400 sınıfı hatalar ölümcül; art arda taşıma hatası eşiği aşılırsa koşu durur
ve o ana kadar yazılan veri korunur (`--resume` ile devam).

**3) İnceleme.** `data_quality --review-sample 30` deterministik örnek çıkarır.
Kabul eşiği aşılırsa ilgili `template_version` ile üretilen veri atılır (§4).

Sınır: doğrulayıcılar **biçimi** denetler (JSON şeması, kod bloğu, dil, uzunluk,
sızıntı). Anlamsal doğruluk otomatik ölçülmez; bu yüzden insan incelemesi
zorunludur. Öğretmen kendi yanıtını denetlediği için öz-eleştiri, kör noktaları
paylaşan bir öğretmende sistematik hatayı yakalamayabilir; farklı roldeki
öğretmenle çapraz denetim bu riski azaltır ama ortadan kaldırmaz.

## 6. Eğitim: ne destekleniyor, ne desteklenmiyor

Uygulanan yol: `transformers` + `peft` ile

| Yöntem | Koşul | Durum |
| --- | --- | --- |
| `full` | Ağırlıklar + gradyan + Adam durumları VRAM'e sığmalı | Kod yolu var; 20B'de engellenir |
| `lora` (bf16/fp16 taban) | Taban ağırlıklar tam hassasiyette bellekte | Uygulanmış |
| `qlora` (NF4) | `bitsandbytes` + CUDA | Uygulanmış |
| Unsloth 4-bit QLoRA | Unsloth + CUDA | **Uygulandı**: `--backend unsloth` (isteğe bağlı bağımlılık) |
| MXFP4 ile yerel eğitim | — | **Desteklenmiyor** (bkz. §6) |

Unsloth yolu (`gokturk/unsloth_backend.py`) yalnızca model yükleme ve adaptör
oluşturmayı devreder; veri tokenizasyonu, prompt maskeleme, doğrulama ve
manifest üretimi bu repoda kalır. Parametreler kurulu Unsloth sürümünün
imzasına göre filtrelenir (API kayması koruması) ve Unsloth yoksa çalışma
**generic yola sessizce düşmez**: `--backend unsloth` açıkça hata verir.
`4-bit` için yayımlanmış referans ≈14 GB, 16-bit LoRA için ≈44 GB'dır; bunlar
tahmin tablosuyla karıştırılmaz, rapor ayrı satırda gösterir.

`train.py` ayrıca: prompt token'larını `-100` ile maskeler (kayıp yalnızca
asistan yanıtına aittir), kesme sonrası yanıtı tamamen kaybolan örnekleri
atar, `transformers` 4.x/5.x farklarını (ör. `warmup_ratio` kaldırılması)
uyarlar, `optim=paged_adamw_8bit` CUDA yoksa `adamw_torch`'a düşer ve bunu
loglar.

## 7. Donanım gerçeği: 2×T4 ve gpt-oss-20b

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

**Sonuç:**

* *Generic yol* (transformers + bitsandbytes) ile gpt-oss-20b 2×T4 üzerinde
  eğitilemez; `env_check` bunu engelleyici bulgu olarak raporlar ve `train.py`
  eğitimi başlatmaz.
* *Unsloth 4-bit yolu* (`--backend unsloth`) bu repoda uygulanmıştır ve
  Unsloth'un yayımlanmış ≈14 GB rakamına dayanarak 16 GB'lık T4'te **prensip
  olarak** çalışabilir. Ancak:
  - bu repoda gerçek bir Kaggle koşusu doğrulanmamıştır (GPU yok);
  - T4 bf16 desteklemediği için Unsloth MoE katmanlarını float32'ye taşır ve
    fp16 kayıp kararlılığı geçmişte sorun olmuştur → 60 adımlık ölçüm koşusu
    (`configs/training_unsloth_t4.yaml`, `max_steps: 60`) zorunlu ilk adımdır;
  - 1024 token bağlam güvenli başlangıçtır; 2048 VRAM ve kararlılık ölçülerek
    denenmelidir;
  - ikinci T4 kullanılmaz (model paralelliği yok).

Alternatifler: ≥48 GB VRAM'li tek GPU ile bf16 LoRA, veya 7–8B sınıfı bir taban
modelle boru hattını doğrulayıp ölçek büyütmek.

Lisans: `openai/gpt-oss-20b` Apache-2.0 (+ kullanım politikası). Bu repo kendi
Apache-2.0 lisansını korur; model ağırlıklarının ve kullanılacak 10.000
örneklik veri setinin lisansı ayrıdır ve **doğrulanmadan devralınmaz**.

## 8. Değerlendirme

Üç mod ayrıdır: `format` (veri biçimi), `score` (sabit test setinde kural
tabanlı geçme oranı), `judge` (öğretmen model yargıç; doğrulanmamış, kapalı
varsayılan). Ölçülmemiş kategoriler `null`/`ölçülmedi` olarak raporlanır;
eksik veri sıfır sayılmaz. Rapordaki hiçbir sayı kamuya açık benchmark yerine
geçmez; kategori başına 4 öğe istatistiksel olarak anlamlı değildir, yalnızca
regresyon yakalamak için bir duman göstergesidir.

## 9. Bilinen sınırlamalar (özet)

1. Uç nokta ve model kimlikleri **herkese açık kaynaklardan** doğrulandı
   (2026-10-25); ancak bu repoda anahtarla **kimlik doğrulanmış bir çağrı**
   yapılmadı — yani `/models` listesi ve kotanın gerçekten açık olduğu
   operatör tarafından bir kez `--list-models` / `--health-check` ile
   teyit edilmelidir. (Öğretmen 2 Kimi-K3'ten GLM-5.3'e taşındı: EVREN'in
   filosunda Kimi yok.)
2. Üretim tek iş parçacıklı ve sıralıdır; PRO modunda örnek başına 3–4 çağrı
   yapıldığından 10.000 örnek ≈30–40 bin çağrı demektir ve oran sınırına bağlı
   olarak saatler sürer. Paralelleştirme eklenmedi (bilinçli: sıralı akış
   ölçülebilir ve tekrar üretilebilir).
3. Yakın-tekrar tespiti kelime 3-gram Jaccard ile sınırlıdır; anlamsal tekrarları
   yakalamaz. Büyük veri setlerinde MinHash/LSH gerekir.
4. Token sayımı yoktur; uzunluk bütçesi karakter/kelime cinsindendir.
5. `evaluate.py` anlam doğruluğunu ölçmez; yargıç modu kalibre edilmemiştir.
6. Eğitim betiği yalnızca tek GPU'yu hedefler; FSDP/DeepSpeed yapılandırması yok.
7. Duman testi rastgele bir model kullanır: kayıp değerleri anlamsızdır,
   yalnızca boru hattının çalıştığını gösterir.
8. 10.000 örneklik veri seti bu repoda yoktur; kullanıcı tarafından sağlanır ve
   lisansı/ kalitesi ayrıca doğrulanmalıdır.
9. PRO modunda öz-eleştiri aynı sağlayıcıdaki bir öğretmenle yapılır; bağımsız
   bir insan/yargıç denetimi yerine geçmez. Ölçüt listeleri elle yazılmıştır ve
   aile başına 2–4 maddeden oluşur.
10. Plan ve maliyet tahmini kabadır (≈4 karakter/token); gerçek token kullanımı
    sağlayıcı sayaçlarından okunmalıdır.
11. Unsloth yolu bu depoda birim testlerle (sahte modül) doğrulanmıştır, gerçek
    GPU üzerinde değil. API sürüm kaymasına karşı imza filtresi vardır ama
    Unsloth'un gelecekteki sürümleri ek doğrulama gerektirebilir.

## 10. Kaynaklar

- OpenAI gpt-oss model kartı ve README (mimari, MXFP4, Apache-2.0).
- Hugging Face transformers v4.55 sürüm notları (gpt-oss desteği).
- Unsloth gpt-oss eğitim belgeleri (≈44 GB bf16 LoRA, ≈14 GB 4-bit, fp16
  ıraksama notları, "MXFP4 eğitimi desteklenmiyor").
- NVIDIA Model Optimizer gpt-oss örneği (MXFP4 → bf16 yükseltme gerekçesi).
- Kaggle notebook dokümantasyonu (T4×2, 12 saat oturum, ~30 saat/hafta GPU,
  20 GB `/kaggle/working`).
- Unsloth gpt-oss eğitim belgeleri: `FastLanguageModel`/`FastModel`
  `from_pretrained` + `get_peft_model`, `use_gradient_checkpointing="unsloth"`,
  ≈14 GB (4-bit) / ≈44 GB (bf16) ve fp16 MoE taşma notları. API ayrıntıları
  sürüme göre değiştiği için `unsloth_backend.py` parametreleri kurulu sürümün
  imzasına göre filtreler.
