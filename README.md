# GökTürk-Ultra

Türkçe yetenekleri güçlü, talimat izleyen ve gerektiğinde araç çağıran bir
model/ajan prototipi için **veri, eğitim ve değerlendirme altyapısı**.

> **Durum: altyapı (v0.1.0).** Bu repoda eğitilmiş model ağırlığı **yoktur** ve
> hiçbir doğruluk/benchmark sonucu **ölçülmemiştir**. Amaç ölçülebilir,
> tekrar üretilebilir bir temel kurmaktır. Ayrıntılı ve dürüst sınırlama listesi:
> [`docs/architecture.md`](docs/architecture.md).

## Neler var?

| Bileşen | Dosya | Ne yapar |
| --- | --- | --- |
| Öğretmen adaptörü | `src/gokturk/teachers.py` | OpenAI-uyumlu sohbet API'si; timeout, 429/5xx tekrarı, 401'de anında durma, oran sınırı |
| Veri üretimi (basic) | `src/gokturk/data_generation.py` | Türkçe tohumlardan iki adımlı (taslak + yanıt) sentetik veri; yarıda kalırsa güvenli durur, devam ettirilebilir |
| Veri üretimi (PRO) | `src/gokturk/data_generation.py` + `task_specs.py` + `curriculum.py` | 10 görev ailesi, N aday örnekleme, öz-eleştiri/onarım, rol yönlendirme, plan + maliyet tahmini |
| Unsloth arka ucu | `src/gokturk/unsloth_backend.py` | gpt-oss-20b için 4-bit QLoRA (≈14 GB VRAM); API sürüm kaymasına dayanıklı |
| Veri kalitesi | `src/gokturk/data_quality.py` | Şema doğrulama, bozuk JSON/boş/uzun/reddetme/tekrar filtreleri, yakın-tekrar temizliği, sızıntısız gruplu bölme |
| Ortam denetimi | `src/gokturk/env_check.py` | GPU/CUDA/VRAM/paket/mimari denetimi + bellek tahmini; engelleyici bulguları raporlar |
| Eğitim | `src/gokturk/train.py` | `--backend transformers` (LoRA/QLoRA/full) veya `--backend unsloth` (4-bit QLoRA); prompt maskeleme; ön denetim geçmezse **başlamaz**; CPU duman testi |
| Değerlendirme | `src/gokturk/evaluate.py` | Biçim kontrolü, sabit Türkçe test setinde kural tabanlı skorlama, opsiyonel (doğrulanmamış) yargıç |
| Kaggle deneyi | `notebooks/kaggle_training.ipynb` | Ortam → paketler → veri doğrulama → duman testi → dürüst tanılama → çıktı kaydı |
| GGUF (adaptör-only) | `src/gokturk/lora_gguf.py` | `convert_lora_to_gguf.py` planı/çalıştırması; birleştirme yok, ≈2 GB RAM; llama.cpp mimari desteğini doğrular |
| GitHub Actions mini eğitim | `.github/workflows/mini-adapter.yml` + `configs/training_mini_cpu.yaml` | CPU runner'da 0.6B LoRA (30–100 adım) → manifest → log denetimi → adaptör-only GGUF; boru hattı kanıtı, kalite iddiası değil |

## Kurulum

```bash
git clone https://github.com/a23521745-hub/GokTurk-Ultra.git
cd GokTurk-Ultra

# Veri hattı + değerlendirme için yeterli (torch gerekmez)
pip install -e .

# Eğitim için ek bağımlılıklar (CUDA ortamı)
pip install -e ".[train]"
pip install -e ".[qlora]"      # generic QLoRA / 4-bit NF4 (bitsandbytes)
pip install -e ".[unsloth]"    # 4-bit QLoRA, gpt-oss'u ~14 GB VRAM'e sığdırır

# Geliştirme
pip install -e ".[dev]"
```

`requirements.txt` alternatif olarak kullanılabilir; ağır ML bağımlılıkları
orada yorum satırı olarak bırakılmıştır.

## Hızlı başlangıç (API anahtarı gerekmez)

```bash
# 1) Donanım ve yöntem uygunluğunu denetle
python -m gokturk.env_check --model openai/gpt-oss-20b --method qlora --load-in-4bit
python -m gokturk.train --plan-only          # denetim + çözümlenmiş plan

# 2) Boru hattını uçtan uca duman testiyle doğrula (model indirmez, CPU'da çalışır)
python -m gokturk.train --smoke --max-steps 2 --output-dir outputs/smoke \
  --train-file data/samples/sample_train.jsonl \
  --eval-file data/samples/sample_train.jsonl
# -> outputs/smoke/run_manifest.json  (kind="smoke", "kalite iddiası yok")

# 3) Örnek veriyi doğrula, temizle ve böl
python -c "from gokturk.data_quality import build_datasets; \
print(build_datasets('data/samples/sample_train.jsonl','data/processed').summary())"

# 4) Biçim kontrolü (model kalitesi ölçmez)
python -m gokturk.evaluate --mode format --data data/processed/train.jsonl

# 5) İnsan incelemesi için örnek çıkar (öğretmen yanıtı doğru kabul edilmez)
python -m gokturk.data_quality --data data/processed/train.jsonl \
  --review-sample 30 --review-out reports/review.md
```

## Hazır CoT verisiyle eğitim (`data/datasets/evren_cot_10k.jsonl`)

Depoda 10.000 satırlık CoT dosyası var. Eğitim şemasında **değildir**; önce çevrilir:

```bash
# 1) instruction/thinking_process/response -> prompt/response (+ tekilleştirme, kalite kapısı, bölme)
python -m gokturk.cot_dataset --input data/datasets/evren_cot_10k.jsonl --out-dir data/processed
# -> train 1271 / valid 28 / test 27 + data/processed/cot_report.json

# 2) Plan: 4B sığıyor mu, disk yetiyor mu, kaç adım
python -m gokturk.train --config configs/training_qwen3_5_4b_cot_t4.yaml --plan-only
```

**Ölçülmüş gerçek:** 10.000 satır yalnızca **920 tekil talimat** içerir (etkin
çeşitlilik %9,2). Hepsiyle eğitim ~10 epoch aynı örnekleri tekrarlamak olur; bu
yüzden araç varsayılan olarak tekilleştirir ve ~1.300 örnek bırakır. Bu dosya
tek başına genel Türkçe talimat takibi öğretmez — ayrıntı, ölçüm tablosu ve
dürüst beklenti: [`docs/veri-seti-hazirlama.md`](docs/veri-seti-hazirlama.md).

## Profesyonel sentetik veri seti (PRO modu)

On görev ailesi — kodlama, hata ayıklama, akıl yürütme, **ileri düzey Türkçe**,
**sistem mimarisi**, araç çağırma, talimat takibi, özetleme, yeniden yazma,
çeviri — üzerinde planlı üretim. Plan ve maliyet tahmini **ağa çıkmaz**:

```bash
python -m gokturk.data_generation --list-families

python -m gokturk.data_generation --mode pro --target 10000 \
  --plan-only --plan-out data/raw/plan.json
# -> aile/seviye/rol dağılımı + kaba token maliyeti; hiçbir istek atılmaz
```

Gerçek üretim (doğrulanmış `EVREN_*` değişkenleri gerekir):

```bash
python -m gokturk.data_generation --mode pro --target 10000 \
  --out data/raw/pro.jsonl \
  --candidates 2          # istem başına aday yanıt
  # --no-critique         # öz-eleştiri adımını kapat (daha hızlı, daha düşük kalite)
  # --families coding,reasoning,system_architecture
  # --levels ileri
  # --plan-in data/raw/plan.json   # planı yeniden kullan
  # --seed 7 --resume
```

Her örnek için: (1) öğretmen bir Türkçe kullanıcı isteği taslaklar, (2) aynı
istem için `--candidates` kadar yanıt üretilir ve doğrulayıcılardan en iyi geçen
seçilir, (3) farklı roldeki bir öğretmen yanıtı ölçütlere göre denetler ve
gerekirse **yalnızca doğrulayıcılardan geçen** onarımı uygular, (4) şema/kalite
kapısı son sözü söyler. Geçmeyenler `*.rejects.jsonl`'e, eleştiri kayıtları
`*.critique.jsonl`'e yazılır.

Doğrulayıcılar **biçimi** denetler (JSON şeması, kod bloğu + dil, Türkçe oranı,
uzunluk, akıl yürütme sızıntısı, araç çağrısı anahtarları). Anlamsal doğruluk
otomatik ölçülmez — insan incelemesi zorunludur (`docs/architecture.md` §4–5).

**Uzun koşular** (2.000 slot ≈ 6.000 çağrı ≈ 6,7 saat @15 istek/dk) süre bütçesiyle
kesilebilir ve kaldığı yerden sürer:

```bash
python -m gokturk.data_generation --mode pro --target 2000 \
  --out data/raw/pro.jsonl --time-budget-min 540
# süre dolunca: stop_reason="time-budget", çıkış kodu 0 (hata değil)
# aynı komutu tekrar çalıştırın -> mevcut id'ler atlanır (resume)
```

Kaggle'da gözetimsiz koşu (6b/6c hücreleri) ve iki oturumlu devam akışı:
[`docs/veri-uretimi-kaggle.md`](docs/veri-uretimi-kaggle.md) · 2026-10-26 olayı ve
düzeltmeler: [`docs/uretim-olay-2026-10-26.md`](docs/uretim-olay-2026-10-26.md)

## Eğitim verisi üretimi (basic mod, API anahtarı gerekir)

```bash
cp .env.example .env        # .env git tarafından izlenmez
# .env içine yalnızca EVREN_API_KEY yazın (adres ve model kimlikleri
# configs/teachers.yaml içinde doğrulanmış olarak hazır)
set -a && source .env && set +a

python -m gokturk.data_generation --dry-run          # plan; ağa çıkmaz
python -m gokturk.data_generation --list-models      # sağlayıcının /models listesi
python -m gokturk.data_generation --health-check     # bağlantı + model kimliği doğrulaması
python -m gokturk.data_generation --per-seed 1 \
  --out data/raw/generated.jsonl
```

- Öğretmen çifti: **MiMo-V2.6-Pro** (`mimo-v2.6-pro`, akıl yürütme/genel) +
  **GLM-5.3** (`glm-5.3`, kod/araç çağırma). Kimi-K3 elendi: EVREN filosunda
  Kimi yok.
- `configs/teachers.yaml` içindeki `base_url`, `mimo-v2.6-pro` ve `glm-5.3`
  **doğrulanmış** değerlerdir (kaynak ve tarih dosyanın başında yazılı).
  Kimlikler değişirse tahmin edilmez: `--list-models` sağlayıcının kendi
  listesini alır, karşılaştırır ve eksik varsa çıkış kodu 2 ile durur.
- Anahtar yoksa üretim **hiç istek atmadan** durur (çıkış kodu 3).
- 401/403 → anında durur. Art arda hata eşiği aşılırsa koşu güvenli biçimde
  durur ve o ana kadar yazılan veri korunur (`--no-resume` ile sıfırdan).

## Taban model seçimi (v1)

| Profil | Taban | T4'te 4-bit QLoRA | CPU'da (2 vCPU) | Ne zaman |
| --- | --- | --- | --- | --- |
| `configs/training_unsloth_t4.yaml` | `openai/gpt-oss-20b` | sınırda (~14 GB, Unsloth) | çok yavaş | H200 sınıfı GPU çıkınca |
| `configs/training_qwen3_5_9b_t4.yaml` | `Qwen/Qwen3.5-9B` | **≈7.2 GB** (seq 1024) | sınırda (~1.5–3.6 tok/s) | önerilen v1 |
| `configs/training_qwen3_14b_t4.yaml` | `Qwen/Qwen3-14B` | ≈11.8 GB (seq 1024) | çok yavaş | metin-only, bugün çalışan yol |
| — | `Qwen/Qwen3.5-4B` | ≈3.2 GB | ~3–8 tok/s | CPU demosu |

Sayılar bu repodaki araçların çıktısıdır; gerekçe, HF Spaces ücretsiz kota
gerçeği ve göç kontrol listesi:
[`docs/taban-model-secimi.md`](docs/taban-model-secimi.md).

```bash
# Eğitim öncesi: sığar mı?
python -m gokturk.train --config configs/training_qwen3_5_9b_t4.yaml --plan-only
# Servis öncesi: CPU'da çalışır mı, kaç token/sn?
python -m gokturk.publish --serve-plan --base-model Qwen/Qwen3.5-9B --ram-gb 16 --vcpu 2
```

> Not (düzeltildi, 2026-10-25): Qwen3.5 ailesi görüntü+metin mimarisidir ama
> metin SFT'si için **AutoProcessor gerekmez** — transformers'ta
> `Qwen3_5ForCausalLM` görüntü kulesini (`^model.visual.*`) yok sayar, mevcut
> `AutoModelForCausalLM` + `AutoTokenizer` yolu yeterlidir. Bugünkü gerçek kısıt
> bellek değil **disktir**: 9B bf16 ≈18 GB, Kaggle'da 19,2 GB boş alan → ilk
> gerçek koşu `configs/training_qwen3_5_4b_t4.yaml` (4B, ≈9,3 GB) ile yapılır.

## Çevrimdışı eğitim (internet/DNS gerekmez)

Veri üretimi API'ye muhtaçtır, **eğitim değildir**: taban ağırlıklar yereldeyse
eğitim internet olmadan koşar.

```bash
python -m gokturk.train --offline \
  --config configs/training_qwen3_5_4b_cot_t4.yaml \
  --model /kaggle/input/qwen35-4b/Qwen3.5-4B \
  --train-file data/processed/train.jsonl --eval-file data/processed/valid.jsonl
```

`--offline` ağa hiç çıkmaz; yerel ağırlık veya veri yoksa **açık hata** verip durur
(9 GB indirmeyi DNS düşmüş makinede denemez). Kaggle'da 7b hücresi bunu otomatik
yapar; ayrıntı ve sınırlar: [`docs/cevrimdisi-egitim.md`](docs/cevrimdisi-egitim.md).

## Eğitim

```bash
# Ön denetim + plan (generic yol)
python -m gokturk.train --config configs/training.yaml --plan-only

# Ön denetim + plan (Unsloth 4-bit yolu — gpt-oss-20b'yi 16 GB karta sığdırır)
python -m gokturk.train --backend unsloth \
  --config configs/training_unsloth_t4.yaml --plan-only

# Gerçek koşu (donanım uygunsa)
python -m gokturk.train --config configs/training.yaml \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/valid.jsonl

# Kaggle 2×T4 için önerilen yol: önce 60 adımlık ölçüm koşusu
python -m gokturk.train --backend unsloth \
  --config configs/training_unsloth_t4.yaml \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/valid.jsonl
```

`--backend unsloth` Unsloth'un `FastLanguageModel`/`FastModel` API'sini kullanır,
Unsloth yoksa sessizce generic yola **düşmez**: açık hata verir. Yayımlanmış
referans rakamlar 4-bit QLoRA için ≈14 GB, 16-bit LoRA için ≈44 GB'dır; bu
repoda gerçek bir Kaggle koşusu doğrulanmamıştır, bu yüzden 60 adımlık ölçüm
koşusu varsayılandır.

Ön denetim engelleyici bir sorun bulursa betik **eğitimi başlatmaz**
(çıkış kodu 2) ve nedeni yazar. `--skip-preflight` yalnızca kendi sorumluluğunuzda
deneme içindir; yine de desteklenmeyen mimariyi destekliyormuş gibi göstermez.

Çıktılar: `outputs/<run>/adapter/` (LoRA ağırlıkları + tokenizer),
`outputs/<run>/run_manifest.json` (config, paket sürümleri, git commit, ön
denetim raporu, metrikler).

### Kaggle'da eğitim

`notebooks/kaggle_training.ipynb` dosyasını Kaggle'a yükleyin,
**Accelerator → GPU T4 x2**, **Internet → On** seçin, Secrets'a `EVREN_*`
değişkenlerini ekleyin. Notebook sırasıyla ortamı denetler, Unsloth dahil
paketleri kurar, veriyi doğrular, sentetik veri planını gösterir, duman testini
çalıştırır, iki arka ucu ön denetimden geçirir. 5b hücresi depodaki CoT dosyasını
şemaya çevirirse 8. hücre **CoT profilini** (`configs/training_qwen3_5_4b_cot_t4.yaml`)
seçer ve adım sayısını veriden hesaplar (`örnek × epoch / 16`, 60–1500); dosya
yoksa 4 örneklik boru hattı koşusunda kalır. Hiçbiri uygun değilse dürüstçe durur ve nedenini yazar
(bkz. `docs/architecture.md` §6–7).

Kaggle'da daha önce bulunmadıysanız veya log okumak zor geliyorsa:
**[`docs/kaggle-rehberi.md`](docs/kaggle-rehberi.md)** — GPU/oturum/kota
kavramları, her hücrenin ne yaptığı, `loss` ve `grad_norm` satırlarının nasıl
okunacağı, hata sözlüğü.

### Eğitim logunu makineye okutmak

```bash
python -m gokturk.logcheck --log kaggle_log.txt
python -m gokturk.logcheck --manifest outputs/gokturk-run/run_manifest.json
```

Dört karar verir: `SAĞLIKLI`, `DİKKAT`, `BOZUK` (NaN/ıraksama), `BAŞARISIZ`
(çökme). Çökme sebebini ve tek satırlık çözümü yazar. **Yalnızca eğitimin
mekaniğini denetler** — model kalitesi hakkında hiçbir şey söylemez; kalite
için `gokturk.evaluate` + insan incelemesi gerekir.

### GitHub Actions ile mini adaptör (CPU, GPU yok)

Boru hattını uçtan uca kanıtlayan küçük koşu; **GökTürk-Ultra'yı eğitmez**.

```bash
# Elle tetiklenir: Actions → "Mini adaptör (CPU)" → Run workflow
# Secret'lar (yalnızca gerektiğinde):
#   HF_TOKEN      → push_to_hub = true
#   EVREN_API_KEY → check_evren  = true  (yalnızca --health-check, veri üretmez)

# Aynı komutlar yerelde:
python -m gokturk.train --config configs/training_mini_cpu.yaml \
  --model Qwen/Qwen3-0.6B --cpu-ok --plan-only
```

Public depoda `ubuntu-latest` = 4 vCPU / 16 GB / **14 GB disk**, **GPU yok**;
GitHub Pro bu tabloyu değiştirmez. Sonuç: ≤1B modelde 30–100 adım LoRA evet,
9B adaptör hayır. `--cpu-ok`, "GPU yok" bulgusunu engelleyici olmaktan çıkarıp
bilgiye indirir; `--skip-preflight` gibi engelleri sessizce yutmaz.

[`docs/github-actions-mini-egitim.md`](docs/github-actions-mini-egitim.md)

## Veri inceleme akışı

Öğretmen çıktısı otomatik olarak doğru sayılmaz:

```bash
python -m gokturk.data_quality --data data/raw/generated.jsonl --out-dir data/processed
python -m gokturk.data_quality --data data/processed/train.jsonl \
  --review-sample 30 --review-out reports/review.md
```

İnceleme dosyasındaki her örnek **(D)** doğru / **(K)** kısmen / **(Y)** yanlış
olarak işaretlenir. Hata oranı eşiği aşarsa ilgili istem şablonu düzeltilir ve o
`template_version` ile üretilen veri atılır. Süreç: `docs/architecture.md` §4.

## Yayınlama: Hugging Face ve GGUF

```bash
# 1) Plan (ağa çıkmaz): dosyalar, boyut, token var mı
python -m gokturk.publish --adapter outputs/gokturk-run/adapter \
  --repo-id kullanici/gokturk-ultra-lora

# 2) Gerçek yükleme (özel depo; token yalnızca HF_TOKEN ortam değişkeninden)
python -m gokturk.publish --adapter outputs/gokturk-run/adapter \
  --repo-id kullanici/gokturk-ultra-lora --manifest outputs/gokturk-run/run_manifest.json --push

# 3) GGUF: iki yol, iki farklı cevap
python -m gokturk.publish --gguf-plan --adapter outputs/gokturk-run/adapter --quant mxfp4
python -m gokturk.lora_gguf --adapter outputs/gokturk-run/adapter \
  --base-model Qwen/Qwen3.5-4B          # adaptör-only: birleştirme YOK, ≈2 GB RAM
```

- Yükleme **adaptörü** gönderir (~yüz MB), 42 GB'lık birleşik modeli değil.
  Model kartı manifestten üretilir ve ölçülmemiş hiçbir başarı iddiası içermez.
- **GGUF otomatik üretilmez**, ama iki ayrı yol var ve karıştırılmamalı:
  * `--gguf-plan` → **birleştirme**: ≈67 GB RAM / ≈100 GB disk ister, Kaggle'da
    imkânsız (tek dosya dağıtımı gerekiyorsa).
  * `--adapter-only` (`python -m gokturk.lora_gguf`) → **birleştirme yok**:
    ≈2 GB RAM, ≈4 GB disk, 5–13 dakika. Kaggle'da **çalışır**; llama.cpp
    adaptörü yükleme anında tabana uygular.
  * Ayrıntı ve doğrulanmış kaynak atıfları: [`docs/gguf-donusumu.md`](docs/gguf-donusumu.md).
- gpt-oss'un doğal biçimi **MXFP4**'tür (uzmanlar zaten 4-bit). Q4_K_M
  desteklenir ama bu modelde ikinci bir yuvarlama ekler; varsayılan MXFP4.
- Ayrıntı: [`docs/yayinlama-ve-gguf.md`](docs/yayinlama-ve-gguf.md).

### Birleştirmeden servis (dinamik LoRA)

```bash
# Dört yolu karşılaştır: peft / vllm-lora / llamacpp / merged
python -m gokturk.serve --plan --base-model Qwen/Qwen3.5-9B --vram-gb 48

# HF Space iskeleti: taban + adaptör, birleştirme yok
python -m gokturk.serve --emit-space space/ \
  --base-model Qwen/Qwen3.5-9B --adapter-repo kullanici/gokturk-ultra-lora
```

| Yol | Birleştirme | Nerede |
| --- | --- | --- |
| `peft` | hayır | Python / Space |
| `vllm-lora` | hayır | GPU sunucusu (bnb tabanda dinamik desteklenmez) |
| `llamacpp` | hayır | CPU; `convert_lora_to_gguf.py` taban ağırlığı istemez |
| `merged` | **evet** | yalnızca tek dosya zorunluysa |

[`docs/servis-ve-dinamik-lora.md`](docs/servis-ve-dinamik-lora.md)

## Değerlendirme

```bash
# Tahminleri (id + response JSONL) sabit test setine göre skorla
python -m gokturk.evaluate --mode score \
  --eval-set data/eval/turkish_eval.jsonl \
  --predictions preds.jsonl \
  --out reports/eval.json --markdown reports/eval.md
```

Sabit set 16 öğe ve 4 kategoridir (Türkçe dil, talimat takibi, kodlama, araç
çağırma). Rapor ölçülmemiş kategorileri `ölçülmedi` olarak yazar; hiçbir sayı
kamuya açık benchmark yerine geçmez. Opsiyonel `--mode judge` öğretmen modeli
yargıç olarak kullanır ve **doğrulanmamış** olduğu raporda belirtilir.

## Testler

```bash
python -m pytest tests -q        # 406 test
ruff check src tests             # statik kontrol
```

Testler ağ erişimi gerektirmez: HTTP yolu yerel bir test sunucusuyla, eğitim
yolu ise rastgele başlatılmış küçük bir modelle (CPU) doğrulanır.

## Depo yapısı

```
configs/     teachers.yaml, training.yaml, training_unsloth_t4.yaml
data/        seeds/, eval/, samples/  (ham/işlenmiş veri .gitignore'da)
docs/        architecture.md, kaggle-rehberi.md, yayinlama-ve-gguf.md,
             taban-model-secimi.md, servis-ve-dinamik-lora.md,
             github-actions-mini-egitim.md, kaggle-gercek-kosus.md,
             gguf-donusumu.md
notebooks/   kaggle_training.ipynb
src/gokturk/ schema, config, teachers, task_specs, curriculum,
             data_generation, data_quality, env_check, unsloth_backend,
             train, evaluate, logcheck, publish, serve
tests/       406 otomatik test
```

## Lisans

Kod Apache-2.0 (`LICENSE`). Temel model (`openai/gpt-oss-20b`) Apache-2.0 +
kullanım politikasıdır; eğitimde kullanılacak veri setinin lisansı ayrıdır ve
bu repoda doğrulanmamıştır. Ayrıntı: `docs/architecture.md` §6.
