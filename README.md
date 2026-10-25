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
çalıştırır, iki arka ucu ön denetimden geçirir ve uygun olanla **60 adımlık
ölçüm koşusu** yapar. Hiçbiri uygun değilse dürüstçe durur ve nedenini yazar
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

# 3) GGUF mümkün mü? (makinenin gerçek RAM/disk sayılarıyla)
python -m gokturk.publish --gguf-plan --adapter outputs/gokturk-run/adapter --quant mxfp4
```

- Yükleme **adaptörü** gönderir (~yüz MB), 42 GB'lık birleşik modeli değil.
  Model kartı manifestten üretilir ve ölçülmemiş hiçbir başarı iddiası içermez.
- **GGUF otomatik üretilmez.** LoRA birleştirmesi ≈42 GB bellek ister; Kaggle
  ücretsiz katmanı (29 GB RAM, 20 GB disk) bunu kaldırmaz. Plan komutu bunu
  sayılarla söyler; yeterli makinede 5 adımlık komut listesini basar.
- gpt-oss'un doğal biçimi **MXFP4**'tür (uzmanlar zaten 4-bit). Q4_K_M
  desteklenir ama bu modelde ikinci bir yuvarlama ekler; varsayılan MXFP4.
- Ayrıntı: [`docs/yayinlama-ve-gguf.md`](docs/yayinlama-ve-gguf.md).

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
python -m pytest tests -q        # 297 test
ruff check src tests             # statik kontrol
```

Testler ağ erişimi gerektirmez: HTTP yolu yerel bir test sunucusuyla, eğitim
yolu ise rastgele başlatılmış küçük bir modelle (CPU) doğrulanır.

## Depo yapısı

```
configs/     teachers.yaml, training.yaml, training_unsloth_t4.yaml
data/        seeds/, eval/, samples/  (ham/işlenmiş veri .gitignore'da)
docs/        architecture.md, kaggle-rehberi.md, yayinlama-ve-gguf.md
notebooks/   kaggle_training.ipynb
src/gokturk/ schema, config, teachers, task_specs, curriculum,
             data_generation, data_quality, env_check, unsloth_backend,
             train, evaluate, logcheck, publish
tests/       297 otomatik test
```

## Lisans

Kod Apache-2.0 (`LICENSE`). Temel model (`openai/gpt-oss-20b`) Apache-2.0 +
kullanım politikasıdır; eğitimde kullanılacak veri setinin lisansı ayrıdır ve
bu repoda doğrulanmamıştır. Ayrıntı: `docs/architecture.md` §6.
