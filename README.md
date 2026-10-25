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
| Veri üretimi | `src/gokturk/data_generation.py` | Türkçe tohumlardan iki adımlı (taslak + yanıt) sentetik veri; yarıda kalırsa güvenli durur, devam ettirilebilir |
| Veri kalitesi | `src/gokturk/data_quality.py` | Şema doğrulama, bozuk JSON/boş/uzun/reddetme/tekrar filtreleri, yakın-tekrar temizliği, sızıntısız gruplu bölme |
| Ortam denetimi | `src/gokturk/env_check.py` | GPU/CUDA/VRAM/paket/mimari denetimi + bellek tahmini; engelleyici bulguları raporlar |
| Eğitim | `src/gokturk/train.py` | LoRA/QLoRA/full; prompt maskeleme; ön denetim geçmezse **başlamaz**; CPU duman testi |
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
pip install -e ".[qlora]"     # QLoRA / 4-bit NF4 (bitsandbytes, yalnızca CUDA)

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

## Eğitim verisi üretimi (API anahtarı gerekir)

```bash
cp .env.example .env        # .env git tarafından izlenmez
# .env içine EVREN_API_KEY, EVREN_BASE_URL, EVREN_MODEL_MIMO, EVREN_MODEL_KIMI yazın
set -a && source .env && set +a

python -m gokturk.data_generation --dry-run          # plan; ağa çıkmaz
python -m gokturk.data_generation --health-check     # bağlantı + model kimliği doğrulaması
python -m gokturk.data_generation --per-seed 1 \
  --out data/raw/generated.jsonl
```

- `configs/teachers.yaml` içindeki `base_url` ve model kimlikleri
  **kasten boştur** (`env:...` referansı). Sağlayıcının kendi belgesinden
  doğrulanmadan doldurulmaz; kod hiçbir adresi uydurmaz.
- Anahtar yoksa üretim **hiç istek atmadan** durur (çıkış kodu 3).
- 401/403 → anında durur. Art arda hata eşiği aşılırsa koşu güvenli biçimde
  durur ve o ana kadar yazılan veri korunur (`--no-resume` ile sıfırdan).

## Eğitim

```bash
# Ön denetim + plan
python -m gokturk.train --config configs/training.yaml --plan-only

# Gerçek koşu (donanım uygunsa)
python -m gokturk.train --config configs/training.yaml \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/valid.jsonl
```

Ön denetim engelleyici bir sorun bulursa betik **eğitimi başlatmaz**
(çıkış kodu 2) ve nedeni yazar. `--skip-preflight` yalnızca kendi sorumluluğunuzda
deneme içindir; yine de desteklenmeyen mimariyi destekliyormuş gibi göstermez.

Çıktılar: `outputs/<run>/adapter/` (LoRA ağırlıkları + tokenizer),
`outputs/<run>/run_manifest.json` (config, paket sürümleri, git commit, ön
denetim raporu, metrikler).

### Kaggle'da eğitim

`notebooks/kaggle_training.ipynb` dosyasını Kaggle'a yükleyin,
**Accelerator → GPU T4 x2**, **Internet → On** seçin, Secrets'a `EVREN_*`
değişkenlerini ekleyin. Notebook sırasıyla ortamı denetler, paketleri kurar,
veriyi doğrular, duman testini çalıştırır ve gerçek eğitimi dener. 2×T4 +
gpt-oss-20b birleşiminde eğitim **başlamaz** ve nedenini alternatifleriyle
yazar (bkz. `docs/architecture.md` §6).

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
python -m pytest tests -q        # 157 test
ruff check src tests             # statik kontrol
```

Testler ağ erişimi gerektirmez: HTTP yolu yerel bir test sunucusuyla, eğitim
yolu ise rastgele başlatılmış küçük bir modelle (CPU) doğrulanır.

## Depo yapısı

```
configs/     teachers.yaml, training.yaml
data/        seeds/, eval/, samples/  (ham/işlenmiş veri .gitignore'da)
docs/        architecture.md
notebooks/   kaggle_training.ipynb
src/gokturk/ schema, config, teachers, data_generation, data_quality,
             env_check, train, evaluate
tests/       157 otomatik test
```

## Lisans

Kod Apache-2.0 (`LICENSE`). Temel model (`openai/gpt-oss-20b`) Apache-2.0 +
kullanım politikasıdır; eğitimde kullanılacak veri setinin lisansı ayrıdır ve
bu repoda doğrulanmamıştır. Ayrıntı: `docs/architecture.md` §6.
