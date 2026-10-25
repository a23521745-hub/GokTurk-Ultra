# GitHub Actions ile mini adaptör eğitimi

Bu belge, GitHub'ın **kendi barındırdığı runner'larında** küçük bir LoRA koşusunun
nasıl yapıldığını anlatır. Kısa özet: **bu yol "GökTürk-Ultra"yı eğitmez**; boru
hattını (veri → ön denetim → eğitim → manifest → log denetimi → adaptör-only GGUF
→ isteğe bağlı HF yüklemesi) uçtan uca kanıtlar. Gerçek 9B adaptörü hâlâ
Kaggle/EVREN HPC işidir.

İlgili dosyalar:

| Dosya | Ne işe yarar |
| --- | --- |
| `.github/workflows/mini-adapter.yml` | Elle tetiklenen iş akışı (2 iş: `evren`, `mini`) |
| `configs/training_mini_cpu.yaml` | CPU'ya uygun mini eğitim profili (0.6B, fp32, LoRA) |
| `tests/test_mini_workflow.py` | İş akışının sözleşmesini koruyan 21 test |
| `docs/servis-ve-dinamik-lora.md` | Adaptör çıktısının nasıl servis edileceği |

---

## 1. Önce beklentiyi doğru kurun

GitHub Pro'nun buradaki tek katkısı **Actions dakikası**: depo **public** olduğu
için standart runner'lar zaten ücretsiz ve kotasız. Pro, GPU veya disk eklemez.

| Kaynak | Public depo (bu depo) | Private depo |
| --- | --- | --- |
| Runner | `ubuntu-latest` = `ubuntu-24.04` | aynı |
| vCPU | **4** | 2 |
| RAM | **16 GB** | 8 GB |
| Disk (SSD) | **14 GB** | 14 GB |
| GPU | **yok** | yok |
| Ücret | ücretsiz, kotasız | Pro dakikalarından düşer |

14 GB disk, ağır işleri kendi başına eler: CUDA'lı PyTorch tekerleği tek başına
≈2,5 GB, bir 9B modelin bf16 ağırlıkları ≈18 GB. Bu yüzden iş akışı önce
**CPU tekerleğini** kurar (`--index-url .../whl/cpu`), sonra `.[train]` ekstrasını
kurar; pip böylece CUDA tekerleğini indirmez.

> `ubuntu-slim` runner'ı **kullanılmaz**: 1 vCPU / 5 GB ve ayrıcalıksız konteyner,
> eğitim için yetersiz (Docker-in-Docker da yok).

**Ne gerçekçi:** ≤1B parametreli taban model + yüzlerce örnek + 30–100 adım LoRA.
**Ne değil:** 4B/9B/20B eğitimi, gerçek 10.000 örneklik koşu, kalite ölçümü.

---

## 2. Kaydedilecek secret'lar (tam adlar)

`Settings → Secrets and variables → Actions → New repository secret`, ya da CLI:

```bash
# Eğitim işi HF'ye adaptör yükleyecekse (yazma izinli token):
gh secret set HF_TOKEN --repo a23521745-hub/GokTurk-Ultra

# EVREN bağlantısını doğrulamak / veri üretmek için:
gh secret set EVREN_API_KEY --repo a23521745-hub/GokTurk-Ultra
```

| Secret adı | Zorunlu mu | Nerede kullanılıyor | Not |
| --- | --- | --- | --- |
| `HF_TOKEN` | Yalnızca `push_to_hub = true` | `mini` işinin yükleme adımı | HF → *Write* token. Yoksa adım `::error::` ile durur |
| `EVREN_API_KEY` | Yalnızca `check_evren = true` | `evren` işi, `--health-check` | Adres `configs/teachers.yaml` içinde; ayrı bir URL secret'ı **gerekmez** |
| `EVREN_BASE_URL` | **Gerekmez** | — | Adresi değiştirmek isterseniz `teachers.yaml`'da `base_url: "env:EVREN_BASE_URL"` yazıp bu değişkeni tanımlayın |

Kurallar:

* Anahtar **yalnızca** `secrets.*` üzerinden `env:`'e geçer; `run:` içine
  gömülmez, `echo` edilmez. Loglarda maskelenir.
* Fork'tan gelen PR'lar secret görmez; bu yüzden iş akışı `workflow_dispatch` ve
  `pull_request` değil — kazara maliyet de olmaz.
* `.env` ve `*.jsonl` çıktıları repoya girmez (bkz. `.gitignore`).

EVREN anahtarını doğrulamanın en hızlı yolu, iş akışını `check_evren = true` ile
bir kez çalıştırmak: `evren` işi `python -m gokturk.data_generation --health-check`
koşar, yalnızca bağlantıyı ve anahtarı sınar, **veri üretmez** (ücret oluşmaz).
Anahtar yoksa komut `3` ile çıkar ve iş kırmızı olur — sessizce geçmez.

---

## 3. İş akışını çalıştırma

1. **Actions → Mini adaptör (CPU) → Run workflow**.
2. Girdiler:

| Girdi | Varsayılan | Ne zaman değiştirilir |
| --- | --- | --- |
| `base_model` | `Qwen/Qwen3-0.6B` | ≤1B kalan herhangi bir metin modeli (tabloya eklemek için `env_check.KNOWN_MODELS`) |
| `max_steps` | `30` | 30–100. 100'ün üstü 90 dakikalık zaman aşımına yaklaşır |
| `train_file` | `data/samples/sample_pro.jsonl` | Kendi JSONL'iniz repoda ise yolu verin |
| `eval_file` | `data/samples/sample_train.jsonl` | Aynı şema (`prompt`/`response`) |
| `build_gguf_lora` | `true` | llama.cpp `convert_lora_to_gguf.py` ile adaptör-only GGUF |
| `push_to_hub` | `false` | `true` ise `HF_TOKEN` + `adapter_repo` şart |
| `adapter_repo` | boş | `kullanici/gokturk-mini-lora` biçiminde |
| `check_evren` | `false` | `true` ise ayrı `evren` işi koşar |

3. Bittiğinde **Artifacts → `mini-adapter-<run_number>`**: `adapter/`,
   `run_manifest.json`, `gokturk-mini-lora.gguf`, `train.log`, `logcheck.txt`,
   `plan.txt` (14 gün saklanır).

### Adımların sırası ve neden böyle

1. **Donanım baskısı** — `nproc`, `free -h`, `df -h`, `nvidia-smi` var mı. GPU
   olmadığını gizlemek yerine ekrana yazıyoruz.
2. **Kurulum** — CPU torch, sonra `.[train,dev]`.
3. **ruff** — statik kontrol, eğitimden önce.
4. **`--plan-only`** — ön denetim raporu + çözümlenmiş plan JSON'u.
5. **Eğitim** — `--cpu-ok` (aşağıya bakın).
6. **`gokturk.logcheck --manifest ... --fail-on broken`** — mekanik sağlık:
   NaN kayıp, ıraksama, erken durma. Ölçüt `broken`: "dikkat" seviyesi işi
   kırmaz, gerçek bozukluk kırar.
7. **GGUF LoRA** — `--merge` **yok**; yalnızca `lora_A`/`lora_B` dönüştürülür.
8. **HF yükleme** — yalnızca istendiğinde, token yoksa açık hatayla. Depo
   **özel (private)** açılır; herkese açık istiyorsanız yüklemeyi yerelde
   `python -m gokturk.publish ... --public` ile tekrarlayın.
9. **Artefakt + özet** — özet tablosu "bu GökTürk-Ultra değildir" notunu taşır.

### GGUF adımının iki tuzağı (2026-10-25'te doğrulandı)

* **`--base` dizin ister, HF kimliği istemez.** `convert_lora_to_gguf.py --base`
  yerel bir dizinde `config.json`/`tokenizer.json` arar. Uzak taban için
  `--base-model-id Qwen/Qwen3-0.6B` kullanılır; bu yalnızca **config**'i indirir,
  ağırlıkları indirmez — birleştirme yapılmaz.
* **`llama.cpp/requirements.txt` kurulmaz.** O dosya `torch==2.11.0` ister ve
  `--extra-index-url` ile PyPI'ı da açık bırakır; 14 GB'lık runner diskine CUDA
  tekerleği (≈2,5 GB + `nvidia-*` bağımlılıkları) inebilir. Dönüştürücü yalnızca
  `torch`, `transformers`, `safetensors`, `gguf` ister; ilk üçü eğitim
  ekstrasıyla gelir, bu yüzden adım sadece `pip install gguf` yapar.

---

## 4. `--cpu-ok` nedir, `--skip-preflight`'tan farkı ne?

Ön denetim (`gokturk.env_check`), GPU yoksa engelleyici `no-gpu` bulgusu üretir
ve eğitim **başlamaz**. Bu davranış doğrudur ve korunur.

CPU mini koşusu için iki yol var:

| Yol | Anlamı | Bu repoda |
| --- | --- | --- |
| `--skip-preflight` | Engelleyici bulguları **yok say**, manifest `preflight.ok = false` kalır; `logcheck` manifesti "başarısız" okur | Kullanılmaz |
| `--cpu-ok` | GPU yokluğu engelleyici **değil**, `cpu-mini` bilgi bulgusu olarak raporlanır; ön denetim yine çalışır, diğer engeller (ör. `gptoss-full-unsupported`) aynen durur | Kullanılır |

Ek olarak `configs/training_mini_cpu.yaml` içinde `require_preflight_ok: false`
var; bu, yapılandırma tarafındaki ikinci kapıyı açar (CLI bayrağı olmadan da
çalışabilsin diye). İki kapı da **açıkça** yazılır, sessiz geçiş yoktur.

`--cpu-ok` ile 1,5B üzeri bir model seçerseniz `cpu-mini-too-big` **uyarısı**
çıkar: 4 vCPU'da adım süresi dakikalara çıkar.

---

## 5. Dürüst sınırlar

* **Kalite iddiası yok.** 3–6 örnekle 30 adım atan bir 0.6B model, "Türkçe
  talimat takibi" öğrenmez. Koşunun manifesti `kind: training`, veri sayısı ve
  `train_loss` ile birlikte bunu kaydeder; `logcheck` "eğitim seti çok küçük"
  uyarısını basar.
* **9B burada eğitilmez.** Disk 14 GB, RAM 16 GB, GPU yok. Qwen3.5-9B'nin
  4-bit Kaggle profili (`configs/training_qwen3_5_9b_t4.yaml`) 2×T4 ister.
* **Süre.** 0.6B / fp32 / 512 token / 30 adım, 4 vCPU'da büyük olasılıkla
  10–30 dakika. 100 adımda 90 dakikalık zaman aşımına dikkat.
* **HF Spaces'te servis** bu iş akışının konusu değil: adaptörü indirip
  `python -m gokturk.serve --emit-space` ile yayınlayın
  (bkz. `docs/servis-ve-dinamik-lora.md`). 2 vCPU'da 0.6B gerçekten konuşur;
  9B konuşmaz.
* **Uzun koşular için doğru yer:** Kaggle notebook (ücretsiz T4) veya EVREN HPC.
  GitHub runner'ı orkestrasyon ve doğrulama için iyi, eğitim için dar.

---

## 6. Yerel ön izleme

Aynı komutları kendi makinenizde, ağa çıkmadan deneyebilirsiniz:

```bash
python -m gokturk.train --config configs/training_mini_cpu.yaml \
  --model Qwen/Qwen3-0.6B --cpu-ok --plan-only

# Gerçek mini koşu (torch + transformers + peft kurulu olmalı):
python -m gokturk.train --config configs/training_mini_cpu.yaml \
  --model Qwen/Qwen3-0.6B --cpu-ok --max-steps 30 --output-dir outputs/mini

python -m gokturk.logcheck --manifest outputs/mini/run_manifest.json --fail-on broken
```
