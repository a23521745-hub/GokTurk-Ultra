# Taban model seçimi — v1 için karar

Bu belge "gpt-oss-20b'yi Kaggle'da eğitemiyoruz, peki ne yapalım?" sorusuna
sayılarla cevap verir. Bütün sayılar ya bu repodaki araçların çıktısıdır ya da
kaynağı belirtilmiş yayımlanmış değerlerdir.

---

## 1. Sorunun gerçek çerçevesi

İki ayrı kısıt var ve karıştırılırsa yanlış model seçilir:

| Kısıt | Nerede | Ne demek |
| --- | --- | --- |
| **Eğitim** | Kaggle 2×T4 (16 GB/kart) | Model tek karta sığmalı; bu repo otomatik model paralelliği yapmaz |
| **Servis** | HF Spaces CPU Basic (2 vCPU, 16 GB RAM) | Ağırlık + KV önbelleği 16 GB'a sığmalı **ve** yanıt süresi katlanılabilir olmalı |

gpt-oss-20b eğitim tarafında sorunlu: bu repodaki tahmin aracı T4'te
**≈21 GB** diyor ve `vram-insufficient` engelleyicisini veriyor; Unsloth'un
yayımladığı ≈14 GB rakamı kendi özel çekirdeklerine ait ve T4'te bf16
olmadığı için ıraksama riski belgelenmiş. Yani "4-bit ile bir şekilde oluyor"
doğru ama **sınırda**; üstüne T4'te fp16 ıraksaması riski biniyor.

## 2. Adaylar (bu repodaki araçların çıktısı)

Eğitim tahmini: `check_environment(..., load_in_4bit=True, backend="unsloth")`,
T4 15.6 GB, 4-bit QLoRA. Servis tahmini:
`python -m gokturk.publish --serve-plan --base-model <model> --ram-gb 16 --vcpu 2`.

| Taban | Parametre | Eğitim (seq 1024) | Eğitim (seq 2048) | Q4_K_M ağırlık | bf16 indirme | CPU 2 vCPU | Karar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `openai/gpt-oss-20b` | 20.9B MoE | ≈12 GB, **engelleyici** | ≈21 GB, **engelleyici** | ≈11.2 GB | ≈13 GB (MXFP4) | 0.7–1.8 tok/s | ❌ eğitim sınırda |
| `Qwen/Qwen3-14B` | 14.8B dense, metin | ≈11.8 GB | ≈14.9 GB (tight) | ≈9.0 GB | **≈29,6 GB → disk yok** | 0.9–2.2 tok/s | ❌ Kaggle diskine sığmaz |
| `Qwen/Qwen3.5-9B` | 9B dense, çok kipli | **≈7.2 GB** | **≈9.1 GB** | ≈5.5 GB | ≈18 GB → **sınırda** | 1.5–3.6 tok/s | 🟡 2. koşu |
| **`Qwen/Qwen3.5-4B`** | **4,66B dense, çok kipli** | **≈3,2 GB** | **≈4,0 GB** | ≈2,4 GB | **≈9,3 GB → rahat** | **3.3–8.3 tok/s** | ✅ **ilk gerçek koşu** |
| `Qwen/Qwen3.5-27B` | 27B | ≈22 GB, sığmaz | — | ≈16.5 GB | ≈54 GB | 0.5–1.2 tok/s | ❌ T4'e ve diske sığmaz |

**Disk, VRAM'den önce gelir (2026-10-25 düzeltmesi).** Kaggle'da ölçülen boş
alan **19,2 GB** (`/kaggle/working` 20 GB). 4-bit eğitim için önce **bf16
ağırlıkları indirmek** gerekir; indirme diske sığmazsa VRAM hesabı anlamsızdır:

* `Qwen/Qwen3-14B` → ≈29,6 GB indirme **sığmaz** → `env_check` artık
  `weights-disk-insufficient` **engelleyici** bulgusu verir.
* `Qwen/Qwen3.5-9B` → ≈18 GB → `weights-disk-tight` **uyarısı** (adaptör, log ve
  llama.cpp kopyası için pay kalmaz).
* `Qwen/Qwen3.5-4B` → HF'teki toplam depolama **9,33 GB** (2 safetensors parçası,
  4.659.865.088 parametre) → rahat.

Bu yüzden **ilk gerçek Kaggle koşusu 4B ile** yapılır
(`configs/training_qwen3_5_4b_t4.yaml`, gerekçe ve adımlar:
[`docs/kaggle-gercek-kosus.md`](kaggle-gercek-kosus.md)). 4B ile boru hattı
kanıtlandıktan sonra 9B'ye geçilir; 9B'de disk payı için `max_seq_len` 1024,
`save_total_limit: 1` ve ara çıktı temizliği gerekir.

Kullanılan komutlar (tekrar üretilebilir):

```bash
python -m gokturk.train --config configs/training_qwen3_5_9b_t4.yaml --plan-only
python -m gokturk.publish --serve-plan --base-model Qwen/Qwen3.5-9B --ram-gb 16 --vcpu 2
```

Kimlikler ve parametre sayıları HF model API'sinden okundu (2026-10-25):
`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-27B` (Apache-2.0, gated
değil), `Qwen/Qwen3-14B` (14.8B, metin). `unsloth/Qwen3.5-9B-GGUF` mevcut.

## 3. Öneri

**v1 tabanı: `Qwen/Qwen3.5-9B`.** Gerekçe:

1. Kaggle'da rahat (≈7.2 GB @ seq 1024), yani 60 adımlık ölçüm koşusu OOM
   riski olmadan koşar ve tam koşuya yer kalır.
2. GGUF'u ≈5.5 GB → 2 vCPU'da "sınırda ama çalışır"; 16 GB RAM'e rahat sığar.
3. Güncel nesil (2026-02), Apache-2.0, Unsloth GGUF'ları hazır.
4. Aynı aile 4B/27B'ye açık: veri seti ve boru hattı değişmeden ölçek değişir.

**Düzeltme (2026-10-25): "AutoProcessor gerekir" iddiası yanlıştı.**

- Qwen3.5 görüntü+metin (`Qwen3_5ForConditionalGeneration`) ama transformers'ta
  `Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected = [r"^mtp.*",
  r"^model.visual.*"]` tanımlı. Yani görüntü kulesi ağırlıkları yüklenirken
  **yok sayılır** ve mevcut `AutoTokenizer` + `AutoModelForCausalLM` metin yolu
  yeterlidir; `AutoProcessor`/`AutoModelForImageTextToText` eklemek
  **gerekmez** (kaynak: transformers `models/qwen3_5/modeling_qwen3_5.py`).
  Bu bir kod okuma bulgusudur: gerçek koşuda yükleme logunda `model.visual.*`
  "unexpected key" olarak görünmelidir — Kaggle koşusunun 1. kontrolü budur.
- Gerçek kısıt **disk**: bkz. yukarıdaki tablo. 14B'nin 29,6 GB bf16 indirmesi
  Kaggle'ın 19,2 GB boş alanına sığmaz, bu yüzden 14B artık önerilmez.

Kısa karar: **ilk gerçek koşu Qwen3.5-4B** (disk rahat, VRAM rahat), **ikinci
koşu Qwen3.5-9B** (disk sınırda, `max_seq_len: 1024`). 14B rafa kalktı.

## 4. HF Spaces gerçeği (planı değiştiren kısım)

Bu, "16 GB RAM / 2 vCPU / 50 GB" kısıtından **daha** belirleyici:

- HF dokümanına göre **yeni** Gradio/Docker Space oluşturmak artık **ücretli
  plan** istiyor (PRO/Team). Ücretsiz kişisel hesaplar yalnızca **2 adet
  ZeroGPU Gradio Space** barındırabiliyor.
- ZeroGPU kotası: **ücretsiz hesap için günde 5 dakika** GPU (PRO 40 dk,
  giriş yapmamış 2 dk). GPU: RTX Pro 6000 Blackwell, **48 GB**.
- CPU Basic (2 vCPU / 16 GB) hâlâ ücretsiz ama **yeni oluşturmak** için plan
  gerekiyor; ayrıca 48 saat kullanılmazsa uykuya geçiyor.
- Depolama **kalıcı değil**: 50 GB disk yeniden başlatmada silinir; ağırlıklar
  her uyanışta yeniden indirilir (ilk açılış 30–90 sn + indirme).

Sonuç: "Ücretsiz Spaces'te 7/24 Türkçe asistan" **mümkün değil**. Gerçekçi
olan iki senaryo:

| Senaryo | Donanım | Ne olur |
| --- | --- | --- |
| **ZeroGPU demo** | 48 GB GPU, günde 5 dk | 9B Q4 çok hızlı; günde ~onlarca kısa yanıt. Demo, ürün değil. |
| **CPU Basic** | 2 vCPU, 16 GB | 9B Q4 sığar, 1.5–3.6 tok/s: 300 token ≈ 1,5–3,5 dk. Sadece "çalışıyor" kanıtı. |
| **CPU'da sohbet** | 2 vCPU, 16 GB | 4B Q4 (3.3–8.3 tok/s) anca katlanılabilir; kalite düşer. |

Yani servis hedefi, taban modeli **4B'ye kadar** aşağı çekebilir. Bu bir zevk
meselesi değil, aritmetik.

## 5. "Knowledge distillation" tam olarak ne oluyor

Doğru kısım: öğretmen modellerin (MiMo-V2.6-Pro, GLM-5.3) ürettiği yanıtları
öğrenci modele SFT ile öğretmek, literatürde **sequence-level distillation**
olarak geçer. "Mini hâlini aktarıyoruz" sezgisi bu çerçevede doğru.

Eksik kalan kısım: klasik KD, öğretmenin **token olasılık dağılımını** (logit)
da kullanır. Biz onu görmüyoruz — API yalnızca metin döndürüyor. Yani:

| Aktarılan | Aktarılmayan |
| --- | --- |
| Üslup, format, dil, görev takibi, örnek çözümler | Öğretmenin kendi belirsizlik/dağılım bilgisi |
| "Bu soruya böyle cevap verilir" | "Bu soruda hangi token ne kadar olası" |

Sonuç: 10.000 örnek, 9B'lik bir modelde **ciddi** bir davranış değişimi
yaratır; ama model asla öğretmenin kapasitesine eşitlenmez. "Ultra" iddiası
ölçümle desteklenmedikçe yazılmamalı.

## 6. Değiştirmek için ne gerekiyor (durum)

| Parça | Durum |
| --- | --- |
| Eğitim profilleri | ✅ `configs/training_qwen3_5_9b_t4.yaml`, `configs/training_qwen3_14b_t4.yaml` |
| Ön denetim sayıları | ✅ `env_check.KNOWN_MODELS` içine Qwen girdileri eklendi |
| GGUF/servis planı | ✅ `publish.BASE_PARAMS_B` + `--serve-plan` (dense/MoE ayrımı doğru) |
| Metin yolu (`AutoTokenizer` + `AutoModelForCausalLM`) | ✅ Qwen3-14B **ve Qwen3.5** için hazır (görüntü kulesi `^model.visual.*` ile yok sayılır; 2026-10-25 düzeltmesi) |
| Disk denetimi | ✅ `env_check` `weights-disk-insufficient` / `weights-disk-tight` üretir (Kaggle 19,2 GB ölçümüyle uyumlu) |
| 4B profili | ✅ `configs/training_qwen3_5_4b_t4.yaml` (ilk gerçek koşu) |
| Veri/plan/öğretmen hattı | ✅ tabandan bağımsız, değişmiyor |

## 7. Ya EVREN HPC eğitimi açılırsa

EVREN'in HPC kümesi (8 düğüm × 8 H200, 141 GB HBM) zaten eğitim için
kullanılabiliyor; düğüm başına 141 GB ile 20B/27B sınıfı 4-bit QLoRA rahat
sığar. O kapı açılırsa **gpt-oss-20b tekrar masaya gelir**. Bu yüzden taban
model hiçbir yere gömülmedi: config + `KNOWN_MODELS` + `BASE_PARAMS_B`
üçlüsünden biri değişince bütün araçlar (preflight, plan, GGUF, servis)
yeni modelle konuşur.

Karar sırası önerisi:

1. **Bugün:** 10k veri üret → **Qwen3.5-4B** ile 60 adım ölçüm → değerlendir
   ([`docs/kaggle-gercek-kosus.md`](kaggle-gercek-kosus.md)).
2. **Sonra:** Qwen3.5-9B'ye geçiş (disk sınırda; `max_seq_len: 1024`,
   `save_total_limit: 1`).
3. EVREN HPC eğitimi açıldığında 20B'ye dönüş (aynı veri setiyle).
