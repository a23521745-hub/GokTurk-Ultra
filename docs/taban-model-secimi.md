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

| Taban | Parametre | Eğitim (seq 1024) | Eğitim (seq 2048) | Q4_K_M ağırlık | CPU 2 vCPU | Karar |
| --- | --- | --- | --- | --- | --- | --- |
| `openai/gpt-oss-20b` | 20.9B MoE | ≈12 GB, **engelleyici** | ≈21 GB, **engelleyici** | ≈11.2 GB | 0.7–1.8 tok/s | ❌ eğitim sınırda |
| `Qwen/Qwen3-14B` | 14.8B dense, metin | ≈11.8 GB | ≈14.9 GB (tight) | ≈9.0 GB | 0.9–2.2 tok/s | 🟡 sınırda |
| **`Qwen/Qwen3.5-9B`** | 9B dense, çok kipli | **≈7.2 GB** | **≈9.1 GB** | ≈5.5 GB | 1.5–3.6 tok/s | ✅ **önerilen** |
| `Qwen/Qwen3.5-4B` | 4B dense, çok kipli | ≈3.2 GB | ≈4.0 GB | ≈2.4 GB | 3.3–8.3 tok/s | ✅ CPU demosu için |
| `Qwen/Qwen3.5-27B` | 27B | ≈22 GB, sığmaz | — | ≈16.5 GB | 0.5–1.2 tok/s | ❌ T4'e sığmaz |

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

**Ama iki uyarı:**

- **Çok kipli.** Qwen3.5 görüntü+metin (`Qwen3_5ForConditionalGeneration`).
  Metin SFT'si çalışır ama yükleyici `AutoProcessor` ister ve görüntü kulesi
  bellekte yer kaplar. Bu repodaki `train.py` metin yolu `AutoTokenizer` +
  `AutoModelForCausalLM` kullanıyor; **Qwen3.5 ile gerçek koşudan önce**
  `AutoModelForImageTextToText`/`AutoProcessor` yolunun eklenmesi gerekir.
  Bu bir TODO'dur, çözülmüş değil.
- Bu yüzden **hızlı ve kesin çözüm** olarak `Qwen/Qwen3-14B` profili de hazır
  (`configs/training_qwen3_14b_t4.yaml`): metin-only, mevcut kod yolu bugün
  olduğu gibi çalışır, seq 1024'te ≈11.8 GB ile T4'e sığar. Bedeli: 2048
  bağlamda sınırda (≈14.9 GB) ve CPU'da daha yavaş.

Kısa karar: **bugün çalışsın** isteniyorsa Qwen3-14B (seq 1024), **en iyi
sonuç** isteniyorsa Qwen3.5-9B + çok kipli yükleyici TODO'su.

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
| Metin yolu (`AutoTokenizer` + `AutoModelForCausalLM`) | ✅ Qwen3-14B için hazır; ❌ Qwen3.5 çok kipli yolu **yok** |
| Veri/plan/öğretmen hattı | ✅ tabandan bağımsız, değişmiyor |

## 7. Ya EVREN HPC eğitimi açılırsa

EVREN'in HPC kümesi (8 düğüm × 8 H200, 141 GB HBM) zaten eğitim için
kullanılabiliyor; düğüm başına 141 GB ile 20B/27B sınıfı 4-bit QLoRA rahat
sığar. O kapı açılırsa **gpt-oss-20b tekrar masaya gelir**. Bu yüzden taban
model hiçbir yere gömülmedi: config + `KNOWN_MODELS` + `BASE_PARAMS_B`
üçlüsünden biri değişince bütün araçlar (preflight, plan, GGUF, servis)
yeni modelle konuşur.

Karar sırası önerisi:

1. **Bugün:** Qwen3-14B ile 10k veri üret → 60 adım ölçüm → değerlendir.
2. Qwen3.5-9B'ye geçiş, çok kipli yükleyici eklendikten sonra.
3. EVREN HPC eğitimi açıldığında 20B'ye dönüş (aynı veri setiyle).
