# `.gguf` ne zaman elimde olur? — adaptör-only yol

Kısa cevap: **birleştirme yapmadan, bugün, ~15–30 dakikada.** Saatler değil.
Ve evet, eminim — ama "mekanik olarak geçerli bir `.gguf`" ile "yayınlanabilir
bir `.gguf`" farkını aşağıda ayırıyorum.

---

## 1. Neden saatler değil: yanlış işi planlıyorduk

Kaggle çıktısında şu satır vardı:

```
GGUF bu makinede yapılabilir mi: HAYIR
✗ RAM yetersiz: birleştirme ≈66.9 GB ister, makinede ≈33.7 GB var.
✗ Disk yetersiz: ≈99.7 GB gerekir, boş ≈20.5 GB.
```

Bu hesap **birleştirme (merge)** içindir: `merge_and_unload()` modelin tamamını
belleğe alır, fp16 kopya yazar, sonra `convert_hf_to_gguf.py` tek dosya üretir.
4B için doğru sayılar ve doğru sonuç: Kaggle'da **yapılamaz**.

Ama tek dosya dağıtımı zorunlu değil. `convert_lora_to_gguf.py` **yalnızca
adaptörü** çevirir: taban modelin sadece `config.json`'ını okur, ağırlıklarına
dokunmaz, çıktı ~40–110 MB'lık bir `.gguf` olur. llama.cpp bunu yükleme anında
tabana uygular:

```bash
llama-server -m base-Q4_K_M.gguf --lora gokturk-lora-qwen35.gguf
```

| | Birleştirme yolu | **Adaptör-only yol** |
| --- | --- | --- |
| RAM | ≈67 GB | **≈2 GB** |
| Disk | ≈100 GB | **≈4 GB** (taban GGUF dahil) |
| Süre (makine) | 30–90 dk + quantise | **4–13 dk** |
| Kaggle'da | ❌ | **✅** |
| Çıktı | tek dosya | taban + küçük `.gguf` |

## 2. Emin miyiz? llama.cpp tarafında doğrulananlar (2026-10-25)

Kaynak koda bakıldı, tahmin yok:

| Kontrol | Sonuç |
| --- | --- |
| `conversion/qwen.py` | `Qwen3_5ForConditionalGeneration`, `Qwen3_5ForCausalLM` kayıtlı → `MODEL_ARCH.QWEN35` |
| `src/models/qwen35.cpp` | var → **runtime** desteği tamam |
| `MODEL_TENSORS[QWEN35]` | `ATTN_Q/K/V/OUT`, `FFN_GATE/DOWN/UP` var → bizim `target_modules` birebir |
| `src/llama-adapter.cpp` | tek koşul: "model arch and LoRA arch mismatch" — mimariye özel LoRA yasağı yok |
| `convert_lora_to_gguf.py` | `--base-model-id` ile Hub kimliği kabul ediyor (`--base` **dizin** ister, kimlik kabul etmez) |
| Taban GGUF | `unsloth/Qwen3.5-4B-GGUF` hazır (~1,2 M indirme) |

Yani teknik engel yok. Tek gerçek risk, dönüştürücünün reddettiği adaptör
şekilleri — ve bizim eğitimimiz onlara girmiyor:

* **DoRA** → desteklenmiyor (bizde `use_dora: false`).
* **`embed_tokens` / `lm_head`** içeren adaptör → reddedilir (bizim 7 modülümüz:
  q/k/v/o/gate/up/down).
* **Taban uyuşmazlığı** → adaptör `Qwen/Qwen3.5-4B` için eğitildiyse taban GGUF
  da aynı model olmalı.

## 3. Adım adım (Kaggle'da ya da yerelde)

```bash
# 0) Plan: ne kadar yer/zaman gerekiyor? (ağa çıkmaz)
python -m gokturk.lora_gguf --adapter /kaggle/working/run-4b/adapter \
  --base-model Qwen/Qwen3.5-4B

# 1) Hazır taban GGUF (Q4_K_M ≈2,8 GB) — bf16 indirmeye gerek yok
pip install -q "huggingface_hub[cli]"
huggingface-cli download unsloth/Qwen3.5-4B-GGUF \
  --include "*Q4_K_M*.gguf" --local-dir gguf/base

# 2) Dönüştürücü (klon ~10 sn, saf Python; cmake gerekmez)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q gguf

# 3) ADAPTÖRÜ çevir — taban ağırlıkları indirilmez, birleştirme YOK
python llama.cpp/convert_lora_to_gguf.py /kaggle/working/run-4b/adapter \
  --base-model-id Qwen/Qwen3.5-4B \
  --outfile gguf/gokturk-lora-qwen35.gguf \
  --outtype f16

# 4) Doğrula: taban + adaptör birlikte yükleniyor mu?
llama.cpp/build/bin/llama-server -m gguf/base/*Q4_K_M*.gguf \
  --lora gguf/gokturk-lora-qwen35.gguf --port 8080
```

Adım 4 için llama.cpp'yi derlemek gerekir (`cmake -B build && cmake --build build -j`);
bu 10–20 dakika sürer ve **isteğe bağlıdır** — `.gguf` dosyası adım 3'te zaten
üretilmiş olur. Doğrulamayı bir kez yapmak yeterlidir.

Bu repoda plan + çalıştırma:

```bash
# Yalnızca plan (JSON ile makine okunur):
python -m gokturk.lora_gguf --adapter adapter/ --base-model Qwen/Qwen3.5-4B --json

# llama.cpp kopyası hazırsa dönüşümü gerçekten çalıştır:
python -m gokturk.lora_gguf --adapter adapter/ --base-model Qwen/Qwen3.5-4B \
  --run llama.cpp
```

## 4. Zaman çizelgesi (gerçekçi)

| Aşama | Süre | Not |
| --- | --- | --- |
| Taban GGUF indirme (2,8 GB) | 2–6 dk | Bağlantıya göre; Kaggle'da ~50 MB/s |
| llama.cpp klonu + `pip install gguf` | 2–4 dk | Derleme yok |
| **Adaptör dönüşümü** | **1–3 dk** | 42–110 MB okur, f16 yazar |
| Doğrulama (llama-server yükleme) | 1–2 dk | Derleme yapılmışsa |
| (İsteğe bağlı) llama.cpp derleme | 10–20 dk | Yalnızca bir kez |
| **Toplam, .gguf elinde** | **≈5–13 dk** | Derleme hariç |

Yani "kaç saat kaldı" sorusunun cevabı: **0 saat** — dosya bugün üretilebilir.

## 5. Ama dürüst olalım: elimizdeki adaptör demo

Son Kaggle koşusu **4 örnekle 60 epoch** çalıştı:

```
örnek sayısı: {'train_examples': 4, 'eval_examples': 1}
train_loss 0.0777  vs  eval_eval_loss 2.2638      (ezberleme)
```

Bundan çıkan `.gguf` **mekanik olarak geçerli**, ama içerik olarak işe yaramaz:
4 örneği ezberlemiş bir adaptördür. Yani:

* "`.gguf` üretebiliyor muyuz?" → **evet, bugün, dakikalar içinde.**
* "Yayınlanabilir bir `.gguf` var mı?" → **hayır**; önce gerçek veriyle eğitim.

Gerçek koşu için gerekenler:

1. `data/processed/train.jsonl` (≥9.000 örnek) + `valid.jsonl` — EVREN ile 10k üretim
   ya da kendi veri setiniz.
2. `max_steps` 60 değil **600–1200**; `epoch` 60 değil ~1–2.
3. Kaggle T4'te 4B / 4-bit / seq 1024 ile tahmini süre: ~7 s/adım × 900 adım ≈
   **1,8 saat** → 12 saatlik oturuma rahat sığar (ölçülen 418 s / 60 adım = 7 s/adım).

Yani gerçek adaptör: aynı gün içinde eğitim (≈2 saat) + dönüşüm (≈10 dk).

## 6. EVREN H200 notu

EVREN'de şu an ücretsiz H200 var ama **eğitim** açık değil (yalnızca YOLO
eğitimi var). Bu, planı değiştirmiyor: 4B adaptör T4'te ~2 saatte eğitiliyor ve
GGUF dönüşümü H200 gerektirmiyor (2 GB RAM yeter). H200 açılırsa orada yapılacak
şey 4B değil, **9B/14B'yi tam precision ya da daha uzun koşuyla** eğitmek olur.
GGUF tarafı yine aynı: adaptör + hazır taban GGUF.

## 7. Sık sorulacaklar

**Sadece `.gguf` istiyorum, adaptör istemiyorum — birleştirmem şart mı?**
Hayır. llama.cpp adaptörü yükleme anında uyguluyor; tek dosya isteyen bir
uygulama (ör. telefonda) için `llama-export-lora` kaldırıldığından birleştirme
gerekir — o zaman Kaggle değil, ≥32 GB RAM'li bir makine gerekir.

**Q4_K_M yerine başka quant?** Taban için `Q4_K_M` iyi denge; adaptör her zaman
f16 (küçük zaten). Taban quant'ı değişirse adaptör aynı kalır.

**Ollama?** Ollama 0.34.1'den beri `ADAPTER` satırını desteklemiyor; llama.cpp
kullanın (bkz. `docs/servis-ve-dinamik-lora.md`).

**HF Spaces'te çalışır mı?** CPU Basic'te llama.cpp + 4B Q4 ≈2,8 GB RAM ile
rahat sığar, hız 3,3–8,3 tok/s (bu repodaki `--serve-plan` tahmini).
