# Dinamik LoRA ile servis — ve önceki bir yanlışımın düzeltmesi

Bu belge, "GGUF için LoRA'yı birleştirmek zorundayız" iddiamın **nerede
yanlış** olduğunu ve bundan sonra ne yapacağımızı anlatır.

> **Özet:** Birleştirme (merge) yalnızca *tek dosya* dağıtmak istediğinizde
> gerekir. Adaptörü kullanmak için gereken üç yol var ve hiçbiri birleştirme
> yapmaz. En önemlisi: `convert_lora_to_gguf.py` yalnızca adaptörü çevirir,
> taban ağırlıkları istemez — yani **GGUF artık Kaggle'da da üretilebilir.**

---

## 1. Düzeltme: neyi yanlış söylemiştim

`docs/yayinlama-ve-gguf.md`'de "GGUF = önce birleştir, bu yüzden 42 GB RAM
gerekir, Kaggle'da imkânsız" yazdım. Birleştirilmiş GGUF için bu doğru; ama
**adaptörlü GGUF** için yanlış. llama.cpp'nin kendi yolu adaptörü ayrı bir GGUF
olarak çevirip taban modelin üstüne `--lora` ile bindiriyor. O çeviri
`adapter_config.json` + `adapter_model.safetensors` okur, tabanın yalnızca
**config**'ine ihtiyaç duyar — ağırlıklarına değil.

```
YANLIŞ: adaptör → (42 GB birleştir) → convert_hf_to_gguf → quantize → gguf
DOĞRU : adaptör → convert_lora_to_gguf → küçük lora.gguf
        taban  → (hazır GGUF indir)   → llama-server -m base --lora lora.gguf
```

## 2. Dört servis yolu

| Yol | Birleştirme | Ne yapar | Ne zaman |
| --- | --- | --- | --- |
| `peft` | ❌ | `PeftModel.from_pretrained(base, adapter)` | Python içinde, tek süreç |
| `vllm-lora` | ❌ | `--enable-lora --lora-modules ad=yol` | GPU sunucusu, çok adaptör |
| `llamacpp` | ❌ | `convert_lora_to_gguf.py` + `--lora` | CPU'da da çalışır, Kaggle'da üretilebilir |
| `merged` | ✅ | birleştir → tek `.gguf` | yalnızca tek dosya zorunluysa |

Hepsini tek komutla karşılaştırın:

```bash
python -m gokturk.serve --plan --base-model Qwen/Qwen3.5-9B --vram-gb 48
```

Çıktı her yolu `[✓]/[✗]`, gereken bellek/disk, komut ve **gerçek tuzaklar**la
listeler; `merged` bu makinede RAM *ve* disk duvarına çarparsa `✗` olur.

## 3. vLLM: dinamik LoRA gerçek, ama bir tuzağı var

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # çalışırken değiştirmek için
vllm serve Qwen/Qwen3.5-9B --enable-lora --max-lora-rank 32 \
     --lora-modules gokturk=kullanici/gokturk-ultra-lora
```

Çalışırken adaptör ekleme/çıkarma: `POST /v1/load_lora_adapter` ve
`/v1/unload_lora_adapter`. Bu uç noktalar **yolu çağıran belirler**, bu yüzden
yalnızca güvenilen ortamda açılmalı.

**Tuzak:** vLLM'de bitsandbytes 4-bit (bnb) tabanla **dinamik** LoRA
desteklenmiyor. bnb QLoRA'da yalnızca `--qlora-adapter-name-or-path` ile **tek**
adaptör, **başlangıçta** yüklenebilir; `--lora-modules` verirseniz adaptör
sessizce yok sayılır ve sunucu düz taban modeli cevaplar. Yani:

- Eğitim QLoRA olabilir (bizimki öyle, r=16) — **sorun değil**;
- ama **servis** ederken taban bnb-4bit olmamalı. AWQ/GPTQ/FP8 taban seçin.

Ayrıca adaptör ranki `--max-lora-rank`'ı aşamaz (bizim r=16, 32 veriyoruz).

## 4. llama.cpp: Kaggle'da GGUF üretmenin yolu

```bash
# 1) Adaptörü çevir (taban ağırlığı gerekmez, saniyeler sürer)
python llama.cpp/convert_lora_to_gguf.py outputs/run/adapter \
    --base Qwen/Qwen3.5-9B --outfile gokturk-lora.gguf --outtype f16

# 2) Taban GGUF + adaptör ile servis
llama-server -m Qwen3.5-9B-Q4_K_M.gguf --lora gokturk-lora.gguf -c 4096 --jinja
```

- Birden fazla adaptör: `--lora a.gguf --lora b.gguf`;
- hepsini yükle ama uygulama: `--lora-init-without-apply`;
- istek başına seçim: gövdede `lora: [{"id": 0, "scale": 1.0}]`;
- `POST /lora-adapters` ile ölçek değiştirme / sıcak değiştirme.

**Tuzaklar:**

- Çevirici yalnızca `lora_A`/`lora_B` tensörlerini anlar: **DoRA** desteklenmez;
  adaptörde `embed_tokens`/`lm_head` varsa (yeni token eklenmişse) hata verir.
  Bizim eğitimde `target_modules` yalnızca projeksiyonlar ve `bias: none` —
  bu yüzden uygun.
- Unsloth adaptörün `adapter_config.json`'ında **quantize edilmiş eğitim kopyası**
  yazılı olabilir (bnb-4bit deposu). O durumda `--base-model-id` ile orijinal
  kimliği verin; yoksa çevirici yanlış config okur.
- LoRA'yı **quantize tabana** uygulamak çalışır ama f16 tabana göre küçük bir
  kalite farkı olur. En temizi: f16 taban + `llama-export-lora` ile tek dosya.
- Ollama 0.34.1'den (Eylül 2026) beri `ADAPTER` satırı **desteklenmiyor**;
  adaptörlü servis için llama.cpp kullanın.

## 5. Hugging Face Space iskeleti

```bash
python -m gokturk.serve --emit-space space/ \
    --base-model Qwen/Qwen3.5-9B --adapter-repo kullanici/gokturk-ultra-lora
```

Üç dosya üretir: `app.py` (Gradio + `PeftModel`, birleştirme yok, `@spaces.GPU`
desteği), `requirements.txt`, `README.md` (Space YAML başlığı + dürüst uyarılar).
Mevcut dosyaların üzerine **yazmaz** (`--overwrite` gerekir).

### Ama önce Spaces gerçeği (değişmedi)

- Ücretsiz kişisel hesapta **yeni Gradio/Docker Space oluşturmak ücretli plan**
  istiyor; ücretsiz istisna: **2 adet ZeroGPU Gradio Space**.
- ZeroGPU kotası **günde 5 dakika** (giriş yapmamış 2 dk, PRO 40 dk).
- CPU Basic (2 vCPU / 16 GB) ücretsiz ama yeni oluşturmak plan istiyor ve
  48 saat kullanılmazsa uyuyor; disk **kalıcı değil**.

Yani dinamik LoRA, "birleştirme/disk" problemini çözer; **"ücretsiz CPU'da
7/24 hızlı asistan"** problemini çözmez. Hız için:

```bash
python -m gokturk.publish --serve-plan --base-model Qwen/Qwen3.5-9B --ram-gb 16 --vcpu 2
# KARAR: SINIRDA — çalışır ama yanıtlar yavaş  (1.5–3.6 token/sn)
```

## 6. Karar

| Hedef | Yol |
| --- | --- |
| HF'de demo (adaptör ayrı, birleştirme yok) | `peft` + `--emit-space` |
| GPU sunucusunda üretim | `vllm-lora` (bnb taban **değil**) |
| İndirilebilir GGUF, Kaggle'da üretim | `llamacpp` (adaptör-only dönüşüm) |
| Tek dosya, "her yerde çalışsın" | `merged` (yalnızca burada 42 GB gerekir) |

Birleştirme artık **varsayılan değil, istisna**.
