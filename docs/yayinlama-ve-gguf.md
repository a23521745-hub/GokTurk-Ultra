# Yayınlama: Hugging Face'e push ve GGUF

Bu belge iki soruyu net cevaplar:

1. Eğitim bitince model **Hugging Face'e** gidiyor mu?
2. Model **`.gguf` / Q4_K_M** olarak çıkacak mı?

Kısa cevap: **(1) evet, artık var** — `gokturk.publish` ile adaptörü özel bir
depoya yükleyebilirsiniz. **(2) hayır, otomatik değil** ve Kaggle ücretsiz
katmanında **fiziksel olarak mümkün değil**. Nedenini ve nerede yapılabileceğini
aşağıda sayılarla yazdım.

---

## 1. Hugging Face'e push

Eğitim çıktısı şu yapıdadır:

```
outputs/gokturk-run/
├── adapter/               # LoRA ağırlıkları (adapter_config.json + safetensors)
└── run_manifest.json      # ne çalıştı, hangi ayarlarla, hangi kayıpla
```

Yükleme **adaptörü** yükler, 20B'lik tam modeli değil. Bu bilinçli: adaptör
~100–400 MB, tam model ~42 GB. Kullanıcı `from_pretrained` + `PeftModel` ile
temel modelin üzerine takar.

### Önce plan (ağa çıkmaz)

```bash
python -m gokturk.publish \
  --adapter outputs/gokturk-run/adapter \
  --repo-id kullanici/gokturk-ultra-lora
```

Çıktı bir JSON: hangi dosyalar, kaç MB, token var mı, eksik ne var. Eksik varsa
çıkış kodu **2**.

### Sonra gerçek yükleme

```bash
export HF_TOKEN=hf_...        # yazma izinli token; KAGGE'da Secret olarak
python -m gokturk.publish \
  --adapter outputs/gokturk-run/adapter \
  --repo-id kullanici/gokturk-ultra-lora \
  --manifest outputs/gokturk-run/run_manifest.json \
  --push
```

* `--public` vermezseniz depo **özel** açılır.
* Token **yalnızca** `HF_TOKEN` ortam değişkeninden okunur; dosyaya yazılmaz,
  model kartına girmez, loglanmaz. Test bunu ayrıca denetler.
* Depo oluşturulurken `repo_type="model"` kullanılır; `dataset` deposu
  `from_pretrained` ile çalışmaz.

### Model kartı

Yükleme sırasında `README.md` **otomatik üretilir** ve manifestten beslenir:
temel model, adaptör türü, eğitim örneği sayısı, adım, train/eval kaybı, arka uç.
Kartta ayrıca şu uyarılar yazılıdır:

* ölçülmüş benchmark sonucu **yoktur**,
* değerlendirme + insan incelemesi yapılmadan üretim için önerilmez,
* veri sentetiktir,
* temel modelin lisansı geçerlidir.

Kartı yüklemeden görmek için:

```bash
python -m gokturk.publish --card-only --manifest outputs/gokturk-run/run_manifest.json
```

---

## 2. GGUF: neden otomatik değil

LoRA'yı GGUF'a çevirmenin ilk adımı **adaptörü temel modele birleştirmektir**
(`merge_and_unload`). Bu, 20.9B parametreli modelin **tamamını** belleğe alır:

| Adım | Boyut | Not |
| --- | --- | --- |
| Birleşik checkpoint (bf16) | **≈42 GB** | 20.9B × 2 bayt |
| bf16 ara GGUF | **≈42 GB** | `convert_hf_to_gguf.py --outtype bf16` |
| Nihai MXFP4 / Q4_K_M | ≈11–12 GB | uzmanlar 4-bit kaldığı için küçük |
| **Tepe disk ihtiyacı** | **≈100 GB** | ara dosya silinince düşer |

Birleştirme sırasında bellek tepesi ≈67 GB (42 GB'ın ~1.6 katı).

Kaggle ücretsiz katmanı: **29 GB RAM, 20 GB `/kaggle/working`**. Yani:

> **Kaggle'da GGUF üretilemez.** Bu bir ayar sorunu değil, RAM ve disk
> duvarıdır. `gokturk.publish --gguf-plan` bunu makinenin gerçek sayılarıyla
> söyler ve yine de komutları yazar.

### Plan komutu

```bash
python -m gokturk.publish --gguf-plan --adapter outputs/gokturk-run/adapter
```

Makinenin RAM'ini ve boş diskini ölçer, üç boyutu tahmin eder ve karar verir:

```
KARAR: BU MAKİNEDE YAPILAMAZ
  ✗ RAM yetersiz: birleştirme ≈66.9 GB ister, makinede ≈29.0 GB var.
  ✗ Disk yetersiz: ≈99.8 GB gerekir, boş ≈20.0 GB.
```

Yeterli makinede (`--quant mxfp4|Q4_K_M|Q5_K_M|Q8_0`) 5 adımlık komut listesi
basılır: birleştir → llama.cpp derle → bf16 GGUF → quantise → ara dosyayı sil.

Nerede çalışır: ≥64 GB RAM ve ≥100 GB boş diski olan herhangi bir Linux
makinesi. EVREN'in HPC kümesi (düğüm başına 141 GB HBM) kaynak olarak yeterli,
ancak kendi konteyner imajınızda `cmake`/`llama.cpp` kurulu olmalıdır — bu
repoda EVREN HPC üzerinde GGUF derlemesi **denenmemiştir**.

---

## 3. Q4_K_M mi, MXFP4 mü?

gpt-oss **zaten 4-bit** doğar: MoE uzman katmanları (parametrelerin >%90'ı)
OpenAI tarafından MXFP4 olarak eğitilmiş. Bu yüzden:

| | MXFP4 | Q4_K_M |
| --- | --- | --- |
| Bit/parametre | 4.25 | 4.84 |
| Dosya boyutu (20B) | ≈12.1 GB | ≈11.7 GB |
| Ek yuvarlama | yok (doğal biçim) | var (bf16 → 4-bit) |
| llama.cpp hedefi | `MXFP4_MOE` | `Q4_K_M` |
| gpt-oss için öneri | **evet** | desteklenir, önerilmez |

Topluluk Q4_K_M dosyaları da FFN uzmanlarını MXFP4'te **tutar**; yani
"Q4_K_M" etiketi çoğunlukla yalnızca uzman olmayan katmanların quant'ını
anlatır. Düz `llama-quantize Q4_K_M` bunu yapmaz ve uzmanları da 4-bit K-quant'a
sokar — bu model ailesinde kalite kaybı buradan gelir.

**Sonuç:** Q4_K_M isterseniz `--quant Q4_K_M` var ve çalışır. Ama bu model için
teknik olarak doğru hedef **MXFP4**'tür; bu yüzden varsayılan odur.

---

## 4. Özet akış

```
eğitim (Kaggle, 60 adım ölçüm) 
   └─> adapter/ + run_manifest.json
         ├─> HF push  : python -m gokturk.publish --push      (Kaggle'da yapılabilir)
         └─> GGUF     : python -m gokturk.publish --gguf-plan (≥64 GB RAM'li makinede)
```

Kaggle'da yapılabilen: eğitim, adaptör kaydı, adaptörü HF'ye yükleme.
Kaggle'da yapılamayan: birleştirme ve GGUF dönüşümü.
