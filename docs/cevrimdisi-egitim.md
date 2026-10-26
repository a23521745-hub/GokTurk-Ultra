# DNS/ağ duvarı eğitimi yakar mı?

Kısa cevap: **hayır — eğitim ağ olmadan da çalışacak şekilde kurulabilir.** Ama
işin iki aşaması var ve yalnızca biri ağa gerçekten muhtaç.

## 1. İki aşama, iki farklı bağımlılık

| Aşama | Ağa ihtiyaç | DNS düşerse ne olur |
| --- | --- | --- |
| **Veri üretimi** (6b, `data_generation --mode pro`) | **Evet, zorunlu** — öğretmen API'si | Üretim durur. Kayıtlar diskte kalır, `resume` ile aynı komut kaldığı yerden sürer. 2026-10-26'daki 5,7 saatlik kayıp tam olarak buydu. |
| **Eğitim** (8. adım, `train.py`) | **Yalnızca taban ağırlıkların ilk indirmesi** | Ağırlıklar yereldeyse **hiç etkilenmez**. Yerelde değilse eğitim başlamaz. |

Yani "her internet gittiğinde eğitim yanar" doğru değil; **eğitim, ağırlıklar bir
kez yerelleştirildiğinde tamamen çevrimdışı** çalışır.

## 2. Eğitimi çevrimdışı yapmak (3 adım, bir kez)

```bash
# 1) İnternet olan bir yerde ağırlıkları indir (Kaggle'da internet açıkken de olur)
huggingface-cli download Qwen/Qwen3.5-4B --local-dir qwen35-4b
```

2) `qwen35-4b/` klasörünü **Kaggle Dataset** olarak yükle (veya `/kaggle/input`
altına bağlanacak şekilde ekle).

3) Notebook'ta **7b** hücresi bunu bulur ve `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1` kurar. Ardından **Internet → Off** ile eğitim koşabilirsin:

```
Çevrimdışı eğitim denetimi
  TABAN AĞIRLIK : YEREL -> /kaggle/input/qwen35-4b/Qwen3.5-4B
  AĞ İHTİYACI   : YOK (eğitim internet olmadan koşabilir)
  VERİ train    : /kaggle/working/data/processed/train.jsonl
  SONUÇ         : Bu koşu tamamen çevrimdışı çalışabilir.
```

Komut satırında aynısı:

```bash
python -m gokturk.train --offline \
  --config configs/training_qwen3_5_4b_cot_t4.yaml \
  --model /kaggle/input/qwen35-4b/Qwen3.5-4B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/valid.jsonl
```

`--offline` verildiğinde araç **sessizce indirmeye çalışmaz**: yerel kopya veya veri
eksikse açık hata verip çıkış kodu `4` ile durur. Amaç, 9,3 GB'lık indirmeyi DNS
düşmüş bir makinede deneyip 20 dakika beklemek değil.

Arama köklerini değiştirmek için: `GTU_MODEL_SEARCH_ROOTS=/a:/b`.

## 3. Üretim tarafı: duvar tamamen aşılır mı?

Hayır, üretim API'ye muhtaç ve DNS bizim tarafımızda değil. Ama hasar sınırlandı:

| Koruma | Etki |
| --- | --- |
| Sağlık denetimi (üretimden önce, 3 DNS denemesi) | Uç nokta ölüyse **10 saniyede** durur |
| `zero_yield_abort_after=3` | 3 slotta tek kayıt yoksa saatler değil **dakikalar** içinde durur |
| `resume` + `pro.jsonl` | Kesinti veri kaybettirmez, aynı komut devam eder |
| `list_models` retry | Tek DNS sıçraması 3b'yi düşürmez |
| `progress.log` | Oturum kapansa da kanıt kalır |

Yani üretim **kayıpsız ve ucuz** kesilir; ama kesilir. Bu bir trade-off, çözüm değil.

## 4. Üretimi hiç kesintisiz yapmak istersen

Seçenekler, dürüst artı/eksileriyle:

1. **Kaggle'da "Internet → On" + gece boyu tek oturum** — mevcut yol. Kesinti olursa
   resume. En basit, en az kurulum.
2. **Kendi makinen / kiralık VPS**: DNS senin kontrolünde, `--time-budget-min` ile
   parçalı koş. Anahtar yine ortam değişkeninde kalır. En kararlı seçenek.
3. **EVREN üzerinde çalıştırmak**: LLM eğitimi orada yok (yalnızca YOLO), ama üretim
   için orada bir ortam varsa ağ en yakın noktada olur. Doğrulanması gerekir.

Hangisi seçilirse seçilsin **eğitim tarafı ağdan bağımsız** kalır; bu yüzden asıl
kazanç, üretimi bir kez tamamlayıp veriyi Kaggle Dataset olarak sabitlemektir.
Sonrasında eğitim, DNS'ten tamamen kopuk tekrar tekrar koşulabilir.

## 5. Ölçülen sınırlar

* `--offline` yolu testlerle doğrulandı (`tests/test_offline.py`), ama **gerçek
  Kaggle oturumunda henüz denenmedi**.
* Yerel ağırlıkların `transformers` sürümüyle uyumu (config/tokenizer) indirme
  anındaki sürümle aynı olmalı; sürüm yükseltirsen yeniden indir.
* Çevrimdışı eğitim **yeni bilgi getirmez**: veri seti neyse o. Üretim aşaması
  atlanırsa eğitilecek veri de yoktur.
