# CoT veri seti hazırlama (`data/datasets/evren_cot_10k.jsonl`)

Bu belge depoda izlenen CoT dosyasını **eğitilebilir** hâle getiren adımı ve
ölçülmüş gerçekleri anlatır. Kısa özet: dosya 10.000 satır ama **~920 tekil
talimat**; satır sayısı çeşitlilik değildir.

## 1. Dosya nedir, ne değildir

`data/datasets/evren_cot_10k.jsonl` (11,5 MB) satır başına şu alanları taşır:

```json
{
  "id": "evren_cot_20251025_000123",
  "domain": "cybersecurity",
  "instruction": "... (id: 123)",
  "thinking_process": "...",
  "response": "...",
  "quality_metrics": {"relevance": 1.0, "coherence": 1.0, "pii_clean": true}
}
```

* `train.py` bu şemayı **okumaz**. Bu repoda eğitim kaydı `id`, `prompt`,
  `response`, `source` alanlarını ister (`src/gokturk/schema.py`,
  `REQUIRED_FIELDS`). Dosyayı doğrudan `--train-file` vermek kalite kapısında
  *"no valid records left after filtering"* hatası üretir.
* `quality_metrics` 10.000 satırın **tamamında aynıdır** (1.0 / 1.0 / true);
  kalite sinyali taşımaz, eğitimde kullanılmaz.
* `instruction` alanlarının sonunda `(id: N)` / `(senaryo id: N)` kalıntısı var.
  Temizlenmezse model yanıt sonunda sayaç taklidi üretmeyi öğrenir.

## 2. Ölçülen gerçekler (2026-10-25, dosyanın kendisi üzerinde)

| Ölçüm | Değer |
| --- | --- |
| Satır | 10.000 |
| Ayrıştırma hatası | 0 |
| `(id: N)` kalıntısı temizlenen satır | 10.000 |
| **Tekil talimat** | **920** (etkin çeşitlilik **%9,2**) |
| Tekil yanıt | 993 |
| Tekil düşünme zinciri | 822 |
| Tekil (düşünme, yanıt) çifti | 1.992 |
| Aynı talimata çelişkili yanıt | 3 talimat |
| Yanıt uzunluğu (medyan) | 158 karakter (11–509) |
| Düşünme uzunluğu (medyan) | 577 karakter (310–1.627) |
| Talimat uzunluğu (medyan) | 11 kelime |

**Yorum:** dosya şablon üretimi. 10.000 satırla eğitim yapmak, ~920 örneğin
üzerinden ~10 epoch geçmek demektir; `logcheck` bunu `epoch-explosion` olarak
işaretler. Bu yüzden dönüştürücü **varsayılan olarak tekilleştirir** ve
`cot_report.json` içine çeşitlilik uyarısını yazar.

## 3. Dönüştürme + bölme

```bash
python -m gokturk.cot_dataset \
  --input data/datasets/evren_cot_10k.jsonl \
  --out-dir data/processed
```

Gerçek çıktı (bu depoda çalıştırıldı):

```
okundu=1992, geçersiz=0, kalite_red=0, tekrar=666, kalan=1326,
bölünme=train:1271/valid:28/test:27
```

* Dönüştürücü `thinking_process` + `response` alanlarını **tek hedefe** birleştirir
  (`thinking\n\nanswer`). Yalnızca nihai yanıtı hedeflemek için `--no-thinking`.
* Tekilleştirme `id` kalıntısından arındırılmış talimat + yanıt çifti üzerinden
  yapılır; ilk kayıt tutulur (deterministik).
* Ardından `gokturk.data_quality.build_datasets` çalışır: şema, uzunluk, tekrar
  ve yakın-tekrar kuralları. Kalan örnekler `train/valid/test.jsonl` olarak yazılır.
* Rapor: `data/processed/cot_report.json` (`--json` ile stdout'a da basılır).
* Az örnek kalırsa araç sessizce devam etmez: `CotDatasetError` ve **çıkış kodu 4**.
  Eşiği `--min-kept` ile değiştirin (varsayılan 400).

Bölünme oranları `QualityConfig` varsayılanlarıdır (valid/test %2). 1.326 örnekte
28/27 doğrulama-test örneği **küçüktür**; bu sayılarla ölçülen kayıp gürültülüdür
ve tek başına kalite kanıtı sayılmaz.

## 4. Eğitime bağlama

```bash
python -m gokturk.cot_dataset           # -> data/processed/{train,valid,test}.jsonl
python -m gokturk.train \
  --config configs/training_qwen3_5_4b_cot_t4.yaml \
  --plan-only                            # sığıyor mu, disk yetiyor mu
```

`configs/training_qwen3_5_4b_cot_t4.yaml` taban modeli, LoRA hedeflerini ve
`max_seq_len: 1024` değerini CoT hedefinin uzunluğuna göre ayarlar. Adım sayısı
sabit değildir:

```
adım = örnek × epoch / (per_device_batch × grad_accum)     # 60–1500 arasına sıkıştırılır
```

`gokturk.cot_dataset.training_steps_for()` bu hesabı yapar; Kaggle notebook'u
(5b ve 8. hücreler) adım sayısını **veriden** hesaplar, sabit 60 yazmaz.

## 5. Dürüst beklenti

1.326 tekilleştirilmiş örnekle 4B model **genel Türkçe talimat takibi öğrenmez**.
Öğrendiği şey bu 920 talimat ailesinin üslubu ve düşünme zinciri biçimidir.
Gerçek bir adaptör için çeşitliliğin artırılması gerekir:

```bash
python -m gokturk.data_generation --mode pro --target 10000 --plan-only --plan-out data/raw/plan.json
```

EVREN üretimi (MiMo-V2.6-Pro + GLM-5.3) tamamlandığında aynı dönüştürücü ve
aynı config kullanılır; yalnızca `data/raw/*.jsonl` büyür. Ölçülmemiş hiçbir
başarı iddiası README'ye veya model kartına yazılmaz.
