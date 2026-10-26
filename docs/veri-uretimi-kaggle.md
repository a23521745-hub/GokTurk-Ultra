# 2.000 örnek üretim koşusu (Kaggle, süre bütçeli)

Amaç: GLM-5.3 + MiMo-V2.6-Pro'yu **öğretmen** olarak kullanıp Qwen3.5-4B öğrencisi
için gerçek bir SFT veri seti üretmek. Bu belge o koşunun nasıl başlatıldığını,
nasıl yarıda kalıp devam ettiğini ve ne kadar sürdüğünü anlatır.

## 0. Sayılar nereden geliyor

`--plan-only` ile ölçüldü (ağa çıkmaz):

| Hedef (slot) | Çağrı (2 aday + 1 öz-eleştiri) | Token | 15 istek/dk'da süre |
| --- | --- | --- | --- |
| 1.000 | 3.000 | ≈4,8 M | ≈3,3 saat |
| **2.000** | **6.000** | **≈9,6 M** | **≈6,7 saat** |
| 5.000 | 15.000 | ≈24 M | ≈16,7 saat |
| 10.000 | 30.000 | ≈48 M | ≈33 saat |

Darboğaz para değil **oran sınırı**: `configs/teachers.yaml` içindeki
`requests_per_minute: 15`. Sağlayıcı daha fazlasına izin veriyorsa süre kısalır;
izin vermiyorsa 429 tekrarı devreye girer ve koşu yavaşlar.

## 1. Ön koşullar

* Kaggle notebook → **Add-ons → Secrets**: `EVREN_API_KEY` eklenmiş ve **Attach**
  edilmiş olmalı. Anahtar hiçbir zaman hücreye yazılmaz; 3. hücre onu ortam
  değişkenine aktarır.
* **Internet → On** (öğretmenler dış API).
* Üretim için **Accelerator → None** yeterlidir: çağrılar GPU kullanmaz, GPU
  kotanız yanmaz.

## 2. Koşu (tek tuş)

Notebook'ta 6. adımdan sonra gelen **6b** hücresi:

```python
RUN_GENERATION = True
GENERATION_TARGET = 2000
TIME_BUDGET_MIN = 540        # 9 saat
```

* `Save & Run All (Commit)` ile başlatın. Etkileşimli oturumda 20 dakika
  dokunulmazsa oturum kapanabilir; commit modu bunu önler.
* `TIME_BUDGET_MIN` dolduğunda üretim **slot bitiminde** durur:
  `stop_reason: "time-budget"`, **çıkış kodu 0** — bu bir hata değil.
* Kabul edilen her kayıt anında diske yazılır (`/kaggle/working/data/raw/pro.jsonl`);
  ret ve öz-eleştiri kayıtları yanındaki `.rejects.jsonl` / `.critique.jsonl`
  dosyalarına gider.

Aynı şey komut satırında:

```bash
python -m gokturk.data_generation --mode pro --target 2000 \
  --out data/raw/pro.jsonl --time-budget-min 540
```

### Klonlanan kod süre bütçesini desteklemiyorsa

Notebook 3. hücresi depoyu `main`'den çeker. `generate_pro(..., time_budget_s=...)`
henüz `main`'de yoksa 6b hücresi **parçalı koşuya** düşer: 30 slotluk parçalar
halinde üretir, her parça arasında saati kontrol eder ve bütçe dolunca durur.
Resume sayesinde parçalar birbirinin üstüne yazmaz. Yani süre bütçesi olmayan bir
`main` ile de gözetimsiz koşu güvenlidir — sadece ilerleme çıktısı daha sık olur.

Tek çağrılı (daha temiz) yol için `main`'de bu özellik bulunmalı; ya PR birleşir
ya da 3. hücrede `REPO_BRANCH` ilgili dala çevrilir.

## 3. Yarıda kalırsa / ikinci oturum

1. İlk oturumun çıktısını **Kaggle Dataset** olarak kaydedin
   (`/kaggle/working/data/raw/` → *New Dataset*).
2. Yeni oturumda o veri setini **Add Input** ile ekleyin.
3. 6b hücresi `/kaggle/input` altında `pro.jsonl` bulursa `/kaggle/working`'e
   kopyalar ve **kaldığı yerden sürer** (`resume`, varsayılan açık): mevcut `id`'ler atlanır
   (`skipped_existing`), plan sırası deterministik olduğu için aynı slot iki kez
   üretilmez.

## 4. Kalite kapısı (6c)

Üretim bittiğinde (ya da yarıda bıraktığınızda) **6c** hücresi:

* şema doğrulaması (`id`, `prompt`, `response`, `source`),
* boş/çok kısa/çok uzun/reddetme/tekrarlı yanıt filtresi,
* yakın-tekrar temizliği ve sızıntısız train/valid/test bölmesi

yapar ve `data/processed/data_report.json` yazar. 200'den az eğitim satırı
kalırsa uyarır — o durumda üretimi sürdürün.

## 5. Eğitim (8. hücre)

`data/processed/train.jsonl` varsa 8. hücre **gerçek veriyle** koşar ve adım
sayısını veriden hesaplar:

```
adım = örnek × epoch / (batch × grad_accum)      # 60–1500 arasına sıkıştırılır
```

Sabit 60 adım kaldırıldı: 4 örnekle 60 adım 60 epoch demekti ve `train_loss`
düşerken model sadece ezberliyordu. 2.000 slotun ~1.600 kabulüyle tipik hesap
`1600 × 3 / 16 ≈ 300` adım, yani ≈3 epoch.

## 6. Dürüst beklenti

* Bu koşu **damıtma**dır (öğretmen çıktısını taklit), sihir değil. 4B öğrenci
  10k örnekte üslup/format/akıl yürütme kalıbını belirgin biçimde öğrenir; yeni
  bilgi edinmez.
* "Daha tutarlı yanıt" iddiası **ölçülmeden** yazılmaz. `data/eval/` seti küçük
  ve kural tabanlı; yayın öncesi en az 100–200 elle etiketli karşılaştırma
  (taban vs adaptör) gerekir.
* Öğretmen yanıtları insan denetiminden geçmemiştir; "altın veri" sayılmaz
  (`docs/architecture.md` §4).
