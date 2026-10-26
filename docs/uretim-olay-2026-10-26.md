# Olay raporu — 2026-10-26 üretim koşusu (5,7 saat, 0 kayıt)

Kaggle'da gece boyunca çalışan öğretmen üretimi **5,7 saat** sürdü ve **tek bir
eğitim kaydı üretmedi**. Koşu kendiliğinden durmadı; kullanıcı sabah elle
durdurdu. Bu belge kök nedenleri, ölçümleri ve yapılan düzeltmeleri kaydeder.

## 1. Ölçülen tablo (log'dan ayıklandı)

Log 781 satır; tekrar eden satırlar çıkarıldığında **505 gerçek olay**:

| Olay | Sayı | Anlamı |
| --- | --- | --- |
| `empty content (finish_reason='length')` | 118 | Yanıt hiç yazılmadan token bütçesi bitti |
| `Temporary failure in name resolution` | 74 | DNS çözümlemesi Kaggle tarafında düştü |
| `read operation timed out` (180 sn) | 20 | Uç nokta yanıt vermedi |
| `failed after N attempts` | 10 | İstek tükendi |
| slot seviyesinde `İstek başarısız` | 25 | ~25 slot denendi, **0 kabul** |
| `Durduruldu` / abort | **0** | **Güvenlik frenlerinin hiçbiri çalışmadı** |

Verim: 2.000 slotluk hedefe karşı **0 kayıt**; slot başına ~14 dakika.

## 2. Kök nedenler

### 2.1 `TeacherUnavailableError` retryable değildi (kritik)

`TeacherClient.chat` denemeler tükendiğinde `TeacherUnavailableError` atıyordu,
ama `RETRYABLE_ERRORS` bu tipi içermiyordu. Sonuç zinciri:

1. `TeacherPool.ask_with_fallback` hatayı **yakalamıyordu** → ikinci öğretmen hiç
   denenmedi.
2. `generate_pro`'nun yedek dalı (`except TeacherError`) yalnızca
   `stats.failed += 1` yapıyordu → `register_failure()` çağrılmadı →
   `abort_after_consecutive_failures: 8` eşiği **hiç dolmadı**.
3. Döngü 2.000 slotu sonuna kadar denemeye çalıştı.

### 2.2 Token bütçesi reasoning modellerine yetmiyordu

`max_tokens: 2048` ile glm-5.3 / mimo-v2.6-pro düşünme zincirinde bütçeyi
tüketip `finish_reason="length"` + boş `content` döndürdü (118 kez). Aynı
parametrelerle 3 kez × 2 öğretmen yeniden denendi: deterministik hata için
yüzlerce dakika.

### 2.3 Ağ: Kaggle tarafında DNS/bağlantı kararsızlığı

`[Errno -3] Temporary failure in name resolution`. Dışarıdan kontrol (2026-10-26):

* `evren-llmapi.ssyz.org.tr` → **195.142.26.68** (çözülüyor)
* TCP 443 → **açık, 0.00 sn**

Yani uç nokta ayakta; sorun Kaggle'ın resolver'ında veya uç noktanın eşzamanlı
istek davranışında. Koşunun en başında (`98. sn`) `Model listesi alınamadı`
uyarısı bunun habercisiydi ve görmezden gelindi.

## 3. Düzeltmeler

| # | Değişiklik | Dosya |
| --- | --- | --- |
| 1 | `TeacherUnavailableError` → `RETRYABLE_ERRORS`; öğretmen değiştirme ve ardışık-hata sayacı artık çalışır | `src/gokturk/teachers.py` |
| 2 | Yedek `except TeacherError` dalı `register_failure()` çağırır + abort eşiğini kontrol eder | `src/gokturk/data_generation.py` |
| 3 | **Sıfır-verim sigortası** (`zero_yield_abort_after`, varsayılan 3): N slotta tek kayıt yoksa durur | `src/gokturk/data_generation.py` |
| 4 | **Üretim öncesi sağlık denetimi** (tek küçük istek): uç nokta/DNS ölüyse hiç başlamaz | `src/gokturk/data_generation.py` |
| 5 | `finish_reason="length"` + boş içerik → `TeacherTruncatedResponseError`; **tek** ek deneme iki katı bütçeyle (`MAX_ESCALATED_TOKENS=8192`), kör tekrar yok | `src/gokturk/teachers.py` |
| 6 | `finish_reason` `length` değilse `reasoning_content`/`reasoning` alanı içerik olarak kabul edilir | `src/gokturk/teachers.py` |
| 7 | Config: `timeout_s 180→120`, `max_attempts_per_sample 3→2`, `max_tokens 2048→4096` | `configs/teachers.yaml` |
| 8 | Notebook **6a** tek çağrılık ölçüm (2 dk), **6b** DNS+TCP tanısı, `progress.log`, `abort_reason` çıktısı | `notebooks/kaggle_training.ipynb` |

## 4. Aynı hata tekrar ederse ne olacak

| Senaryo | Yeni davranış |
| --- | --- |
| DNS ölü | 6b üretimi hiç başlatmaz; ~10 sn içinde "AĞ HATASI" yazar |
| Uç nokta yanıt vermiyor | Sağlık denetimi düşer → tek slot bile denenmez, `stop_reason=health-check-failed` |
| Yanıtlar hep boş | İlk 3 slotta kabul yoksa `sıfır verim` ile durur (~dakikalar, saatler değil) |
| Tek tek boş yanıt | Bütçe 4096→8192 yükseltilir, bir kez denenir |
| Bir öğretmen çöktü | Diğeri denenir; 8 ardışık hatada koşu durur |
| Uzun sürdü | `TIME_BUDGET_MIN` dolunca durur, `stop_reason=time-budget`, resume ile sürer |

## 4b. Sınırlar — ne düzelmedi

Bu düzeltmeler hatayı **görünür ve ucuz** hale getirir; hatanın kendisini ortadan
kaldırmaz. Açıkça:

* **DNS sorunu çözülmedi.** Kaggle tarafındaki `Temporary failure in name resolution`
  hâlâ olabilir. Değişen şey, artık üretim başlamadan (≈10 sn) görülmesi ve
  saatlerce boşa dönülmemesi. Kaynağı bizde olmayan bir ağ sorununu "düzeltmek"
  yerine "erken tespit" ediyoruz — kabul edilebilir bir trade-off.
* **`timeout_s: 120` hâlâ tahmin.** 180→120 indirimi ölçüme değil, "120 sn meşru
  bir yanıt için fazlasıyla yeterli" gerekçesine dayanıyor. **6a hücresi bunu da
  doğrular**: gerçek gecikmeleri saniye cinsinden yazar.
* **460 test geçiyor ama gerçek uç nokta doğrulaması yapılmadı.** Testler sahte
  transport'la çalışır; gerçek EVREN davranışı (token bütçesi, gecikme, DNS)
  yalnızca 6a ile doğrulanır. Bu ortamda anahtar yok.

Sıra bu yüzden sabit: **merge → notebook güncelle → 6a → 6b → 6c**. 6a'da sorun
çıkarsa config (bu repodan) yeniden güncellenir.

## 5. Dürüst sonuç

* 2026-10-26 koşusundan **hiç veri çıkmadı**; hiçbir model eğitilmedi.
* Yukarıdaki düzeltmeler testlerle kilitlendi (bkz. `tests/test_teachers.py`,
  `tests/test_data_generation_pro.py`, `tests/test_notebook.py`).
* Düzeltmeler gerçek uç noktaya karşı **doğrulanmadı** (bu ortamda anahtar yok).
  İlk gerçek koşuda önce **6a** hücresi çalıştırılmalı: hangi token bütçesinde
  içerik geldiğini gösterir.
