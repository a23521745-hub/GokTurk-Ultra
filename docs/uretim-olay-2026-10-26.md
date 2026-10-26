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
| 9 | `TeacherDnsError`: `gaierror` artık timeout değil, ayrı tip (retryable kalır) | `src/gokturk/teachers.py` |
| 10 | `list_models` retry'lı (`chat` ile aynı politika); 401/403 tekrar denenmez | `src/gokturk/teachers.py` |
| 11 | `diagnose_endpoint()` (DNS×3 + TCP + anahtar, enjekte edilebilir I/O) ve `describe_error()` (tür bazlı Türkçe teşhis) | `src/gokturk/teachers.py` |
| 12 | 3b hücresi: önce tanı, sonra liste; koşulsuz "anahtar yok" cümlesi kaldırıldı; hücrenin opsiyonel olduğu yazılı | `notebooks/kaggle_training.ipynb` |
| 13 | 6b hücresi elle yazılmış socket tanısı yerine `diagnose_endpoint()` kullanır (tek kaynak) | `notebooks/kaggle_training.ipynb` |
| 14 | Sağlık denetimi ping bütçesi `8 → 64` token (`HEALTH_CHECK_MAX_TOKENS`); `finish_reason='length'` artık **sağlıklı** sayılır (uç nokta yanıt verdi) | `src/gokturk/teachers.py` |
| 15 | Bütçe yükseltmesi **en fazla bir kez** (`escalated` bayrağı); 8→16→32→64→… tırmanışı bitti | `src/gokturk/teachers.py` |
| 16 | `chat(..., escalate_truncation=False)` — ölçüm hücresi hangi bütçenin yettiğini görebilsin | `src/gokturk/teachers.py` |
| 17 | 6a hücresi `NET_OK`'a bağımlı değil: kendi `diagnose_endpoint()` tanısını yapar; bütçe ölçümü `128/512/2048/4096/8192` | `notebooks/kaggle_training.ipynb` |

### 2.4 3b hücresi yanlış yeri suçladı

Aynı gece 3b hücresi şunu yazdı:

```
Model listesi alınamadı: TeacherTimeoutError connection error: [Errno -3] Temporary failure in name resolution
En sık sebep: EVREN_API_KEY tanımlı değil (Add-ons > Secrets > Attach).
```

Anahtar **tanımlıydı** (log: `EVREN_API_KEY TANIMLI`). Üç kusur:

1. **Tek `except TeacherError` dalı:** `TeacherConfigError` (gerçekten anahtar yok),
   `TeacherAuthError` (401) ve `TeacherTimeoutError` (DNS) aynı yere düşüyordu; altına
   koşulsuz "anahtar yok" cümlesi basılıyordu.
2. **DNS, timeout olarak sınıflanıyordu:** `urllib.error.URLError(DNS)` →
   `TeacherTimeoutError("connection error: ...")`. İkisi farklı müdahale gerektirir.
3. **`list_models` tek deneme yapıyordu:** tek bir DNS sıçraması hücreyi düşürüyordu.

Düzeltmeler (bkz. §3 tablo, satır 8–11): `TeacherDnsError` ayrı tip, `list_models`
retry, `diagnose_endpoint()`, `describe_error()` ve hücrede tür bazlı teşhis.

### 2.5 Sağlık denetimi kendi kendini boğdu (ikinci koşu)

İkinci koşunun logunda üretim başlamadan şu tekrarlandı:

```
[gokturk.teachers|WARNING] model mimo-v2.6-pro: boş içerik (finish_reason=length),
                           max_tokens 8 -> 16 ile bir kez daha denenecek
                           16 -> 32, 32 -> 64, 64 -> 128 ...
```

Üç ayrı kusur üst üste bindi:

1. **Ping bütçesi 8 token'dı.** Reasoning modeli 8 token'da yalnızca düşünmeye
   başlar; `content` hiç yazılmaz. Sağlık denetimi bu yüzden kurulum hatası değil,
   *ölçüm hatası* üretiyordu.
2. **Yükseltme her kesilmede bir adım daha ekliyordu.** "Tek kademeli yükseltme"
   tasarımı `max_tokens=8` gibi küçük bir tabanla başlayınca 11 denemeye çıktı
   (8→16→32→64→128→256→512→1024→2048→4096→8192). Öğretmen başına ~2 dakika.
3. **Sağlık denetiminin ölçütü yanlıştı.** Soru "uç nokta ayakta ve yetkili mi?";
   cevabı "tam bir yanıt yazabiliyor mu?" değil. HTTP 200 + geçerli `choices` +
   doğru model kimliği yeterli. `finish_reason='length'` bir sağlık arızası değil.

Düzeltmeler (bkz. §3 tablo, satır 14–17): ping bütçesi 64, kesilme = sağlıklı,
yükseltme en fazla bir kez, 6a kendi tanısını yapar ve yükseltmeyi kapatır.

Ayrıca **6a hücresi hiç çalışmamıştı**: `globals().get("NET_OK", False)` kontrolü
`NET_OK`'ı üreten hücreden (6b) *önce* koşuyordu, yani değişken her zaman yoktu.
İki koşudur "ağ tanısı başarısız" yazan satır buydu; ağda sorun yoktu.

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
  Retry'lar tek sıçramayı atlatır; **aralıklı** bozulma sürerse oturumu yeniden
  başlatmak dışında yapabileceğimiz bir şey yok (uç nokta dışarıdan sağlam).
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

## 2.6 İki kayıt oluştu; içerik denetimi yeni kusurları gösterdi

Kullanıcının paylaştığı iki satır incelendi: özetleme isteğinde abstract yoktu;
öğretmenin metni istemesi doğru olsa da bu kayıt özetleme örneği değildi.
Şablondaki “metni veya metnin türünü” ifadesi eksik girdiye izin veriyordu.
Artık özetleme/yeniden yazma/çeviri taslaklarında `<kaynak>...</kaynak>` içinde
kaynak zorunlu (en az 40 karakter); eksikse çözücü çağrılmadan ret dosyasına yazılır.
Bu yapısal kontrol kaynak doğruluğunu veya yeterliliğini kanıtlamaz.

İkinci kayıtta eleştirinin bulduğu çok-adımlı çözüm önerisi, düzeltmede de kalmıştı.
Sert biçim kontrolleri semantik sorunu yakalamadı. Düzeltme artık ayrı bir öğretmen
çağrısında yeniden incelenir; sorun sürüyorsa, eleştiri okunamıyorsa veya doğrulama
isteği başarısızsa kayıt kabul edilmez. Bu bir model denetimidir, insan onayı değildir.
Düzeltmeyi üreten öğretmen/model ve çağrı metaverisi ayrıca tutulur; aday çağrı
metaverisi tüm pipeline maliyetini temsil etmez.

Notebook kararsız DNS'i kabul etmez; varsayılan üretim 3 slot/15 dakika pilotudur.
Süre sınırı slotlar arasında uygulanır, kesin duvar-saati sınırı değildir.
Kabul edilen her kayıt INFO günlüğüne yazılır. Pilot sonrasında insan incelemesi
olmadan uzun üretime geçilmemelidir. Önceki 3 çağrı/slot tahmini eksikti:
taslak + iki aday + eleştiri en az 4 çağrı, düzeltme doğrulaması ek çağrıdır.
DNS ve sağlayıcı zaman aşımı çözülmüş değildir; canlı Kaggle doğrulaması yapılmadı.
