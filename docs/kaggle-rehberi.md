# Kaggle Rehberi — sıfırdan, bu proje için

Bu belge "Kaggle nedir"den başlar ve **bizim notebook'un çıktısını nasıl
okuyacağınıza** kadar gelir. Teknik ön bilgi varsaymaz. Log okumakta
zorlanıyorsanız §5 ve §6 tam size göre; hiç okumak istemiyorsanız §7'deki
araç logu sizin yerinize okur.

---

## 1. Kaggle aslında ne?

Kaggle, **ücretsiz GPU kiraladığınız bir web sitesi**. Bilgisayarınızda GPU
olmasına gerek yok; tarayıcıdan bir "notebook" (Jupyter defteri) açıp
Google'ın/Amazon'un sunucularındaki ekran kartlarında kod çalıştırıyorsunuz.

Bizim için önemli olan sayılar:

| Kaynak | Değer | Ne demek |
| --- | --- | --- |
| GPU | **2 × Tesla T4, her biri 16 GB** | İki kart var ama **bizim betik tek kart kullanır**. 16 GB, 20B'lik bir model için *çok az* — bu yüzden Unsloth gerekiyor. |
| Oturum süresi | **12 saat** | 12 saat sonra makine kapanır. Kodun o ana kadarki çıktıları kaybolabilir. |
| Haftalık kota | **~30 GPU saati** | Haftada toplam ~30 saat GPU kullanabilirsiniz. 12 saatlik 2-3 koşu demek. |
| RAM | **~29 GB** | Sistem belleği. Veri dosyanız çok büyükse burada takılır. |
| Disk | **20 GB** (`/kaggle/working`) | Kalıcı olan tek dizin. Diğer her şey oturum bitince silinir. |
| Internet | **Kapalı (varsayılan)** | Açmayı unutursanız `pip install` ve model indirme çalışmaz. |

> **En sık yapılan hata:** Internet'i açmayı unutmak. O zaman logda
> `Could not find a version that satisfies the requirement...` görürsünüz.

---

## 2. Notebook mantığı: hücre hücre çalışan defter

`notebooks/kaggle_training.ipynb` bir defter. İçinde **hücre** denen kutular var;
her kutu bir parça kod. İki çalıştırma biçimi:

| Biçim | Ne yapar | Ne zaman |
| --- | --- | --- |
| **Interactive** (tek tek) | Seçtiğiniz hücreyi çalıştırır, sonucu hemen altında gösterir | Öğrenirken, hata ayıklarken |
| **Save & Run All (Commit)** | Baştan sona sırayla çalıştırır, arka planda koşar, sonucu kaydeder | Uzun eğitimde (12 saat sınırına kadar) |

Bizim defterdeki 11 adım:

| # | Hücre | Ne yapıyor | Kaç dakika |
| --- | --- | --- | --- |
| 1 | Ortam bilgisi | Python sürümü, RAM, `nvidia-smi` ile GPU | saniye |
| 2 | Paketler | `pip install` (transformers, unsloth…) | 3–8 dk |
| 3 | Depo + Secrets | Kodu çeker, API anahtarını ortam değişkenine alır | 1 dk |
| 4 | Ön denetim | "Bu donanımda ne çalışır?" sorusunun cevabı | 1 dk |
| 5 | Veri doğrulama | JSONL'i şemaya göre denetler | 1–2 dk |
| 6 | Veri planı | 10.000 örneklik plan + maliyet (ağa çıkmaz) | saniye |
| 7 | Duman testi | Rastgele küçük modelle 2 adım (boru hattı testi) | 1–2 dk |
| 8 | **Gerçek eğitim** | 60 adımlık ölçüm koşusu | 10–25 dk |
| 9 | Kayıp kontrolü | Manifesti okur, metrikleri yazar | saniye |
| 10 | Değerlendirme | Test seti envanteri + biçim kontrolü | saniye |
| 11 | Kaydet | Çıktıları zip'ler | saniye |

**Kritik:** 8. hücre 60 adımda durur. Bu bilinçli — 12 saatlik oturumu yakmadan
"eğitim sağlıklı mı?" sorusunu cevaplamak için. Sağlıklıysa `max_steps: -1`
yapıp tam koşuya geçilir.

---

## 3. İlk kurulum (5 dakika, bir kez)

1. Kaggle'da **Code → New Notebook**.
2. Sağ üstte **File → Import Notebook** ile `kaggle_training.ipynb` yükleyin.
3. Sağ panelde **Settings**:
   - **Accelerator → GPU T4 x2** ← bu olmadan eğitim olmaz
   - **Internet → On** ← bu olmadan `pip install` olmaz
4. **Add-ons → Secrets → Add a new secret**, dört tane ekleyin:
   `EVREN_API_KEY`, `EVREN_BASE_URL`, `EVREN_MODEL_MIMO`, `EVREN_MODEL_GLM`.
   Her birinin yanındaki **"Attach"** düğmesine basın (bu adım atlanırsa
   defter anahtarı görmez).
5. Veri dosyanızı **Datasets → New Dataset** ile yükleyin. Adı ne olursa olsun,
   Kaggle onu `/kaggle/input/<veri-adı>/...` altına bağlar.

---

## 4. Log nerede?

Üç ayrı yer var, karıştırılmasın:

| Yer | Nerede | Ne var |
| --- | --- | --- |
| **Hücre çıktısı** | Her hücrenin hemen altında | O hücrenin yazdırdığı şey (bizim `print`'lerimiz) |
| **Session log** | Sağ panelde "Logs" / üstte "View logs" | Her şey, ham hâliyle; çökme sebepleri |
| **Dosyalar** | Sağ panel "Output" veya `/kaggle/working` | `run_manifest.json`, `plan.json`, zip |

Bir şey ters gittiyse **önce hücre çıktısına**, orada yoksa **session log'a**
bakılır. Bizim betik önemli her şeyi hücre çıktısına da yazdırıyor, bu yüzden
çoğu zaman session log'a inmeye gerek kalmaz.

---

## 5. Bizim betiğin çıktısını okuma (en önemli bölüm)

4. hücre şuna benzer bir şey basar. **Satır satır** anlamı:

```
=== Donanım ===
Python 3.11.13 | Linux 6.6 | CPU 4 çekirdek
RAM 29.0 GB | boş disk 19.2 GB
CUDA 12.4 | torch 2.6.0+cu124 | 2 GPU
  [0] Tesla T4: 15.6 GB, compute 7.5, bf16=desteklenmiyor
  [1] Tesla T4: 15.6 GB, compute 7.5, bf16=desteklenmiyor
```

- `2 GPU` görüyorsanız donanım doğru.
- `bf16=desteklenmiyor` **normaldir**, T4'ün sınırı. Korkmayın, fp16/float32
  yoluna düşülür. Ama nedeniyle birlikte "dikkat" uyarısı çıkar.

```
=== Bellek tahmini (unsloth-qlora, fp16) ===
  temel ağırlıklar ... ≈14.0 GB
```

- `≈` işareti **tahmin** demek; kesin değil. `14.0` Unsloth'un yayımladığı
  rakam. 15.6 GB'lık karta sığıyor *görünüyor* → makul.

```
=== Bulgular ===
  [i] unsloth-vram-ok: VRAM 15.6 GB ≥ yayımlanmış ≈14 GB gereksinim.
  [!] unsloth-fp16-moe: GPU bfloat16 desteklemiyor (T4)...
  [✗] unsloth: Unsloth kurulu değil
```

İşaretlerin anlamı:

| İşaret | Adı | Ne yapmalı |
| --- | --- | --- |
| `[i]` | bilgi | Hiçbir şey. Sadece durum bilgisi. |
| `[!]` | uyarı | Devam edilebilir, ama göz kulak ol. |
| `[✗]` | **engelleyici** | Eğitim başlamaz. Mutlaka çözülmeli. |

En altta tek satır:

```
SONUÇ: EĞİTİM İÇİN UYGUN     → devam
SONUÇ: ENGELLEYİCİ SORUN VAR → dur, yukarıdaki [✗] satırlarını oku
```

> **Sık görülen:** `[✗] unsloth: Unsloth kurulu değil` → 2. hücre (paket
> kurulumu) çalışmamış veya `Runtime → Restart` gerekiyor. En basit çözüm:
> menüden **Run → Restart & Run All**.

---

## 6. Eğitim logunu okuma (Trainer satırları)

8. hücre çalışırken akan satırlar iki türlüdür.

### 6a. İlerleme çubuğu

```
 42%|████▏     | 25/60 [04:12<05:52, 10.08s/it]
```

| Parça | Anlamı |
| --- | --- |
| `42%` | İşin yüzde kaçı bitti |
| `25/60` | 60 adımın 25'i tamamlandı |
| `04:12` | Geçen süre |
| `<05:52` | **Kalan tahmini süre** ← en çok işinize yarayacak sayı |
| `10.08s/it` | Adım başına saniye ("it" = iteration) |

**Bütçe hesabı:** `10 s/adım × 60 adım = 10 dakika`. Tam koşu için
`10 s/adım × toplam adım`. 9.600 örnek, batch 8 → 1.200 adım → ~3,5 saat.
12 saatlik oturuma sığar; 2.000 adımı geçerse bölmek gerekir.

### 6b. Kayıp satırı (süslü parantezli)

```
{'loss': 1.8342, 'grad_norm': 0.612, 'learning_rate': 4.8e-05, 'epoch': 0.05}
```

| Alan | Anlamı | İyi değer |
| --- | --- | --- |
| `loss` | Modelin hata puanı. **Tek bakılacak sayı bu.** | Düşüyor olmalı |
| `grad_norm` | Öğrenme sinyalinin büyüklüğü | < 10. 40+ ise sıkıntı |
| `learning_rate` | O adımdaki öğrenme hızı | Cosine ile düşer, normal |
| `epoch` | Verinin kaç kez baştan sona görüldüğü | 0→1 artar |

### 6c. Trafik ışığı — kayıp eğrisi

Elinizde şöyle bir seri var diyelim:

```
2.41  1.83  1.50  1.29  1.11  1.00  0.91  0.85
```

Bu **sağlıklı**: her satırda düşüyor. Mutlak değeri önemsemeyin, **yönü**
önemseyin.

| Görüntü | Yorum | Yapılacak |
| --- | --- | --- |
| 2.4 → 0.8 düzenli düşüş | 🟢 Sağlıklı | Devam |
| 2.4 → 2.3 → 2.4 → 2.3 (yassı) | 🟡 Öğrenmiyor olabilir | Daha uzun koş; düzelmezse lr/veri kontrol |
| 1.5 → 3.2 → 7.1 (yükseliyor) | 🔴 **Iraksama** | Durdur. Learning rate'i yarıya indir |
| `nan` veya `inf` | 🔴 **Bozuk** | Durdur. lr düşür, fp16 yerine bf16/float32 |
| 0.8'e indi, sonra 1.3'e çıktı | 🟡 Aşırı öğrenme | Erken checkpoint kullan |

> **En önemli kural:** `loss` sayısı **modelin ne kadar iyi konuştuğunu
> söylemez**. Sadece "öğrenme işlemi sağlıklı mı" sorusunu cevaplar. Kayıp
> 0.9 olan bir model saçmalıyor olabilir; kalite ayrı ölçülür (`evaluate.py`
> + insan incelemesi).

---

## 7. Log okumak istemiyorsanız: `gokturk-logcheck`

Logu **kopyalayıp bir dosyaya** yapıştırın, sonra:

```bash
python -m gokturk.logcheck --log kaggle_log.txt
```

Çıktı örneği:

```
============================================================
KARAR: SAĞLIKLI — eğitim mekanik olarak düzgün görünüyor
============================================================

--- Özet sayılar ---
  steps                  8
  first_loss             2.4132
  last_loss              0.8511
  min_loss               0.8511
  progress               (60, 60)

--- Bulgular ---
  [✓] loss-decreasing: Kayıp genel olarak düşüyor.
        kanıt: ilk: 2.4132
        kanıt: en iyi: 0.8511
```

Dört olası karar:

| Karar | Anlamı |
| --- | --- |
| **SAĞLIKLI** | Mekanik olarak düzgün. Kalite iddiası değil. |
| **DİKKAT** | Çalışıyor ama yassı kayıp, eval farkı, küçük veri gibi bir konu var. |
| **BOZUK** | NaN veya ıraksama. Bu checkpoint kullanılmaz. |
| **BAŞARISIZ** | Çöktü (OOM, 401, eksik dosya…). Altındaki `yapılacak:` satırı çözümü yazar. |

Biten bir koşu için manifest de okutabilirsiniz:

```bash
python -m gokturk.logcheck --manifest /kaggle/working/gokturk-run/run_manifest.json
```

Bu mod ayrıca "duman testi mi gerçek koşu mu", "eğitim seti çok mu küçük",
"eval kaybı train'den çok mu yüksek" gibi şeyleri söyler.

---

## 8. Hata sözlüğü — logda bunu görürseniz

| Logda gördüğünüz | Anlamı | Çözüm |
| --- | --- | --- |
| `CUDA out of memory` | 16 GB doldu | `max_seq_len: 512`, `batch: 1`, `grad_accum: 16` |
| `Killed` | 29 GB RAM bitti | Veri dosyasını küçült |
| `No module named 'unsloth'` | Paket kurulmamış | 2. hücreyi çalıştır + **Runtime → Restart** |
| `401 Client Error` | API anahtarı yok/yanlış | Secrets ekle + **Attach**'e bas |
| `No such file or directory: data/...` | Veri bulunamadı | Dataset eklendi mi? `/kaggle/input/...` yolu |
| `Could not find a version that satisfies` | Internet kapalı | Settings → Internet → On |
| `Read-only file system` | `/kaggle/input`'a yazmaya çalışılıyor | Çıktıyı `/kaggle/working`'e yaz |
| `Session was stopped` | 12 saat doldu | Checkpoint'ten devam et |
| `loss: nan` | Sayısal bozulma | lr ↓, `max_grad_norm: 0.5` |
| `Repository Not Found` | Model kimliği yanlış/gated | `name_or_path` doğrula, `HF_TOKEN` ekle |

---

## 9. Oturum bütçesi ve "iş yarıda kaldı" durumu

- **12 saat** sınırı: uzun koşuyu bölün. `save_steps: 30` sayesinde her 30
  adımda checkpoint yazılır.
- **Devam etme:** aynı defterde eğitim hücresini tekrar çalıştırırken
  `resume_from_checkpoint` kullanılabilir; bu repoda otomatik değildir,
  `outputs/.../checkpoint-XXX` klasörünü kontrol edin.
- **Kalıcılık:** `/kaggle/working` **20 GB** ve oturum bitince garanti değil.
  Adaptörü kalıcı yapmak için **Output → New Dataset** ile yayımlayın veya
  Hugging Face'e push edin.
- **Kota:** ~30 saat/hafta. Bir 12 saatlik yanlış koşu kotanın %40'ı demek —
  bu yüzden önce 60 adımlık ölçüm koşusu.

---

## 10. Bu rehberin sınırı

Bu belge **altyapıyı** anlatır. Şunları söylemez ve söyleyemez:

- Modelin iyi Türkçe konuştuğunu (ölçülmedi),
- Benchmark skorlarını (koşulmadı),
- Unsloth'un 14 GB iddiasının bu Kaggle hesabında doğrulandığını
  (bu repoda gerçek GPU koşusu yapılmadı).

Dürüst kontrol listesi: **kayıp düşüyor mu** → **NaN yok mu** → **manifest
yazıldı mı** → **değerlendirme + insan incelemesi yapıldı mı**. Bu dördü
tamamlanmadan "model eğitildi" denmez.
