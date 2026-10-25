# Kaggle'da gerçek koşu — kontrol listesi (Qwen3.5-4B)

Bu belge, **ilk gerçek GPU koşusunu** adım adım yürütür: Kaggle'da ücretsiz
T4 üzerinde 4-bit QLoRA ile `Qwen/Qwen3.5-4B` eğitimi, adaptörün indirilmesi,
Hugging Face'e yüklenmesi ve servis edilmesi.

Neden 4B ve neden 14B değil: [`docs/taban-model-secimi.md`](taban-model-secimi.md)
§2. Kısa hâli: **disk, VRAM'den önce gelir.** Kaggle'da ölçülen boş alan
19,2 GB; 14B'nin bf16 indirmesi ≈29,6 GB (sığmaz), 9B ≈18 GB (sınırda),
4B ≈9,3 GB (rahat).

Beklenti: bu koşu **kalite** üretmez, **boru hattını gerçek donanımda kanıtlar**
(veri → ön denetim → QLoRA → manifest → logcheck → adaptör-only GGUF → HF →
servis). 60 adım eğitilmiş 4B model "Türkçe talimat takibi" öğrenmez; öğrenip
öğrenmediğini ölçmek `gokturk.evaluate` + insan incelemesi işidir.

---

## 0. Kaggle gerçekleri (kontrol edilmiş sayılar)

| Kaynak | Değer | Not |
| --- | --- | --- |
| GPU | 2 × Tesla T4, 16 GB | Betik **tek kart** kullanır; ikincisi boş durur |
| Oturum | 12 saat | Uzun koşuyu bölün |
| Haftalık kota | ~30 GPU saati | 4B/60 adım bunun çok altında |
| RAM | ~29 GB | Veri dosyası sorun değil |
| Disk | 20 GB (`/kaggle/working`) | Kalıcı olan tek dizin; **asıl kısıt bu** |
| bf16 | **Yok** (compute 7.5) | `dtype: auto` → fp16; MoE'de ıraksama riski (bu yüzden 4B dense) |

---

## 1. Veri (koşudan önce, GPU harcamadan)

```bash
# Örnek veriyle iskeleti doğrula (ağ yok, ücret yok)
python -m gokturk.data_generation --mode pro --plan-only --target 10000

# Gerçek üretim (EVREN_API_KEY gerekir; Kaggle Secrets'ta tutun)
python -m gokturk.data_generation --mode pro --teachers-config configs/teachers.yaml \
  --target 10000 --out data/raw/generated.jsonl
```

- Kendi 10.000 örneklik Türkçe veri setiniz varsa aynı şemayı kullanın:
  her satırda `prompt` ve `response`.
- **Sızıntısız bölme:** `python -m gokturk.data_quality --split ...` →
  `data/processed/train.jsonl` ve `data/processed/valid.jsonl`. Config bu iki
  yolu bekler.
- Kontrol: `data/processed/train.jsonl` ≥ 9.000 satır, `valid.jsonl` ayrı
  gruplardan (aynı prompt ailesi iki tarafta olmamalı).

---

## 2. Kaggle notebook'unu hazırla

1. Yeni notebook → **Accelerator: GPU T4 ×2**, **Internet: On**.
2. Repoyu klonla ve kur:

```bash
!git clone https://github.com/a23521745-hub/GokTurk-Ultra.git
%cd GokTurk-Ultra
!pip install -q -e ".[train,qlora]"     # torch CUDA tekerleği burada sorun değil
```

3. İlk iş **diski ölçmek** (bu belgenin varlık sebebi):

```bash
!df -h /kaggle/working; free -h
```

4. Ön denetim (model indirmez, GPU harcamaz):

```bash
!python -m gokturk.train --config configs/training_qwen3_5_4b_t4.yaml --plan-only
```

Beklenen: `vram` ve `disk` bulguları **temiz**; `fp16-fallback` bilgi/uyarı
olarak görünür (T4 bf16 desteklemez). `weights-disk-insufficient` görürseniz
**durmayın, devam etmeyin** — daha küçük tabana geçin.

---

## 3. Yükleme doğrulaması (çok kipli tabanın metin yolu)

Qwen3.5 çok kiplidir ama metin SFT'si için `AutoProcessor` gerekmez: transformers
`Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected = [r"^mtp.*",
r"^model.visual.*"]` ile görüntü kulesini yok sayar. Bunu **gerçek checkpoint
üzerinde** bir kez görmek gerekir:

```bash
import transformers, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=torch.float16, device_map={"": 0}
)
print(type(model).__name__)          # Qwen3_5ForCausalLM beklenir
```

- Yükleme sırasında `model.visual.*` anahtarları **"unexpected"** olarak
  listelenmeli. Listelenmiyorsa ya görüntü kulesi de yüklenmiştir (VRAM artar)
  ya da sınıf eşlemesi değişmiştir → sürüm notlarını kontrol edin.
- `Qwen3_5ForCausalLM` yerine başka bir sınıf gelirse: `transformers` sürümünü
  yükseltin (`pip install -U transformers`) veya `Qwen3_5TextConfig` ile metin
  alt-modelini açıkça yükleyin.

---

## 3.5 Bilinen hata: Unsloth füzyonlu kayıp (fused loss)

**Belirti (2026-10-25, Kaggle, notebook'un eski varsayılanı):** eğitim ilk
adımda düşer, iz:

```
File ".../unsloth_zoo/fused_losses/cross_entropy_loss.py", line 506, in accumulate_chunk
    (chunk_loss, (unscaled_loss,)) = torch.func.grad_and_value(
RuntimeError: grad_and_value(f)(*args): Expected f(*args) to return a scalar Tensor, got tensor with 2 dims. Maybe you wanted to use the vjp or jacrev APIs instead?
```

**Ne değil:** veri hatası, config hatası, disk/VRAM sorunu değil. Model yüklendi,
ilk `forward` çalıştı; hata Unsloth'un füzyonlu CE çekirdeğinin içinde.

**Nedeni:** `unsloth_zoo`'nun füzyonlu kayıp çekirdeği, kurulu `torch` /
`transformers` sürümüyle uyumsuz. Kaggle imajı 2026-10-25 itibarıyla
Python 3.13 + güncel torch getiriyor; Unsloth tarafında bu kombinasyon için
düzeltilmiş bir sürüm yok. `logcheck` bu izi tanır:
`python -m gokturk.logcheck --log train.log` → `unsloth-fused-loss` (BAŞARISIZ).

**Çözümler, tercih sırasıyla:**

1. **Unsloth'u hiç kullanmayın (önerilen).** 4B'de Unsloth gerekmez; bu repodaki
   transformers + bitsandbytes yolu (NF4 QLoRA) 4B'yi T4'e rahat sığdırıyor:

   ```bash
   python -m gokturk.train --config configs/training_qwen3_5_4b_t4.yaml \
     --max-steps 60 --output-dir /kaggle/working/run-4b
   ```

   Notebook'ta `RUN_PROFILE = "qwen3.5-4b"` (varsayılan) bu yolu seçer.
2. **Füzyonlu kaybı kapatın:** `--unsloth-plain-loss` (arkaplanda `import
   unsloth`'tan önce `UNSLOTH_RETURN_LOGITS=1`). Düz logits + standart CE
   kullanılır; bedeli **daha çok VRAM** — gpt-oss-20b'de 16 GB'a sığmayabilir,
   4B'de sorun değil.
3. **Normalizasyon anahtarını kapatın:** `TrainingArguments(...,
   average_tokens_across_devices=False)`. Tek GPU'da matematik değişmez; ilgili
   Unsloth hatası (#3769) bu bayrakla aşılıyor.
4. Sürümleri sabitleyin: `unsloth` ve `unsloth_zoo`'yu aynı sürüme kilitleyin;
   Kaggle'da her oturumda `pip install unsloth` çekmek sürüm kayması demektir.

Unsloth yalnızca **gpt-oss-20b** gibi generic yolun sığmadığı modellerde gerekir;
o koşu ertelendi (bkz. `docs/taban-model-secimi.md`).

---

## 3.6 İkinci hata: iki GPU'ya kopyalama (`replica 0`)

**Belirti (2026-10-25, ikinci koşu):** aynı iz, ama üstünde `torch/nn/parallel/data_parallel.py`
ve `RuntimeError: Caught RuntimeError in replica 0 on device 0` satırları var.

**Ne oluyor:** Kaggle iki T4 gösterdiği için `Trainer`/`accelerate` modeli
`nn.DataParallel` ile iki karta kopyalamaya çalışıyor. Bu repo hiçbir yerde model
paralelliği uygulamaz ve **4-bit (bitsandbytes) ağırlıklar çoğaltılamaz**; hata
kopyanın içinde, asıl nedenden çok uzakta görünür.

**Çözüm (artık varsayılan):** süreç tek kart görür.

- Notebook: ilk kod hücresi, `torch` import edilmeden önce
  `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")` yapar.
- CLI: `python -m gokturk.train ...` varsayılan `--gpus 0` ile aynı şeyi yapar;
  `CUDA_VISIBLE_DEVICES` zaten ayarlıysa **ezilmez**.
- `--gpus 0,1` yazarsanız uyarı basılır: bu yol 4-bit modellerde çalışmaz.
- `logcheck` bu izi `dataparallel-replica` olarak tanır ve tek satırda ne
  yapılacağını söyler.

Unsloth yolu da (gpt-oss-20b) aynı kurala tabidir: tek kart.

**Not:** ikinci koşunun izinde hâlâ `unsloth_zoo` kareleri var, yani o koşu
Unsloth arka ucuyla başlamış. 4B profili `--backend transformers` kullanır;
notebook'u güncel sürümden açtığınızdan emin olun (`git pull`, sonra 8. hücre:
`RUN_PROFILE = "qwen3.5-4b"`). Unsloth'ta kalmak isterseniz bu repoda artık
füzyonlu kayıp **varsayılan olarak kapalı** (`--unsloth-fused-loss` ile açılır).

---

## 4. 60 adımlık ölçüm koşusu

```bash
!python -m gokturk.train --config configs/training_qwen3_5_4b_t4.yaml \
  --max-steps 60 --output-dir /kaggle/working/run-4b
```

Çıktı: `/kaggle/working/run-4b/adapter/` + `run_manifest.json`.

Coşkuya kapılmadan önce bakılacaklar:

```bash
!python -m gokturk.logcheck --manifest /kaggle/working/run-4b/run_manifest.json
```

- `BOZUK` / `BAŞARISIZ` → adım sayısını artırmayın; `learning_rate` 5e-5,
  `max_seq_len` 512 ile tekrar deneyin.
- `DİKKAT` → normaldir (küçük veri, kısa koşu). Kaybın **düştüğünü** doğrulayın.
- `train_runtime` ve s/adım değerini not edin: 4B'de 60 adım tipik olarak
  oturumun küçük bir kısmını alır; 9B'ye geçerken bütçe hesabı bununla yapılır.

Kararlıysa adımı artırın (ör. 300–600) **ama** önce `save_steps: 30` ile
checkpoint aldığınızdan emin olun (12 saat sınırı).

---

## 5. Adaptörü indir (birleştirme YOK)

`adapter/` klasörü tipik olarak 30–120 MB: `adapter_config.json`,
`adapter_model.safetensors`. `run_manifest.json`'ı da yanına alın.

- Kaggle'da **Output** sekmesinden indirin, ya da doğrudan HF'ye yükleyin:

```python
!python -m gokturk.publish --adapter /kaggle/working/run-4b/adapter \
  --repo-id <kullanici>/gokturk-ultra-lora --manifest /kaggle/working/run-4b/run_manifest.json \
  --base-model Qwen/Qwen3.5-4B --push
```

`HF_TOKEN`'ı `kaggle_secrets` ile alın; token'ı hücreye yazmayın. Depo özel
açılır; herkese açık istiyorsanız `--public` ekleyin.

**Yapılmayacak:** `gokturk.publish --merge`. Bu repoda birleştirme varsayılan
değil; adaptör + taban ayrı taşınır (bkz. `docs/servis-ve-dinamik-lora.md`).

---

## 6. Servis ve doğrulama

```bash
# CPU'da / Space'te çalışır mı, kaç token/sn?
python -m gokturk.publish --serve-plan --base-model Qwen/Qwen3.5-4B --ram-gb 16 --vcpu 2

# Space iskeleti (taban + adaptör, birleştirme yok)
python -m gokturk.serve --emit-space space/ \
  --base-model Qwen/Qwen3.5-4B --adapter-repo <kullanici>/gokturk-ultra-lora
```

4B Q4_K_M ≈2,4 GB, 16 GB RAM / 2 vCPU'da 3,3–8,3 tok/s: **konuşur**, hızlı
değil. 9B ≈5,5 GB ve 1,5–3,6 tok/s ile sınırda. ZeroGPU kotası (5 dk/gün)
değişmedi.

---

## 7. Kısa kontrol listesi (kopyala–yapıştır)

- [ ] `df -h /kaggle/working` → boş alan ≥ 12 GB (4B için)
- [ ] `--plan-only` → `weights-disk-insufficient` **yok**
- [ ] Yükleme testi → sınıf `Qwen3_5ForCausalLM`, `model.visual.*` unexpected
- [ ] `RUN_PROFILE = "qwen3.5-4b"` (Unsloth yolu değil; fused-loss hatası için §3.5)
- [ ] `CUDA_VISIBLE_DEVICES=0` (ilk hücre; `replica 0` hatası için §3.6)
- [ ] 60 adım → `logcheck` `SAĞLIKLI` veya `DİKKAT` (kayıp düşüyor)
- [ ] `adapter/` + `run_manifest.json` indirildi
- [ ] HF push (özel depo) → adaptör boyutu 30–120 MB
- [ ] `--serve-plan` → 4B uygun; token/sn not edildi
- [ ] **Hiçbir yerde** "eğitildi, Türkçe öğrendi" iddiası yok — ölçüm yok

---

## 8. 9B'ye geçiş (2. koşu)

9B'de disk payı kritik: ≈18 GB indirme + adaptör + log. Yapılacaklar:

1. `configs/training_qwen3_5_9b_t4.yaml` içinde `max_seq_len: 1024`,
   `save_total_limit: 1`, `per_device_train_batch_size: 1`,
   `gradient_accumulation_steps: 16`.
2. `df -h` ile payı doğrula; `weights-disk-tight` uyarısı **beklenir**.
3. Ara checkpoint'leri ve `outputs/` artıklarını silerek yer açın.
4. Aynı 60 adım ölçümü → s/adım × adım sayısı ile 12 saat bütçesini hesaplayın.
