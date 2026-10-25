# data/ dizini

| Dizin | İçerik | Repoda tutulur mu? |
| --- | --- | --- |
| `data/seeds/` | Öğretmen modellere verilen Türkçe tohum görevler | Evet (küçük, sürüm kontrollü) |
| `data/eval/` | Sabit, elle yazılmış değerlendirme seti | Evet (küçük, sürüm kontrollü) |
| `data/samples/` | Boru hattını denemek için elle yazılmış örnekler (`sample_train.jsonl` 6, `sample_pro.jsonl` 3) | Evet (küçük, **eğitim veri seti değildir**) |
| `data/raw/` | Öğretmenlerden üretilen ham JSONL | **Hayır** (`.gitignore`) |
| `data/processed/` | Filtrelenmiş train/valid/test JSONL | **Hayır** (`.gitignore`) |

## Veri akışı

### PRO modu (önerilen — 10 aile, çok adaylı, öz-eleştirili)

```bash
# 1a) Plan + maliyet tahmini (ağa çıkmaz)
python -m gokturk.data_generation --mode pro --target 10000 \
  --plan-only --plan-out data/raw/plan.json

# 1b) Üretim (EVREN_API_KEY + doğrulanmış model kimlikleri gerekir)
python -m gokturk.data_generation --mode pro --target 10000 \
  --out data/raw/pro.jsonl --candidates 2 --seed 7
```

Çıktılar: `data/raw/pro.jsonl` (kabul edilen), `pro.rejects.jsonl` (ret
nedenleri), `pro.critique.jsonl` (öz-eleştiri kayıtları).

### Basic mod (tohum dosyasından)

```bash
python -m gokturk.data_generation \
  --teachers-config configs/teachers.yaml \
  --seeds data/seeds/turkish_seed_tasks.jsonl \
  --out data/raw/generated.jsonl --per-seed 1

# 2) Kalite kontrolü + tekrar temizliği + bölme
python - <<'PY'
from gokturk.data_quality import build_datasets
report = build_datasets("data/raw/generated.jsonl", "data/processed")
print(report.summary())
PY

# 3) Biçim denetimi (model kalitesi ölçmez)
python -m gokturk.evaluate --mode format --data data/processed/train.jsonl
```

`data/processed/data_report.json` her koşuda kabul/ret sayılarını ve ret
nedenlerini kaydeder. Veri seti sürümü, bu raporun ve üretim çıktısının
birlikte arşivlenmesiyle sabitlenir (ağırlıklar gibi veri de repoya girmez).
