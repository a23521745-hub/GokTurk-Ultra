# data/ dizini

| Dizin | İçerik | Repoda tutulur mu? |
| --- | --- | --- |
| `data/seeds/` | Öğretmen modellere verilen Türkçe tohum görevler | Evet (küçük, sürüm kontrollü) |
| `data/eval/` | Sabit, elle yazılmış değerlendirme seti | Evet (küçük, sürüm kontrollü) |
| `data/samples/` | Boru hattını denemek için 6 elle yazılmış örnek | Evet (küçük, **eğitim veri seti değildir**) |
| `data/raw/` | Öğretmenlerden üretilen ham JSONL | **Hayır** (`.gitignore`) |
| `data/processed/` | Filtrelenmiş train/valid/test JSONL | **Hayır** (`.gitignore`) |

## Veri akışı

```bash
# 1) Üretim (öğretmen API'si gerekir; anahtar ortam değişkeninde olmalı)
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
