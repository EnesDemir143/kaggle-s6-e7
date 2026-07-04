# Kaggle S6E7 — Deney, Sweep ve Submission Uygulama Rehberi

Bu belge E001–E008 deneylerinin neyi ölçtüğünü, hangi sırada çalıştırılacağını, her
komutun hangi dosyaları ürettiğini ve günlük 10 submission hakkının nasıl kullanılacağını
tanımlar. Notebooklar yalnız EDA içindir; tekrar üretilebilir model deneyleri bu belgedeki
scriptlerle çalıştırılır.

> **Önemli:** `--dry-run` skorları model kalitesi veya submission seçimi için kullanılmaz.
> Dry-run yalnız kod, cache, artefakt ve veri akışını doğrular.

## 1. Ortak deney standardı

| Alan | Değer |
|---|---|
| Model | LightGBM multiclass |
| CV | StratifiedKFold, 3 fold, shuffle, seed 42 |
| Ana seçim metriği | Balanced accuracy |
| Yardımcı metrikler | Class recall, macro F1, confusion matrix, prediction distribution |
| Early stopping | 300 round, validation multi-logloss |
| Class sırası | `at-risk`, `fit`, `unhealthy` |
| Compute | CPU/OpenMP; Apple MPS kullanılmaz |
| Local thread ayarı | M2 Pro için `n_jobs=6`, `OMP_NUM_THREADS=6` |
| Satır silme | Yok |
| IQR temizleme | Yok |
| Feature istatistikleri | Yalnız fold-train üzerinde fit edilir |

Başlangıç model parametreleri `configs/lgbm_base.yaml`, deney tanımları
`configs/experiments.yaml` içindedir. Bu aşamada HPO yapılmaz; bütün feature ablation
deneyleri aynı model parametreleriyle karşılaştırılır.

### Apple M2 Pro çalışma ayarı

Bu makine 10 çekirdekli Apple M2 Pro (`6 performance + 4 efficiency`) ve 16 GB unified
memory kullanır. LightGBM’in macOS GPU sürümü desteklenmediği için MPS/Metal üzerinden
eğitim yapılmaz; native CPU/OpenMP build kullanılır. Gerçek proje feature’larıyla yapılan
120-tree benchmark sonucu:

| Thread | Ortalama süre |
|---:|---:|
| 6 | 4.27 sn |
| 8 | 5.13 sn |
| 10 | 6.19 sn |

Bu nedenle `n_jobs=-1` yerine altı performance core’u hedefleyen `n_jobs=6` kullanılır.
Runner ayrıca `OMP_NUM_THREADS=6` ayarlar. LightGBM tabular training’de neural network
tarzı `batch_size` parametresi yoktur; fold verisi LightGBM Dataset olarak işlenir.

## 2. Cache davranışı

Cache varsayılan olarak açıktır:

```text
outputs/cache/raw/       CSV -> content-addressed Parquet
outputs/cache/folds/     fold train/valid/test feature Parquet dosyaları
```

Fold cache anahtarı şunlardan oluşur:

- Train/test dosyalarının SHA-256 fingerprint’i
- Fold numarası ve fold-train indeks hash’i
- Tam feature config’i
- Cache schema version

Bu nedenle dry-run ile full run, E002 ile farklı feature config’leri veya değişmiş veri
birbirinin cache’ini yanlışlıkla kullanmaz. E008’in feature set’i E002 ile aynı olduğu için
E002 fold cache’ini güvenli biçimde yeniden kullanır; yalnız sample weights değişir.

Cache kullanmadan teşhis koşusu:

```bash
uv run python scripts/run_experiment.py --exp E002 --no-cache
```

Cache temizleme:

```bash
rm -rf outputs/cache
```

Runner başlamadan önce varsayılan olarak en az 8 GiB boş disk kontrolü yapar.

## 3. Deney kataloğu

### E001 — V1 baseline

**Hipotez:** Median imputasyon, categorical sentinel ve missing flag’ler güvenli bir
referans skor oluşturur.

**Özellikler:**

- Numeric median imputation
- Categorical `missing` ve unknown handling
- Her ham kolon için missing flag
- Ratio, interaction, outlier, clipping ve class weight yok

```bash
uv run python scripts/run_experiment.py --exp E001
```

**Karşılaştırma rolü:** E002–E008’in tamamı E001’e göre yorumlanır.

### E002 — V2-Core

**Hipotez:** EDA tarafından önerilen ana feature set E001’i anlamlı biçimde geçer.

**E001 üzerine eklenenler:**

- `missing_count`
- Altı ratio feature
- `stress_sleep_quality`, `activity_diet`, `smoking_activity`
- Fold-train `%0.5/%99.5` outlier flag’leri
- `outlier_count`

```bash
uv run python scripts/run_experiment.py --exp E002
```

**Kabul:** E001’e göre `+0.003` güçlü sinyal; `+0.001–0.003` ek seed/CV5 doğrulama
adayıdır. E002 ana feature pipeline ve ilk multiplier kaynağıdır.

### E003 — V2-Core + gender_activity

**Hipotez:** Gender/activity interaction ek sinyal taşır; ancak train-test gender shift
nedeniyle genelleme riski vardır.

```bash
uv run python scripts/run_experiment.py --exp E003
```

**Kabul:** E002’yi en az yaklaşık `+0.0015–0.002` geçmeli ve fold std/prediction
distribution bozulmamalıdır. Aksi halde final feature set’e alınmaz.

### E004 — V2-Core + rule flags

**Hipotez:** Uyku, BMI, heart-rate ve step threshold’ları özellikle `fit` ve `unhealthy`
recall değerlerini artırır.

```bash
uv run python scripts/run_experiment.py --exp E004
```

**Kabul:** Sadece mean balanced accuracy değil, rare class recall artışı da görülmelidir.
Prediction distribution aşırı rare-class tahminine kayarsa reddedilir.

### E005 — V2-Core + clipping

**Hipotez:** Fold-train `%0.1/%99.9` clipping uzun kuyrukları yumuşatır.

```bash
uv run python scripts/run_experiment.py --exp E005
```

Outlier flag’leri clipping’den önce üretilir. E005 ancak E002’yi açıkça geçerse submission
adayı olur; aksi halde extreme değerlerin target sinyali korunur ve clipping reddedilir.

### E006 — V2-Core + log ratio variants

**Hipotez:** Original ratio’lara eklenen güvenli `log1p` varyantları uzun kuyrukları daha
kolay modelletir.

```bash
uv run python scripts/run_experiment.py --exp E006
```

**Kabul:** E002’yi geçerse log varyantları korunur; original ratio’ların yerine geçmez.

### E007 — OOF class multiplier tuning

E007 yeni model eğitmez. Eğitilmiş deneyin `oof_proba.npy` dosyası üzerinde karar
sınırlarını balanced accuracy’ye göre ayarlar.

```bash
uv run python scripts/tune_multipliers.py --exp E002
uv run python scripts/make_submission.py --exp E002 --postprocess multipliers
```

Arama:

1. `at-risk=1.0` referansıyla `fit` ve `unhealthy` için `0.80–1.50`, step `0.025` grid
2. En iyi nokta çevresinde `±0.05`, seeded 2,000 local random trial
3. Final multiplier vektörünü mean=1 olacak şekilde normalize etme

**Kabul:** Tuned OOF balanced accuracy argmax OOF değerini anlamlı biçimde geçmeli;
`at-risk` recall çökmemeli ve tahmin dağılımı gerçekçi kalmalıdır.

### E008 — V2-Core + sqrt balanced weights

**Hipotez:** Fold-train class dağılımından hesaplanan `sqrt_balanced` sample weights,
full balanced weight kadar agresif olmadan rare-class recall değerlerini artırır.

```bash
uv run python scripts/run_experiment.py --exp E008
uv run python scripts/tune_multipliers.py --exp E008
uv run python scripts/make_submission.py --exp E008 --postprocess multipliers
```

**Kabul:** E008 argmax ve E008 tuned sonuçları, E002 tuned ile balanced accuracy,
üç class recall ve prediction distribution birlikte değerlendirilerek seçilir.

## 4. Her model eğitiminin ürettiği dosyalar

Her eğitim `outputs/experiments/<EXP_ID>/` altında şunları üretir:

| Dosya | Kullanım |
|---|---|
| `config.json` | Çözülmüş feature/training/model config ve dry-run bilgisi |
| `metrics.json` | Overall OOF metrikleri ve fold mean/std |
| `fold_metrics.csv` | Her fold balanced accuracy, recall, macro F1, best iteration |
| `oof_proba.npy` | Threshold/multiplier ve blend için train OOF probabilities |
| `test_proba.npy` | Submission, multiplier ve blend için test probabilities |
| `oof_pred.csv` | ID, gerçek label ve argmax OOF prediction |
| `submission_argmax.csv` | Modelin doğrudan argmax submission’ı |
| `feature_importance.csv` | Fold ve mean gain/split importance |
| `classification_report.txt` | Precision, recall ve F1 raporu |
| `confusion_matrix.csv` | Sabit class sıralı OOF confusion matrix |
| `model_fold0.txt`–`model_fold2.txt` | Fold LightGBM modelleri |
| `label_mapping.json` | Class adı -> probability column sırası |
| `run_manifest.json` | Veri fingerprint, cache key ve runtime provenance |
| `progress.jsonl` | Streaming per-fold ve experiment-complete JSONL kaydı |
| `test_proba_fold.npy` | Per-fold test probabilities `(n_test × n_class × n_fold)` — blending/ensemble için |
| `oof_proba_fold.npy` | Per-fold OOF probabilities `(n_train × n_class × n_fold)`, sparse NaN — stacked ensemble için |
| `fold_assignments.csv` | Her satırın hangi fold'da valid olduğu `(-1=train-all)` |
| `training_history.json` | Her fold'un iterasyon başına train/valid multi_logloss — overfitting teşhisi |
| `roc_curves.png` | OOF multiclass OvR ROC eğrileri (görsel tanı) |

Multiplier tuning ayrıca şunları üretir:

| Dosya | Kullanım |
|---|---|
| `best_multipliers.json` | Class isimleriyle final multiplier değerleri |
| `metrics_tuned.json` | Tuned balanced accuracy, recall, macro F1 ve distribution |
| `submission_tuned.csv` | Tuned test prediction |

`make_submission.py` ayrıca açık isimli `submission_<EXP>_argmax.csv` veya
`submission_<EXP>_tuned.csv` üretir.

## 5. Önerilen iki aşamalı çalışma

Sweep base experiment’ı sonuçları görmeden seçilmemelidir. Bu nedenle en güvenli çalışma
iki komuttur.

### Aşama A — Eğitim, multiplier ve submission adayları

```bash
RUN_SWEEP=0 bash scripts/experiment_runner.sh
```

Bu komutun sırası:

```text
E001/E002/E003/E004/E005/E006/E008 training
    -> outputs/leaderboard_local.csv
    -> E002/E004/E006/E008 multiplier tuning
    -> argmax ve tuned submission dosyaları
```

Sonra incele:

```bash
column -s, -t < outputs/leaderboard_local.csv | less -S
cat outputs/experiments/E002/metrics_tuned.json
cat outputs/experiments/E004/metrics_tuned.json
cat outputs/experiments/E006/metrics_tuned.json
cat outputs/experiments/E008/metrics_tuned.json
```

### Aşama B — Kazanan feature pipeline üzerinde model sweep

Örneğin E004 seçildiyse:

```bash
BASE_EXP=E004 N_TRIALS=20 bash scripts/experiment_runner.sh
```

Runner tamamlanmış eğitim/multiplier/submission adımlarını atlar ve sweep’i çalıştırır.
Sweep trial’ları `outputs/experiments/SWEEP_000/`, `SWEEP_001/`, ... altında normal
deneylerle aynı probability ve metric artefaktlarını üretir.

Sweep sonrası leaderboard’u yenile:

```bash
uv run python scripts/compare_experiments.py --output outputs/leaderboard_local.csv
```

En iyi sweep trial örneğin `SWEEP_007` ise:

```bash
uv run python scripts/tune_multipliers.py --exp SWEEP_007
uv run python scripts/make_submission.py --exp SWEEP_007 --postprocess argmax
uv run python scripts/make_submission.py --exp SWEEP_007 --postprocess multipliers
```

## 6. “Raw rate sweep” uygulanacak mı?

Referans binary/ranking projedeki `positive-rate sweep`, tek pozitif sınıf oranını farklı
değerlere zorlar. Bu yarışma üç sınıflı olduğu için aynı raw positive-rate yaklaşımını
doğrudan kullanmak doğru değildir.

Bizdeki karşılığı **class multiplier sweep**’tir:

```python
prediction = argmax(probability * class_multipliers)
```

Bu işlem `tune_multipliers.py` tarafından yalnız OOF üzerinde yapılır. Test class rate’e
bakarak tuning yapılmaz. Böylece submission distribution’ı elle LB’ye uydurmak yerine
balanced accuracy için doğrulanmış karar sınırları kullanılır.

İlk turda ayrıca sabit class-rate zorlayan bir submission sweep yapılmayacaktır. Gerekirse
ilerleyen aşamada yalnız OOF ile doğrulanmış multiplier komşuluklarından birkaç farklı
submission adayı üretilebilir; günlük submit hakkı uğruna doğrulanmamış raw rate’ler
üretilmemelidir.

Model parameter sweep ise farklıdır: `run_sweep.py`, learning rate, leaves,
min-child-samples, regularization ve row/column sampling parametrelerini arar. Önce feature
set seçilir, sonra yalnız kazanan feature set üzerinde çalıştırılır.

## 7. Submission seçim kuralları

Bir aday yalnız mean balanced accuracy ile seçilmez. Şunlar birlikte incelenir:

1. Overall ve fold mean balanced accuracy
2. Fold standard deviation
3. `fit` recall
4. `unhealthy` recall
5. `at-risk` recall
6. Prediction distribution
7. Argmax ile tuned multiplier farkı
8. Benzer modellerin feature/model çeşitliliği

Pratik eşikler:

| OOF farkı | Karar |
|---|---|
| `>= +0.003` | Güçlü aday |
| `+0.001 – +0.003` | Ek seed veya CV5 ile doğrula |
| `< +0.001` | Gürültü kabul et; ancak çeşitlilik için blend adayı olabilir |

### İlk submission günü için önerilen bütçe

10 hakkın tamamını tek seferde kullanma. İlk turda en fazla 4–6 dosya:

1. `E002_tuned` — ana güvenli referans
2. E004 veya E006’dan OOF’ta daha iyi olanın `tuned` sonucu
3. `E008_tuned` — yalnız recall dağılımı sağlıklıysa
4. Kazanan sweep trial’ın `tuned` sonucu
5. Gerekirse kazanan deneyin argmax sonucu; multiplier’ın gerçek katkısını ölçmek için
6. İleride OOF ile doğrulanmış probability blend

E003, E005 veya zayıf sweep trial’ları yalnız OOF’ta açık sinyal varsa gönderilir. Aynı
modelin çok benzer multiplier varyantlarıyla günlük hak tüketilmez.

### Submission öncesi son kontrol

```bash
uv run python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("outputs/experiments/E002/submission_E002_tuned.csv")
df = pd.read_csv(path)
print(path)
print(df.shape)
print(df["health_condition"].value_counts(normalize=True))
print(df.head())
PY
```

Beklenenler:

- Satır sayısı `295753`
- Kolonlar tam olarak `id`, `health_condition`
- ID sırası sample submission ile aynı
- Label’lar yalnız `at-risk`, `fit`, `unhealthy`
- Tek sınıfa tamamen çökmemiş prediction distribution

## 8. Tek komut ve dry-run komutları

Tam pipeline’ı sonuç görmeden varsayılan E002 üzerinde sweep dahil çalıştırmak mümkündür:

```bash
bash scripts/experiment_runner.sh
```

Ancak önerilen yöntem önce sweep’i kapatmak, sonuçlardan `BASE_EXP` seçmek ve ikinci kez
runner çalıştırmaktır.

Tüm zincirin küçük doğrulaması:

```bash
bash scripts/dry_run_pipeline.sh
```

Tek deney dry-run:

```bash
uv run python scripts/run_experiment.py --exp E004 --dry-run --force
```

Dry-run artefaktları `outputs/dry_runs/` altında tutulur ve gerçek deneylerle karışmaz.
