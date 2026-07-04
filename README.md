# Kaggle Playground Series S6E7 — EDA ve Modelleme Workspace

Bu depo model eğitiminden önce veri, missingness, distribution shift, outlier ve feature engineering kararlarını belgeler.

## Ana preprocessing: P_MAIN_V2_CORE

Fold içinde `V2CorePreprocessor.fit(fold_train)` çağrılır; median ve %0.5/%99.5
outlier eşikleri yalnız fold-train'den öğrenilir. Gender interaction, rule flag ve
clipping ana pipeline'a dahil değildir.

```python
from kaggle_s6_e7.preprocessing import V2CorePreprocessor

preprocessor = V2CorePreprocessor()
X_fold_train = preprocessor.fit_transform(X_fold_train_raw)
X_fold_valid = preprocessor.transform(X_fold_valid_raw)
```

Tüm train üzerinde model-ready artefakt üretmek için:

```bash
uv run python scripts/prepare_v2_core.py
```

## Kurulum

```bash
uv sync --dev
```

## Notebook sırası

1. `01_basic_eda.ipynb`
2. `02_missing_analysis.ipynb`
3. `03_distribution_shift.ipynb`
4. `04_outlier_analysis.ipynb`
5. `05_feature_engineering_candidates.ipynb`

Notebookları proje kökünden açın:

```bash
uv run jupyter lab
```

Tablolar `reports/tables/`, grafikler `reports/figures/` altında oluşur. Doğrulanan yorum ve kararlar `reports/eda_findings.md` içine elle, `Gözlem/Aday/Kabul/Red` etiketleriyle aktarılır.

## Kalite kontrolleri

```bash
uv run python scripts/check.py
```


Bu komut pytest, Ruff, Mypy ve Python bytecode derleme kontrolünü birlikte çalıştırır.

## Deney düzeni

Deneyler `E001`, `E002`, … biçiminde planlanacak. Kayıt ve artefaktlar `experiments/E###/`, submission CSV dosyaları `submissions/E###_submission.csv` altında tutulacak. Ayrıntılı sözleşmeler klasör README'lerinde ve `.agents/rules/` altında tanımlıdır.
