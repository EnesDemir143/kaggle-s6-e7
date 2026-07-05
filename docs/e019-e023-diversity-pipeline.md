# E019–E023 Diversity Pipeline

Bu pipeline E002 probability artefaktını ana omurga olarak tutar, bir XGBoost ve bir
CatBoost V2-Core modeli eğitir ve üç küçük probability ensemble üretir.

## Deneyler

| Deney | Tanım |
|---|---|
| E019 | XGBoost V2-Core, sqrt-balanced |
| E020 | CatBoost V2-Core, sqrt-balanced |
| E021 | %95 E002 + %5 E019 |
| E022 | %97 E002 + %3 E020 |
| E023 | %94 E002 + %3 E019 + %3 E020 |

Multiplier scale'leri ham multiplier değildir. Fit ve unhealthy scale değerleri E002'nin
`best_multipliers.json` değerleriyle çarpılır; at-risk multiplier değiştirilmez.

## Çalıştırma

```bash
# Önce küçük doğrulama (önerilir)
bash scripts/run_e019_e023_pipeline.sh --stage dry-run

# Normal uçtan uca çalışma; fallback'leri otomatik çalıştırmaz
bash scripts/run_e019_e023_pipeline.sh --stage all
```

İstenirse aşamalar ayrı çalıştırılabilir:

```bash
bash scripts/run_e019_e023_pipeline.sh --stage train-xgb
bash scripts/run_e019_e023_pipeline.sh --stage train-cat
bash scripts/run_e019_e023_pipeline.sh --stage ensemble
bash scripts/run_e019_e023_pipeline.sh --stage validate
```

`RESUME=1` varsayılandır. Tamamlanmış model artefaktları tekrar üretilmez. Yeniden üretmek
için `FORCE=1` kullanılır. Pipeline varsayılan olarak en az 8 GiB boş disk ister;
`MIN_FREE_GB` ile değiştirilebilir.

## Koşullu fallback

Ana model tuned OOF'u E002'nin 0.0015 altına düşerse veya tuned test disagreement `%0.10`
altında kalırsa `fallback_decision.json` fallback önerir. Fallback otomatik değildir:

```bash
bash scripts/run_e019_e023_pipeline.sh --stage fallback-xgb
bash scripts/run_e019_e023_pipeline.sh --stage fallback-cat
bash scripts/run_e019_e023_pipeline.sh --stage ensemble
```

Karar dosyası fallback gerektirmiyorsa fallback stage hard-fail eder. Ana/fallback arasında
tuned OOF'u daha yüksek kaynak `e019_e020_selected_sources.json` dosyasına yazılır.

## Cache ve artefaktlar

- Raw CSV Parquet cache: `outputs/cache/raw/`
- Fold V2-Core cache: `outputs/cache/folds/`
- Kısa koşular: `outputs/dry_runs/E019`, `outputs/dry_runs/E020`
- Tam modeller: `outputs/experiments/E019`, `outputs/experiments/E020`
- Ensemble'lar: `outputs/experiments/E021`, `E022`, `E023`
- Toplu rapor: `outputs/experiments/e021_e023_eligibility_report.csv`

Her model dizini OOF/test probability, fold probability, fold metric, tuned multiplier,
model dosyası ve run manifest içerir. Ensemble dizinlerinde `metrics.json`,
`risk_summary.json`, `eligibility.json` ve yalnız gate geçerse submission bulunur.

## Submission güvenlik kapıları

- OOF balanced accuracy E002 tuned OOF'tan yüksek olmalı.
- En az 2/3 fold gerilememeli.
- Changed rows E021/E022 için 30–150, E023 için 50–180 olmalı.
- At-risk → minority toplamı en fazla 150 olmalı.
- Minority → at-risk toplamı en fazla 30 olmalı.
- Sınıf başına test distribution sapması en fazla 0.10 yüzde puan olmalı.

Gate geçmeyen aday submission üretmez ve `rejection_reasons.json` yazar.

## Süre

Mevcut E002 ölçümüne göre normal toplam yaklaşık 32–68 dakikadır. XGB fallback yaklaşık
15–30 dakika, CatBoost fallback 20–45 dakika ekleyebilir. İlk dry-run gerçek makine
süresini görmek için kullanılmalıdır.
