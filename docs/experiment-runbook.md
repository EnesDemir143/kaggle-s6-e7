# LightGBM Experiment Runbook

E001–E008 hipotezleri, tüm artefaktlar, sweep sonrası seçim ve submission bütçesi için
ayrıntılı rehber: [`experiment-execution-guide.md`](experiment-execution-guide.md).

## Kurulum ve doğrulama

```bash
uv sync --dev
uv run python scripts/validate_experiments.py
uv run python scripts/check.py
```

Önerilen kullanım shell orkestratörleridir:

```bash
# Önce bütün adımları küçük modellerle doğrula
bash scripts/dry_run_pipeline.sh

# Feature ablation -> multiplier -> submission -> sweep
bash scripts/experiment_runner.sh

# Sweep'in hangi başarılı feature pipeline üzerinde çalışacağını açıkça seç
BASE_EXP=E004 N_TRIALS=20 bash scripts/experiment_runner.sh
```

`experiment_runner.sh` idempotenttir: tamamlanmış artefaktı yeniden üretmez. Bir training
adımı başarısız olursa o deneye bağlı multiplier ve submission adımları çalıştırılmaz.
Varsayılan sweep base `E002`'dir; E001-E008 sonuçları incelendikten sonra `BASE_EXP`
değişkeniyle değiştirilmelidir. Sweep'i tamamen kapatmak için `RUN_SWEEP=0` kullanılır.
Runner varsayılan olarak en az 8 GiB boş disk ister; eşik `MIN_FREE_GB` ile değiştirilebilir.
Apple M2 Pro üzerinde LightGBM MPS kullanmaz; ölçülmüş en hızlı yerel ayar altı performance
core için `n_jobs=6` ve `OMP_NUM_THREADS=6` değerleridir.

Dry-run sonuçları `outputs/dry_runs/`, gerçek deneyler `outputs/experiments/`, tekrar
kullanılabilir fold feature cache dosyaları `outputs/cache/` altında tutulur. Cache'i
atlamak için `--no-cache`, tamamlanmış çıktıyı yeniden üretmek için `--force` kullanın.

## Feature ablation deneyleri

```bash
uv run python scripts/run_experiment.py --exp E001 --config configs/experiments.yaml
uv run python scripts/run_experiment.py --exp E002 --config configs/experiments.yaml
uv run python scripts/run_experiment.py --exp E003 --config configs/experiments.yaml
uv run python scripts/run_experiment.py --exp E004 --config configs/experiments.yaml
uv run python scripts/run_experiment.py --exp E005 --config configs/experiments.yaml
uv run python scripts/run_experiment.py --exp E006 --config configs/experiments.yaml
uv run python scripts/run_experiment.py --exp E008 --config configs/experiments.yaml
```

Her komuta `--dry-run --force` ekleyerek 3,000 satırlık doğrulama koşusu yapılabilir.
E007 model eğitimi değildir; aşağıdaki multiplier tuning adımıdır.

## Multiplier tuning ve submission

```bash
uv run python scripts/tune_multipliers.py --exp E002
uv run python scripts/make_submission.py --exp E002 --postprocess argmax
uv run python scripts/make_submission.py --exp E002 --postprocess multipliers

uv run python scripts/tune_multipliers.py --exp E004
uv run python scripts/tune_multipliers.py --exp E006
uv run python scripts/tune_multipliers.py --exp E008
```

Tuning yalnız `oof_proba.npy` ve OOF gerçek etiketlerini kullanır. Test üzerinde tuning
yapılmaz. Üretilen dosyalar ilgili experiment dizinindeki `submission_tuned.csv` ve
`submission_<EXP>_tuned.csv` dosyalarıdır.

## Karşılaştırma ve sweep

```bash
uv run python scripts/compare_experiments.py --output outputs/leaderboard_local.csv
uv run python scripts/run_sweep.py --base-exp E002 --sweep configs/sweeps.yaml --n-trials 20
```

Sweep yalnız feature ablation bittikten sonra çalıştırılmalıdır. Başlangıç aşamasında HPO
yoktur; `configs/lgbm_base.yaml` bütün E001-E008 modellerinin sabit parametre kaynağıdır.

## Önerilen sıra

1. E001 ve E002.
2. E002 belirgin biçimde iyiyse E003-E006.
3. E002/E004/E006 üzerinde multiplier tuning.
4. E008 ve ardından E008 multiplier tuning.
5. En iyi preprocessing üzerinde sweep.

`+0.003` güçlü sinyal, `+0.001–0.003` ek seed/CV5 doğrulama adayı, `<+0.001`
muhtemel gürültü kabul edilir. Seçimde balanced accuracy yanında fold std, üç sınıfın
recall değerleri ve prediction distribution birlikte değerlendirilir.

## Cache yönetimi

- Cache anahtarı veri SHA-256 fingerprint'i, fold train indeksleri, preprocessing config'i
  ve cache schema version'dan oluşur.
- Feature config değişirse otomatik yeni cache oluşur.
- Şüpheli veya eski cache için `rm -rf outputs/cache` uygulanabilir; ham veri ve kaynak
  kod etkilenmez.
- `--no-cache` yalnız teşhis içindir; normal koşularda cache açık bırakılmalıdır.
