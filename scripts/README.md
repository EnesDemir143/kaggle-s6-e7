# Scripts

Tek kalite komutu:

```bash
uv run python scripts/check.py
```

Bu komut pytest, Ruff, Mypy ve `compileall` çalıştırır. E001-E008 komutları ve cache
yönetimi için `docs/experiment-runbook.md` dosyasına bakın.

Tam sıralı çalışma:

```bash
bash scripts/dry_run_pipeline.sh
bash scripts/experiment_runner.sh
```

E002 merkezli E009–E013 postprocess adayları:

```bash
bash scripts/dry_run_postprocess_experiments.sh
bash scripts/run_postprocess_experiments.sh
```

Dry-run yalnız `outputs/dry_runs/postprocess_experiments/` altına yazar ve feature
cache ile production deney dizinlerinin değişmediğini doğrular. Production yeniden
üretimi gerektiğinde `FORCE=1 bash scripts/run_postprocess_experiments.sh` kullanılır.
