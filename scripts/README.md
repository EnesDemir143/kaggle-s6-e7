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
