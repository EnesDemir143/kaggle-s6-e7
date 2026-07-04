# Kaggle S6E7 — Predicting Student Health Risk

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E7-blue)](https://www.kaggle.com/competitions/playground-series-s6e7)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Leaderboard:** [`LEADERBOARD.md`](LEADERBOARD.md) — full CV and public LB scores.  
**Best submission:** `E002_tuned` — **0.94960** balanced accuracy.

---

## Competition Overview

[Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7/overview) — predict student health risk (`at-risk`, `fit`, `unhealthy`) from lifestyle metrics. Synthetic dataset generated from real-world survey data.

- **Metric:** Balanced accuracy (multi-class)
- **Train:** 690,088 rows × 13 features (7 numeric, 6 categorical)
- **Test:** 295,753 rows
- **Deadline:** July 31, 2026

---

## Results Summary

| Approach | CV (OOF) | Public LB |
|:---------|---------:|----------:|
| 🔹 V2-Core + sqrt-balanced weights (E008) | 0.9140 ± 0.0011 | 0.91517 |
| 🔸 **V2-Core + class multiplier tuning (E002_tuned)** | **0.9486** | **0.94960** |
| 🔸 V2-Core + rule flags + tuning (E004_tuned) | 0.9482 | 0.94941 |
| 🔸 V2-Core + log ratios + tuning (E006_tuned) | 0.9483 | 0.94905 |
| 🔸 V2-Core + sqrt-balanced + tuning (E008_tuned) | 0.9478 | 0.94894 |

Detailed breakdown in [`LEADERBOARD.md`](LEADERBOARD.md).

---

## Repository Structure

```
├── configs/                  ← Experiment YAML configs
│   ├── experiments.yaml      ← E001–E008 feature definitions
│   ├── lgbm_base.yaml        ← LightGBM hyperparameters
│   └── sweeps.yaml           ← Sweep search space
│
├── src/kaggle_s6_e7/         ← Core Python library
│   ├── cache.py              ← Content-addressed Parquet cache
│   ├── config.py             ← Paths & constants
│   ├── data.py               ← Schema validation
│   ├── evaluation.py         ← Multi-class metrics + ROC plots
│   ├── experiment_config.py  ← YAML loading + config inheritance
│   ├── features.py           ← Feature engineering functions
│   ├── postprocess.py        ← Class multiplier tuning
│   └── training.py           ← 3-fold CV experiment runner
│
├── scripts/                  ← Pipeline entry points
│   ├── run_experiment.py     ← Train one experiment
│   ├── run_sweep.py          ← LightGBM parameter sweep
│   ├── tune_multipliers.py   ← OOF class multiplier search
│   ├── make_submission.py    ← Generate submission CSV
│   ├── compare_experiments.py ← Build local leaderboard
│   ├── experiment_runner.sh  ← Full orchestration pipeline
│   ├── dry_run_pipeline.sh   ← Small-data validation
│   └── check.py              ← Quality gate suite
│
├── docs/
│   ├── experiment-runbook.md       ← Quick command reference
│   └── experiment-execution-guide.md ← Hypothesis & submission rules
│
├── outputs/                  ← All generated artifacts (gitignored)
│   ├── experiments/          ← Per-experiment: metrics, models, submissions
│   ├── cache/                ← Fold feature Parquet cache
│   └── logs/                 ← Pipeline logs
│
├── tests/                    ← pytest suite (17 tests)
├── LEADERBOARD.md            ← Full CV and public LB scores
└── README.md                 ← This file
```

---

## Pipeline

### Quick start

```bash
# Setup
uv sync --dev

# Full pipeline (training → leaderboard → multiplier tuning → sweep)
bash scripts/experiment_runner.sh

# Dry-run (small data, seconds)
bash scripts/dry_run_pipeline.sh
```

### Step by step

```bash
# Train all 7 ablation experiments
uv run python scripts/run_experiment.py --exp E002

# OOF multiplier tuning
uv run python scripts/tune_multipliers.py --exp E002

# Generate submissions
uv run python scripts/make_submission.py --exp E002 --postprocess argmax
uv run python scripts/make_submission.py --exp E002 --postprocess multipliers

# Parameter sweep on chosen base experiment
uv run python scripts/run_sweep.py --base-exp E002 --n-trials 20

# Build local leaderboard
uv run python scripts/compare_experiments.py --output outputs/leaderboard_local.csv
```

### Quality gates

```bash
uv run python scripts/check.py
# Runs: pytest → ruff → mypy → compileall
```

---

## Experiments

| ID | Description | Feature Set | Sample Weights | Multiplier Tuned |
|:---|:------------|:------------|:---------------|:----------------:|
| E001 | V1 baseline | Median imputation + missing flags | — | No |
| E002 | V2-Core main | Ratios, interactions, outlier flags | — | **Yes** |
| E003 | V2-Core + gender interaction | + gender×activity | — | No |
| E004 | V2-Core + rule flags | + sleep/BMI/HR/steps thresholds | — | **Yes** |
| E005 | V2-Core + clipping | Clip 0.1%/99.9% | — | No |
| E006 | V2-Core + log ratios | + log1p ratio variants | — | **Yes** |
| E008 | V2-Core + balanced weights | Same as E002 | `sqrt_balanced` | **Yes** |

Key finding: **Sample weights (E008)** outperform all other feature additions by **+0.037** in CV.  
Class multiplier tuning raises all models to **~0.948–0.949** LB regardless of base performance.

---

## Key Design Decisions

- **Stratified 3-fold CV** with fixed seed 42
- **Feature statistics** learned only on fold-train (no data leakage)
- **Content-addressed Parquet cache** for fold features (automatic cache invalidation)
- **OOF-only class multiplier tuning** — never touches test labels
- **Deterministic LightGBM** (`random_state=42`, `deterministic=true`, all seeds pinned)
- **Per-fold extended artifacts** — per-fold test probs, OOF probs, loss history, ROC curves

---

## Submission Files

Generated under `outputs/experiments/<EXP>/`:

| File | Description |
|:-----|:------------|
| `submission_argmax.csv` | Direct argmax prediction |
| `submission_tuned.csv` | Postprocessed with OOF-tuned class multipliers |
| `submission_<EXP>_<type>.csv` | Copy in project root with descriptive name |

---

## License

MIT License — see [`LICENSE`](LICENSE).

This repository contains code for the [Kaggle Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) competition.  
The dataset is synthetically generated — see [competition rules](https://www.kaggle.com/competitions/playground-series-s6e7/rules) for data usage terms.
