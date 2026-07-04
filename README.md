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
| 🔸 **E002_tuned** (V2-Core + multiplier tuning) | **0.9486** | **0.94960** |
| 🔸 E011_blend (60% E002 + 30% E004 + 10% E006) | 0.9487 | 0.94957 |
| 🔸 E009_blend (75% E002 + 25% E004) | 0.9486 | 0.94948 |
| 🔸 E004_tuned (V2-Core + rule flags + tuning) | 0.9482 | 0.94941 |
| 🔸 E006_tuned (V2-Core + log ratios + tuning) | 0.9483 | 0.94905 |
| 🔸 E010_SWEEP_002_tuned | 0.9484 | 0.94901 |
| 🔸 E008_tuned (V2-Core + sqrt-weights + tuning) | 0.9478 | 0.94894 |
| 🔹 E008_argmax (V2-Core + sqrt-balanced weights) | **0.9140** | 0.91517 |

Detailed breakdown in [`LEADERBOARD.md`](LEADERBOARD.md).

---

## Repository Structure

```
├── configs/
│   ├── experiments.yaml              ← E001–E008 feature definitions
│   ├── lgbm_base.yaml                ← LightGBM hyperparameters
│   ├── sweeps.yaml                   ← Sweep search space
│   ├── postprocess_experiments.yaml  ← E009–E013 configs
│   ├── e014_e015_experiments.yaml    ← E014–E015 blend configs
│   ├── e016_experiment.yaml          ← E016 micro-multiplier config
│   └── e017_experiment.yaml          ← E017 margin correction config
│
├── src/kaggle_s6_e7/                 ← Core Python library
│   ├── cache.py                      ← Content-addressed Parquet cache
│   ├── candidate_experiments.py      ← Postprocess experiment engine
│   ├── config.py                     ← Paths & constants
│   ├── data.py                       ← Schema validation
│   ├── ensemble.py                   ← Blend, multiplier search, disagreement
│   ├── evaluation.py                 ← Multi-class metrics + ROC plots
│   ├── experiment_config.py          ← YAML loading + config inheritance
│   ├── features.py                   ← Feature engineering functions
│   ├── postprocess.py                ← Class multiplier tuning
│   └── training.py                   ← 3-fold CV experiment runner
│
├── scripts/
│   ├── run_experiment.py             ← Train one ablation experiment
│   ├── run_sweep.py                  ← LightGBM parameter sweep
│   ├── tune_multipliers.py           ← OOF class multiplier search
│   ├── make_submission.py            ← Generate submission CSV
│   ├── compare_experiments.py        ← Build local leaderboard
│   ├── experiment_runner.sh          ← Full ablation + sweep pipeline
│   ├── dry_run_pipeline.sh           ← Small-data validation
│   ├── run_postprocess_experiments.sh ← E009–E013 pipeline
│   ├── dry_run_postprocess_experiments.sh
│   ├── generate_postprocess_experiments.py
│   ├── run_e014_e015_experiments.sh
│   ├── run_e016_experiment.sh
│   ├── run_e017_experiment.sh
│   ├── generate_experiment_report.py ← PDF report generation
│   └── check.py                      ← Quality gate suite
│
├── docs/
│   ├── experiment-runbook.md                ← Quick command reference
│   ├── experiment-execution-guide.md         ← Hypothesis & submission rules
│   ├── experiments-detailed-explanation.md   ← All experiments explained
│   ├── final-experiment-report.md            ← Final PDF report source
│   ├── e009-e013-postprocess-pipeline-results.md
│   ├── e014-e015-micro-blend-experiments.md
│   ├── e016-e002-rare-class-micro-multiplier-results.md
│   ├── e017-selective-margin-correction-results.md
│   └── rules/
│
├── tests/                   ← pytest suite
├── LEADERBOARD.md           ← Full CV + public LB scores
├── LICENSE                  ← MIT
└── README.md
```

---

## Experiment Catalogue

### E001–E008 — Feature Ablation (LightGBM)

| ID | Feature Set | Sample Weight | Postprocess | Best LB |
|:---|:------------|:--------------|:------------|:--------|
| E001 | V1 baseline (median + missing flags) | — | — | — |
| E002 | V2-Core (ratios, interactions, outlier flags) | — | ✅ multiplier | **0.94960** |
| E003 | V2-Core + gender×activity interaction | — | — | — |
| E004 | V2-Core + BMI/sleep/HR/step rule flags | — | ✅ multiplier | 0.94941 |
| E005 | V2-Core + clip 0.1%/99.9% | — | — | — |
| E006 | V2-Core + log1p ratio variants | — | ✅ multiplier | 0.94905 |
| E008 | V2-Core (same features as E002) | `sqrt_balanced` | ✅ multiplier | 0.94894 |

Key finding: **E008 (sample weights)** outperformed all other feature additions by **+0.037** in CV.

### E009–E017 — Postprocess (no new training)

| ID | Method | OOF | LB |
|:---|:-------|:----|:---|
| E009 | 75% E002 + 25% E004 blend + multiplier | 0.948639 | 0.94948 |
| E010 | SWEEP_002 + E002 multiplier style | 0.948392 | 0.94901 |
| E011 | **60% E002 + 30% E004 + 10% E006 blend + multiplier** | **0.948654** | **0.94957** |
| E012 | E002 fit-up / unhealthy-down boundary perturbation | 0.948540 | — |
| E013 | Consensus: pick E004 label only when E004=E006≠E002 | 0.948418 | — |
| E014 | E011 with softer multiplier (closer to E002) | 0.948654 | — |
| E015 | 70% E002 + 20% E004 + 10% E006 blend | 0.948614 | — |
| E016 | Rare-class micro-multiplier (±0.25%–0.75%) | —¹ | — |
| E017 | Selective margin correction (E004→unhealthy where confident) | 0.948658 | — |

¹E016 produced no submission — all 5 candidates failed the disagreement filter.

---

## Key Design Decisions

- **Stratified 3-fold CV** with fixed seed 42 — deterministic splits
- **Feature statistics** learned only on fold-train (no leakage)
- **Content-addressed Parquet cache** — automatic invalidation on data/config change
- **OOF-only class multiplier tuning** — never touches test labels
- **Deterministic LightGBM** — `random_state=42`, `deterministic=true`, all seeds pinned
- **Eligibility filters** on postprocess candidates — prevent near-identical submissions
- **Per-fold extended artifacts** — per-fold test probs, OOF probs, training loss history, ROC curves

---

## Pipeline

```bash
# Setup
uv sync --dev

# Full ablation + sweep
bash scripts/experiment_runner.sh

# Postprocess pipeline (blends, corrections)
bash scripts/run_postprocess_experiments.sh

# Individual steps
uv run python scripts/run_experiment.py --exp E002
uv run python scripts/tune_multipliers.py --exp E002
uv run python scripts/make_submission.py --exp E002 --postprocess multipliers
uv run python scripts/compare_experiments.py --output outputs/leaderboard_local.csv

# Quality gates
uv run python scripts/check.py
```

---

## Submission Files

All generated submissions under `outputs/experiments/`.

### E001–E008 — Training experiments

| Experiment | Files | Submitted (LB) |
|:-----------|:------|:---------------|
| E001 | `submission_argmax.csv` | — |
| E002 | `submission_argmax.csv`, `submission_tuned.csv`, `submission_E002_argmax.csv`, `submission_E002_tuned.csv` | **tuned: 0.94960** |
| E003 | `submission_argmax.csv` | — |
| E004 | `submission_argmax.csv`, `submission_tuned.csv`, `submission_E004_argmax.csv`, `submission_E004_tuned.csv` | tuned: 0.94941 |
| E005 | `submission_argmax.csv` | — |
| E006 | `submission_argmax.csv`, `submission_tuned.csv`, `submission_E006_argmax.csv`, `submission_E006_tuned.csv` | tuned: 0.94905 |
| E008 | `submission_argmax.csv`, `submission_tuned.csv`, `submission_E008_argmax.csv`, `submission_E008_tuned.csv` | argmax: 0.91517, tuned: 0.94894 |

### E009–E017 — Postprocess experiments

| Experiment | File | Submitted (LB) |
|:-----------|:-----|:---------------|
| E009 (75/25 blend) | `submission_tuned.csv` | 0.94948 |
| E010 (SWEEP_002 tuned) | `submission_tuned.csv` | 0.94901 |
| E011 (60/30/10 blend) | `submission_tuned.csv` | **0.94957** |
| E012 (fit-up / unhealthy-down) | `submission.csv` | — |
| E013 (consensus correction) | `submission.csv` | — |
| E014 (softer multiplier) | `submission.csv` | — |
| E015 (70/20/10 blend) | `submission.csv` | — |
| E016 (micro-multiplier) | *(no submission — all candidates filtered out)* | — |
| E017 (margin correction) | `submission.csv` | — |

### Sweep trials (SWEEP_000–SWEEP_019)

Each sweep trial produces `submission_argmax.csv`.  
Best sweep on LB: **E010** (SWEEP_002 tuned) → **0.94901**.

---

## License

MIT License — see [`LICENSE`](LICENSE).

This repository contains code for the [Kaggle Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) competition.  
The dataset is synthetically generated — see [competition rules](https://www.kaggle.com/competitions/playground-series-s6e7/rules) for data usage terms.
