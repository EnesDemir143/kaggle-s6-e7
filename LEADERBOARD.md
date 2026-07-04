# Leaderboard — Kaggle Playground Series S6E7

## Public Leaderboard Scores

| Rank | Submission | Public LB | Notes |
|-----:|:-----------|----------:|:------|
| 1 | **E002_tuned** | **0.94960** | V2-Core + multiplier tuning |
| 2 | E004_tuned | 0.94941 | V2-Core + rule flags + multiplier tuning |
| 3 | E006_tuned | 0.94905 | V2-Core + log ratios + multiplier tuning |
| 4 | E008_tuned | 0.94894 | V2-Core + sqrt-balanced weights + multiplier tuning |
| 5 | E008_argmax | 0.91517 | V2-Core + sqrt-balanced weights (direct argmax) |

**Best submission:** `E002_tuned` — 0.94960 balanced accuracy.

---

## Cross-Validation (OOF) Scores

### Raw argmax (no postprocessing)

| Experiment | Bal Acc ± Std | at-risk recall | fit recall | unhealthy recall | Features | Description |
|:-----------|--------------:|:--------------|:-----------|:----------------|:---------|:------------|
| **E008** | **0.9140 ± 0.0011** | 0.975 | **0.887** | **0.879** | 63 | V2-Core + sqrt-balanced weights |
| E001 | 0.8781 ± 0.0015 | 0.991 | 0.832 | 0.811 | 26 | V1 baseline (median imputation) |
| E004 | 0.8775 ± 0.0014 | 0.991 | 0.831 | 0.811 | 71 | V2-Core + rule flags |
| E002 | 0.8769 ± 0.0019 | 0.991 | 0.830 | 0.810 | 63 | V2-Core main |
| E006 | 0.8769 ± 0.0017 | 0.991 | 0.830 | 0.810 | 69 | V2-Core + log ratios |
| E003 | 0.8768 ± 0.0015 | 0.991 | 0.830 | 0.810 | 64 | V2-Core + gender_activity |
| E005 | 0.8767 ± 0.0017 | 0.991 | 0.829 | 0.810 | 63 | V2-Core + clipping |

### Tuned (OOF class multiplier optimization)

| Experiment | Tuned Bal Acc | at-risk recall | fit recall | unhealthy recall | Description |
|:-----------|--------------:|:--------------|:-----------|:----------------|:------------|
| **E002_tuned** | **0.9486** | — | — | — | V2-Core + multiplier tuning |
| E006_tuned | 0.9483 | — | — | — | V2-Core + log ratios + multiplier tuning |
| E004_tuned | 0.9482 | — | — | — | V2-Core + rule flags + multiplier tuning |
| E008_tuned | 0.9478 | 0.941 | 0.943 | 0.959 | V2-Core + sqrt-balanced weights + multiplier tuning |

> **Note:** Tuned scores are computed on the same OOF predictions used for multiplier selection, so they carry an overfit risk. Public LB showed they generalise well (E002_tuned: 0.9486 OOF → 0.94960 LB).

---

## Sweep Trials

LightGBM random parameter sweep over E002 V2-Core feature set (20 trials).

| Trial | Bal Acc ± Std | Learning Rate | Num Leaves | Min Child | Reg Alpha | Reg Lambda | Subs ample | Col sample |
|:------|--------------:|:-------------|:-----------|:----------|:----------|:-----------|:----------|:-----------|
| SWEEP_002 | 0.8779 ± 0.0018 | 0.04 | 127 | 200 | 0.0 | 10.0 | 0.85 | 0.95 |
| SWEEP_000 | 0.8777 ± 0.0016 | 0.02 | 127 | 400 | 0.05 | 2.0 | 0.95 | 0.75 |
| SWEEP_003 | 0.8776 ± 0.0012 | 0.03 | 31 | 800 | 0.5 | 5.0 | 0.85 | 1.0 |
| SWEEP_007 | 0.8773 ± 0.0019 | 0.05 | 127 | 50 | 0.05 | 2.0 | 0.85 | 0.75 |
| SWEEP_001 | 0.8772 ± 0.0016 | 0.04 | 63 | 50 | 0.1 | 10.0 | 0.95 | 1.0 |
| SWEEP_005 | 0.8772 ± 0.0011 | 0.02 | 191 | 800 | 0.05 | 5.0 | 0.75 | 1.0 |
| SWEEP_004 | 0.8771 ± 0.0014 | 0.04 | 96 | 200 | 0.0 | 0.5 | 0.85 | 1.0 |
| SWEEP_006 | 0.8770 ± 0.0014 | 0.04 | 63 | 50 | 0.5 | 2.0 | 0.95 | 0.95 |
| SWEEP_008 | 0.8767 ± 0.0011 | 0.04 | 31 | 400 | 0.1 | 10.0 | 0.95 | 0.85 |
| SWEEP_009 | 0.8767 ± 0.0013 | 0.05 | 96 | 100 | 0.5 | 1.0 | 0.75 | 0.85 |
| SWEEP_010 | 0.8762 ± 0.0009 | 0.05 | 31 | 200 | 0.0 | 5.0 | 0.85 | 0.85 |
| SWEEP_011 | 0.8774 ± 0.0012 | 0.02 | 96 | 400 | 0.5 | 2.0 | 0.75 | 1.0 |

> Sweep trials did not significantly improve over the default E002 params (0.8769). The largest gain was +0.0009 (SWEEP_002). Sample weights (E008) provided +0.037 over E002 — far above any HPO gain.

---

## True Training Distribution

| Class | Train Ratio |
|:------|-----------:|
| at-risk | 85.87% |
| fit | 5.77% |
| unhealthy | 8.36% |

---

## Submission Pipeline

```bash
# Train all ablation experiments
bash scripts/experiment_runner.sh

# Run multiplier tuning + submissions for selected experiment
uv run python scripts/tune_multipliers.py --exp E002
uv run python scripts/make_submission.py --exp E002 --postprocess multipliers

# Full pipeline reference: docs/experiment-runbook.md
```

Output path: `outputs/experiments/<EXP_ID>/submission_<EXP>_<type>.csv`
