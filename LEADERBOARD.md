# Leaderboard — Kaggle Playground Series S6E7

**Metric:** Balanced accuracy (multi-class)  
**Best submission:** `E002_tuned` — **0.94960**  

---

## Public Leaderboard (Submitted)

| Rank | Submission | Public LB | Δ Best | Description |
|-----:|:-----------|----------:|:-------|:------------|
| 1 | **E002_tuned** | **0.94960** | — | V2-Core + class multiplier tuning |
| 2 | E011_blend_60_30_10 | 0.94957 | -0.00003 | 60% E002 + 30% E004 + 10% E006 + tuning |
| 2 | **E018C_blend_85_12_03** | **0.94957** | -0.00003 | 85% E002 + 12% E004 + 3% E006 + aggressive multiplier (×1.04) |
| 4 | E009_blend_75_25 | 0.94948 | -0.00012 | 75% E002 + 25% E004 + tuning |
| 5 | E004_tuned | 0.94941 | -0.00019 | V2-Core + rule flags + tuning |
| 6 | E006_tuned | 0.94905 | -0.00055 | V2-Core + log ratios + tuning |
| 7 | E010_SWEEP_002_tuned | 0.94901 | -0.00059 | Best sweep trial + E002-style tuning |
| 8 | E008_tuned | 0.94894 | -0.00066 | V2-Core + sqrt-balanced weights + tuning |
| 9 | E008_argmax | 0.91517 | -0.03443 | V2-Core + sqrt-balanced weights (direct argmax) |

---

## Cross-Validation (OOF) Scores — All Experiments

### E001–E008 — Ablation experiments (LightGBM training)

| Experiment | Bal Acc ± Std | at-risk recall | fit recall | unhealthy recall | Features | Description |
|:-----------|--------------:|:--------------|:-----------|:----------------|:---------|:------------|
| **E008** | **0.9140 ± 0.0011** | 0.975 | **0.887** | **0.879** | 63 | V2-Core + sqrt-balanced weights |
| E001 | 0.8781 ± 0.0015 | 0.991 | 0.832 | 0.811 | 26 | V1 baseline (median imputation) |
| E004 | 0.8775 ± 0.0014 | 0.991 | 0.831 | 0.811 | 71 | V2-Core + rule flags |
| E002 | 0.8769 ± 0.0019 | 0.991 | 0.830 | 0.810 | 63 | V2-Core main |
| E006 | 0.8769 ± 0.0017 | 0.991 | 0.830 | 0.810 | 69 | V2-Core + log ratios |
| E003 | 0.8768 ± 0.0015 | 0.991 | 0.830 | 0.810 | 64 | V2-Core + gender_activity |
| E005 | 0.8767 ± 0.0017 | 0.991 | 0.829 | 0.810 | 63 | V2-Core + clipping |

### E009–E018 — Postprocess experiments (blend / multiplier / correction)

| Experiment | OOF Bal Acc | Δ E002 OOF | LB | Description |
|:-----------|------------:|:-----------|:---|:------------|
| **E018C** | **0.948809** | +0.000225 | **0.94957** | 85% E002 + 12% E004 + 3% E006 + aggressive multiplier (×1.04) |
| **E017** | **0.948658** | +0.000074 | — | Selective margin correction (E004 at-risk→unhealthy, 90 rows) |
| **E014** | **0.948654** | +0.000070 | — | E011 multiplier softened toward E002 (173 rows changed) |
| **E011** | **0.948654** | +0.000069 | **0.94957** | 60/30/10 blend + tuned |
| **E015** | 0.948614 | +0.000030 | — | 70/20/10 blend + tuned (E002-heavy) |
| **E009** | 0.948639 | +0.000055 | **0.94948** | 75/25 blend + tuned |
| **E012** | 0.948540 | -0.000044 | — | Fit-up / unhealthy-down boundary perturbation |
| **E013** | 0.948418 | -0.000166 | — | Consensus correction (E004=E006→override) |
| **E010** | 0.948392 | -0.000192 | **0.94901** | SWEEP_002 + E002-style tuning |
| **E016** | — | — | — | Rare-class micro-multiplier (no candidate passed filters) |

> **Note:** E016 produced no submission — all 5 candidates failed the minimum-disagreement filter.

---

## Sweep Trials — LightGBM Parameter Sweep

20 random trials on E002 V2-Core feature set (`configs/sweeps.yaml`).  
Sorted by OOF balanced accuracy.

| Trial | Bal Acc ± Std | LR | Leaves | Min Child | α | λ | Sub | Col |
|:------|-------------:|:---|:-------|:----------|:--|:--|:----|:----|
| SWEEP_002 | **0.8779 ± 0.0018** | 0.04 | 127 | 200 | 0.0 | 10.0 | 0.85 | 0.95 |
| SWEEP_000 | 0.8777 ± 0.0016 | 0.02 | 127 | 400 | 0.05 | 2.0 | 0.95 | 0.75 |
| SWEEP_003 | 0.8776 ± 0.0012 | 0.03 | 31 | 800 | 0.5 | 5.0 | 0.85 | 1.0 |
| SWEEP_011 | 0.8774 ± 0.0012 | 0.02 | 96 | 400 | 0.5 | 2.0 | 0.75 | 1.0 |
| SWEEP_015 | 0.8774 ± 0.0012 | 0.04 | 63 | 100 | 0.5 | 2.0 | 0.75 | 1.0 |
| SWEEP_007 | 0.8773 ± 0.0019 | 0.05 | 127 | 50 | 0.05 | 2.0 | 0.85 | 0.75 |
| SWEEP_017 | 0.8774 ± 0.0015 | 0.05 | 127 | 50 | 0.0 | 5.0 | 0.85 | 0.85 |
| SWEEP_012 | 0.8773 ± 0.0015 | 0.04 | 127 | 50 | 0.05 | 5.0 | 0.95 | 0.85 |
| SWEEP_001 | 0.8772 ± 0.0016 | 0.04 | 63 | 50 | 0.1 | 10.0 | 0.95 | 1.0 |
| SWEEP_005 | 0.8772 ± 0.0011 | 0.02 | 191 | 800 | 0.05 | 5.0 | 0.75 | 1.0 |
| SWEEP_004 | 0.8771 ± 0.0014 | 0.04 | 96 | 200 | 0.0 | 0.5 | 0.85 | 1.0 |
| SWEEP_016 | 0.8771 ± 0.0013 | 0.05 | 96 | 50 | 0.5 | 1.0 | 0.85 | 0.85 |
| SWEEP_014 | 0.8771 ± 0.0014 | 0.03 | 127 | 400 | 0.0 | 1.0 | 0.95 | 0.85 |
| SWEEP_006 | 0.8770 ± 0.0014 | 0.04 | 63 | 50 | 0.5 | 2.0 | 0.95 | 0.95 |
| SWEEP_013 | 0.8772 ± 0.0017 | 0.03 | 96 | 50 | 0.0 | 0.5 | 0.85 | 0.85 |
| SWEEP_008 | 0.8767 ± 0.0011 | 0.04 | 31 | 400 | 0.1 | 10.0 | 0.95 | 0.85 |
| SWEEP_009 | 0.8767 ± 0.0013 | 0.05 | 96 | 100 | 0.5 | 1.0 | 0.75 | 0.85 |
| SWEEP_018 | 0.8770 ± 0.0013 | 0.03 | 63 | 400 | 0.5 | 10.0 | 0.85 | 0.95 |
| SWEEP_019 | 0.8769 ± 0.0015 | 0.05 | 31 | 50 | 0.5 | 10.0 | 0.85 | 0.95 |
| SWEEP_010 | 0.8762 ± 0.0009 | 0.05 | 31 | 200 | 0.0 | 5.0 | 0.85 | 0.85 |

> Sweep gain over default E002 params: **+0.0009** (SWEEP_002).  
> By contrast, sqrt-balanced weights (E008) added **+0.037** — HPO is not a substitute for proper class weighting.

---

## True Training Distribution

| Class | Train Ratio |
|:------|-----------:|
| at-risk | 85.87% |
| fit | 5.77% |
| unhealthy | 8.36% |

---

## Experiment Summary

| ID | Type | Feature Set | Weights | Tuned | LB Score |
|:---|:-----|:------------|:--------|:------|:---------|
| E001 | Ablation | V1 baseline (median imputation) | — | — | — |
| E002 | Ablation | V2-Core main (ratios, interactions, outliers) | — | ✅ | **0.94960** |
| E003 | Ablation | V2-Core + gender_activity | — | — | — |
| E004 | Ablation | V2-Core + rule flags | — | ✅ | 0.94941 |
| E005 | Ablation | V2-Core + clipping | — | — | — |
| E006 | Ablation | V2-Core + log ratios | — | ✅ | 0.94905 |
| E008 | Ablation | V2-Core | sqrt_balanced | ✅ | 0.94894 |
| E009 | Postprocess | 75% E002 + 25% E004 blend | — | ✅ | 0.94948 |
| E010 | Postprocess | SWEEP_002 + E002-style tuning | — | ✅ | 0.94901 |
| E011 | Postprocess | 60% E002 + 30% E004 + 10% E006 blend | — | ✅ | **0.94957** |
| E012 | Postprocess | E002 fit-up / unhealthy-down boundary | — | — | — |
| E013 | Postprocess | E002 consensus (E004=E006→override) | — | — | — |
| E014 | Postprocess | E011 with softer multiplier | — | — | — |
| E015 | Postprocess | 70% E002 + 20% E004 + 10% E006 blend | — | — | — |
| E016 | Postprocess | Rare-class micro-multiplier (no output) | — | — | — |
| E017 | Postprocess | Selective margin correction (90 rows) | — | — | — |
| E018C | Postprocess | 85% E002 + 12% E004 + 3% E006 blend | — | ✅ | **0.94957** |

---

## Submission Pipeline

```bash
# Train all ablation experiments
bash scripts/experiment_runner.sh

# Run multiplier tuning
uv run python scripts/tune_multipliers.py --exp E002

# Run postprocess pipeline (E009–E017)
bash scripts/run_postprocess_experiments.sh

# Generate submission
uv run python scripts/make_submission.py --exp E002 --postprocess multipliers
```

Submission files: `outputs/experiments/<EXP_ID>/submission.csv` or `submission_<EXP>_tuned.csv`
