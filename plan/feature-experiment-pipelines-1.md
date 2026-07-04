---
goal: Implement E001-E008 experiment behavior
version: 1.0
date_created: 2026-07-04
last_updated: 2026-07-04
owner: repository maintainer
status: 'Completed'
tags: [feature, modeling]
---
# Introduction
![Status: Completed](https://img.shields.io/badge/status-Completed-brightgreen)

Implement isolated preprocessing ablations and OOF postprocessing.

## 1. Requirements & Constraints
- **REQ-001**: Each experiment changes only its declared ablation.
- **CON-001**: E007 must not retrain a model.

## 2. Implementation Steps
### Implementation Phase 1
- GOAL-001: Implement feature and postprocess variants.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Implement configurable fold preprocessor. | ✅ | 2026-07-04 |
| TASK-002 | Implement class weights and multipliers. | ✅ | 2026-07-04 |
| TASK-003 | Dry-run E001-E008. | ✅ | 2026-07-04 |

## 3. Alternatives
- **ALT-001**: Global preprocessing rejected as leakage.

## 4. Dependencies
- **DEP-001**: Existing feature and evaluation modules.

## 5. Files
- **FILE-001**: `src/kaggle_s6_e7/preprocessing.py`.
- **FILE-002**: `src/kaggle_s6_e7/postprocess.py`.

## 6. Testing
- **TEST-001**: Feature-toggle and fold-statistic tests.

## 7. Risks & Assumptions
- **RISK-001**: Dry-run metrics are not model-quality evidence.
- **ASSUMPTION-001**: E002 remains the initial multiplier source.

## 8. Related Specifications / Further Reading
- `reports/eda_findings.md`
