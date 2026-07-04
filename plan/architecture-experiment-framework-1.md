---
goal: Reproducible LightGBM experiment framework
version: 1.0
date_created: 2026-07-04
last_updated: 2026-07-04
owner: repository maintainer
status: 'Completed'
tags: [architecture, experiment]
---
# Introduction
![Status: Completed](https://img.shields.io/badge/status-Completed-brightgreen)

Build the common configuration, training, metric, and artifact contracts.

## 1. Requirements & Constraints
- **REQ-001**: Run E001-E008 from YAML configuration.
- **CON-001**: Learn all preprocessing state from fold-train only.
- **GUD-001**: Preserve fixed class order.

## 2. Implementation Steps
### Implementation Phase 1
- GOAL-001: Implement shared framework.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Add configs and inheritance loader. | ✅ | 2026-07-04 |
| TASK-002 | Add CV trainer and artifacts. | ✅ | 2026-07-04 |
| TASK-003 | Validate dry-runs. | ✅ | 2026-07-04 |

## 3. Alternatives
- **ALT-001**: Notebook execution rejected because experiment lineage must be reproducible.

## 4. Dependencies
- **DEP-001**: LightGBM 4.x, pandas, scikit-learn, PyYAML.

## 5. Files
- **FILE-001**: `src/kaggle_s6_e7/training.py`.
- **FILE-002**: `configs/experiments.yaml`.

## 6. Testing
- **TEST-001**: Config and complete artifact contract tests.

## 7. Risks & Assumptions
- **RISK-001**: Full CV is intentionally not run during implementation.
- **ASSUMPTION-001**: CPU training is the target runtime.

## 8. Related Specifications / Further Reading
- `docs/experiment-runbook.md`
