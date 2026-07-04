---
goal: Leakage-safe fold feature cache
version: 1.0
date_created: 2026-07-04
last_updated: 2026-07-04
owner: repository maintainer
status: 'Completed'
tags: [data, cache]
---
# Introduction
![Status: Completed](https://img.shields.io/badge/status-Completed-brightgreen)

Cache expensive fold transforms without crossing validation boundaries.

## 1. Requirements & Constraints
- **REQ-001**: Key cache by data, fold indices, config and schema version.
- **CON-001**: Never fit cached transformations on validation or test rows.

## 2. Implementation Steps
### Implementation Phase 1
- GOAL-001: Implement atomic Parquet cache.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Add fingerprints and stable keys. | ✅ | 2026-07-04 |
| TASK-002 | Add manifest validation and atomic writes. | ✅ | 2026-07-04 |
| TASK-003 | Verify invalidation tests. | ✅ | 2026-07-04 |

## 3. Alternatives
- **ALT-001**: Joblib frames rejected due to slower large-frame interoperability.

## 4. Dependencies
- **DEP-001**: PyArrow Parquet engine.

## 5. Files
- **FILE-001**: `src/kaggle_s6_e7/cache.py`.

## 6. Testing
- **TEST-001**: Cache hit, miss, corruption and invalidation.

## 7. Risks & Assumptions
- **RISK-001**: Cache may consume several GB during full experiments.
- **ASSUMPTION-001**: Outputs are disposable and gitignored.

## 8. Related Specifications / Further Reading
- `docs/experiment-runbook.md`
