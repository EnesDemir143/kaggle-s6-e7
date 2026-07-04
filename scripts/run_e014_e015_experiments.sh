#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_RUN=(uv run --no-sync python)
FORCE="${FORCE:-0}"
ARGS=(
    --config configs/e014_e015_experiments.yaml
    --source-root outputs/experiments
    --output-root outputs/experiments
)
if [[ "$FORCE" == "1" ]]; then
    ARGS+=(--force)
fi

echo "════════════════════════════════════════════════════════════"
echo "E014-E015 E002-CENTERED MICRO-EXPERIMENT RUN"
echo "Force: $FORCE"
echo "════════════════════════════════════════════════════════════"

"${PYTHON_RUN[@]}" scripts/generate_postprocess_experiments.py "${ARGS[@]}"

echo "✓ E014 and E015 generation complete."
echo "Report: outputs/experiments/e014_e015_eligibility_report.csv"
