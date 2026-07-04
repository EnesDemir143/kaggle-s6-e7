#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_RUN=(uv run --no-sync python)
FORCE="${FORCE:-0}"
ARGS=(
    --config configs/e016_experiment.yaml
    --source-root outputs/experiments
    --output-root outputs/experiments
)
if [[ "$FORCE" == "1" ]]; then
    ARGS+=(--force)
fi

echo "════════════════════════════════════════════════════════════"
echo "E016 E002-ONLY RARE-CLASS MICRO-MULTIPLIER RUN"
echo "Force: $FORCE"
echo "════════════════════════════════════════════════════════════"

"${PYTHON_RUN[@]}" scripts/generate_postprocess_experiments.py "${ARGS[@]}"

SUBMISSION="outputs/experiments/E016_E002_rare_class_tiny_boost/submission.csv"
if [[ -f "$SUBMISSION" ]]; then
    echo "✓ E016 passed all filters: $SUBMISSION"
else
    echo "⚠ No E016 candidate passed all filters; no submission was produced. Keep E002 final."
fi
echo "Report: outputs/experiments/e016_eligibility_report.csv"
