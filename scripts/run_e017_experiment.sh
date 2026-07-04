#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_RUN=(uv run --no-sync python)
FORCE="${FORCE:-0}"
ARGS=(
    --config configs/e017_experiment.yaml
    --source-root outputs/experiments
    --output-root outputs/experiments
)
if [[ "$FORCE" == "1" ]]; then
    ARGS+=(--force)
fi

echo "════════════════════════════════════════════════════════════"
echo "E017 E002 SELECTIVE MARGIN CORRECTION RUN"
echo "Force: $FORCE"
echo "════════════════════════════════════════════════════════════"

"${PYTHON_RUN[@]}" scripts/generate_postprocess_experiments.py "${ARGS[@]}"

SUBMISSION="outputs/experiments/E017_E002_selective_margin_correction/submission.csv"
if [[ -f "$SUBMISSION" ]]; then
    echo "✓ E017 passed all filters: $SUBMISSION"
else
    echo "⚠ No E017 rule passed all filters; no submission was produced."
fi
echo "Report: outputs/experiments/e017_eligibility_report.csv"
