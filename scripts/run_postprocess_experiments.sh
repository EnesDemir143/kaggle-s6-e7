#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_RUN=(uv run --no-sync python)
FORCE="${FORCE:-0}"
ARGS=(--source-root outputs/experiments --output-root outputs/experiments)
if [[ "$FORCE" == "1" ]]; then
    ARGS+=(--force)
fi

echo "════════════════════════════════════════════════════════════"
echo "E009-E013 POSTPROCESS PRODUCTION RUN"
echo "Force: $FORCE"
echo "════════════════════════════════════════════════════════════"

"${PYTHON_RUN[@]}" scripts/generate_postprocess_experiments.py "${ARGS[@]}"

echo "✓ Candidate generation complete."
echo "Review PASS/FAIL: outputs/experiments/eligibility_report.csv"
