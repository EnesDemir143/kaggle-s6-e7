#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_RUN=(uv run --no-sync python)
FORCE="${FORCE:-0}"
ARGS=(
    --config configs/e018_experiments.yaml
    --source-root outputs/experiments
    --output-root outputs/experiments
)
if [[ "$FORCE" == "1" ]]; then
    ARGS+=(--force)
fi

echo "════════════════════════════════════════════════════════════"
echo "E018 SERIES  E002-HEAVY MICRO BLEND EXPERIMENTS"
echo "Force: $FORCE"
echo ""
echo "Adaylar:"
echo "  E018A = 90% E002 + 8% E004 + 2% E006"
echo "  E018B = 88% E002 + 10% E004 + 2% E006"
echo "  E018C = 85% E002 + 12% E004 + 3% E006"
echo "  E018D = 92% E002 + 6% E004 + 2% E006"
echo "  E018E = 95% E002 + 5% E004"
echo ""
echo "Her aday icin 7x7=49 multiplier scale grid taranacak."
echo "════════════════════════════════════════════════════════════"

"${PYTHON_RUN[@]}" scripts/generate_postprocess_experiments.py "${ARGS[@]}"

echo ""
echo "Done. E018 serisi tamamlandi."
echo "Rapor: outputs/experiments/e018_eligibility_report.csv"
echo "Rapor: outputs/experiments/e018_eligibility_report.json"
