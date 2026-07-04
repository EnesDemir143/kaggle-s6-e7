#!/usr/bin/env bash
set -euo pipefail

# Bounded pre-flight for every production pipeline stage.
# It trains only tiny 20-tree models and never writes real experiment outputs.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

echo "════════════════════════════════════════════════════════════"
echo "KAGGLE S6E7 PIPELINE DRY RUN"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

uv run --no-sync python scripts/validate_experiments.py
uv run --no-sync python scripts/check.py

echo ""
echo "✓ Dry-run complete. Production command:"
echo "  bash scripts/experiment_runner.sh"
echo ""
echo "Optional explicit sweep base:"
echo "  BASE_EXP=E004 N_TRIALS=20 bash scripts/experiment_runner.sh"
