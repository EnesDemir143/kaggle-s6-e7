#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/outputs/cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON=("$PYTHON_BIN")
else
  PYTHON=(uv run --no-sync python)
fi
STAGE="${1:-all}"
if [[ "$STAGE" == "--stage" ]]; then
  STAGE="${2:-all}"
fi
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
EXPERIMENT_CONFIG="configs/e019_e020_experiments.yaml"
SCALES=(0.98 0.99 1.00 1.01 1.02)

run_exp() {
  local exp="$1" model_config="$2" extra="${3:-}"
  local root="outputs/experiments"
  [[ "$extra" == "dry" ]] && root="outputs/dry_runs"
  local metrics="$root/$exp/metrics.json"
  if [[ "$RESUME" == "1" && -f "$metrics" && "$FORCE" != "1" ]]; then
    echo "SKIP $exp: completed artefact exists"
    return
  fi
  local args=(scripts/run_experiment.py --exp "$exp" --config "$EXPERIMENT_CONFIG" --model-config "$model_config")
  [[ "$extra" == "dry" ]] && args+=(--dry-run)
  [[ "$FORCE" == "1" ]] && args+=(--force)
  "${PYTHON[@]}" "${args[@]}"
}

tune_exp() {
  local exp="$1" extra="${2:-}"
  local args=(scripts/tune_multipliers.py --exp "$exp" --scales "${SCALES[@]}")
  [[ "$extra" == "dry" ]] && args+=(--dry-run)
  "${PYTHON[@]}" "${args[@]}"
}

require_fallback() {
  local exp="$1"
  "${PYTHON[@]}" - "$exp" <<'PY'
import json, sys
from pathlib import Path
path = Path("outputs/experiments") / sys.argv[1] / "fallback_decision.json"
if not path.is_file() or not json.loads(path.read_text())["fallback_required"]:
    raise SystemExit(f"Fallback blocked: {path} does not require it")
PY
}

preflight() {
  local free_kb
  free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
  if (( free_kb < MIN_FREE_GB * 1024 * 1024 )); then
    echo "At least ${MIN_FREE_GB} GiB free disk is required" >&2
    exit 1
  fi
  "${PYTHON[@]}" -m pytest tests/test_model_training.py tests/test_ensemble.py -q
}

case "$STAGE" in
  preflight)
    preflight ;;
  dry-run)
    preflight
    run_exp E019 configs/xgb_v2_core.yaml dry
    tune_exp E019 dry
    run_exp E020 configs/catboost_v2_core.yaml dry
    tune_exp E020 dry ;;
  train-xgb)
    run_exp E019 configs/xgb_v2_core.yaml
    tune_exp E019
    "${PYTHON[@]}" scripts/assess_diversity_fallback.py --exp E019 ;;
  train-cat)
    run_exp E020 configs/catboost_v2_core.yaml
    tune_exp E020
    "${PYTHON[@]}" scripts/assess_diversity_fallback.py --exp E020 ;;
  fallback-xgb)
    require_fallback E019
    run_exp E019_ALT configs/xgb_v2_core_alt.yaml
    tune_exp E019_ALT
    "${PYTHON[@]}" scripts/select_diversity_sources.py ;;
  fallback-cat)
    require_fallback E020
    run_exp E020_ALT configs/catboost_v2_core_alt.yaml
    tune_exp E020_ALT
    "${PYTHON[@]}" scripts/select_diversity_sources.py ;;
  ensemble)
    "${PYTHON[@]}" scripts/select_diversity_sources.py
    args=(scripts/run_e021_e023_ensembles.py)
    [[ "$FORCE" == "1" ]] && args+=(--force)
    "${PYTHON[@]}" "${args[@]}" ;;
  validate)
    "${PYTHON[@]}" scripts/check.py ;;
  all)
    preflight
    run_exp E019 configs/xgb_v2_core.yaml
    tune_exp E019
    "${PYTHON[@]}" scripts/assess_diversity_fallback.py --exp E019
    run_exp E020 configs/catboost_v2_core.yaml
    tune_exp E020
    "${PYTHON[@]}" scripts/assess_diversity_fallback.py --exp E020
    "${PYTHON[@]}" scripts/select_diversity_sources.py
    "${PYTHON[@]}" scripts/run_e021_e023_ensembles.py
    "${PYTHON[@]}" scripts/check.py ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Use: preflight|dry-run|train-xgb|train-cat|fallback-xgb|fallback-cat|ensemble|validate|all" >&2
    exit 2 ;;
esac
