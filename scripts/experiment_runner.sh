#!/usr/bin/env bash
set -euo pipefail

# Full idempotent pipeline:
#   feature experiments -> local leaderboard -> multiplier tuning/submissions
#   -> LightGBM sweep on the explicitly selected base experiment
#
# Usage:
#   bash scripts/experiment_runner.sh
#   BASE_EXP=E004 N_TRIALS=20 bash scripts/experiment_runner.sh
#   RUN_SWEEP=0 bash scripts/experiment_runner.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_RUN=(uv run --no-sync python)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
EXPERIMENTS=(E001 E002 E003 E004 E005 E006 E008)
POSTPROCESS_EXPERIMENTS=(E002 E004 E006 E008)
BASE_EXP="${BASE_EXP:-E002}"
N_TRIALS="${N_TRIALS:-20}"
RUN_SWEEP="${RUN_SWEEP:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"

LOG_DIR="$PROJECT_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo -e "$1" | tee -a "$PIPELINE_LOG"; }

run_step() {
    local label="$1"
    shift
    log ""
    log "════════════════════════════════════════════════════════════"
    log "[$(timestamp)] $label"
    log "════════════════════════════════════════════════════════════"
    set +e
    "$@" 2>&1 | tee -a "$PIPELINE_LOG"
    local exit_code=${PIPESTATUS[0]}
    set -e
    if [[ $exit_code -ne 0 ]]; then
        log "✗ $label FAILED (exit=$exit_code)"
        return "$exit_code"
    fi
    log "✓ $label"
}

train_experiment() {
    local exp="$1"
    local metrics="outputs/experiments/$exp/metrics.json"
    if [[ -f "$metrics" ]]; then
        log "[skip] $exp training already complete: $metrics"
        return 0
    fi
    run_step "$exp — 3-fold LightGBM training" \
        "${PYTHON_RUN[@]}" scripts/run_experiment.py --exp "$exp"
}

postprocess_experiment() {
    local exp="$1"
    local root="outputs/experiments/$exp"
    if [[ ! -f "$root/oof_proba.npy" || ! -f "$root/test_proba.npy" ]]; then
        log "✗ $exp probabilities missing; multiplier/submission skipped"
        return 1
    fi

    if [[ ! -f "$root/best_multipliers.json" ]]; then
        run_step "$exp — OOF multiplier tuning" \
            "${PYTHON_RUN[@]}" scripts/tune_multipliers.py --exp "$exp"
    else
        log "[skip] $exp multipliers already complete"
    fi

    if [[ ! -f "$root/submission_${exp}_argmax.csv" ]]; then
        run_step "$exp — argmax submission" \
            "${PYTHON_RUN[@]}" scripts/make_submission.py --exp "$exp" --postprocess argmax
    else
        log "[skip] $exp argmax submission already exists"
    fi

    if [[ ! -f "$root/submission_${exp}_tuned.csv" ]]; then
        run_step "$exp — tuned submission" \
            "${PYTHON_RUN[@]}" scripts/make_submission.py --exp "$exp" --postprocess multipliers
    else
        log "[skip] $exp tuned submission already exists"
    fi
}

log "════════════════════════════════════════════════════════════"
log "KAGGLE S6E7 FULL PIPELINE — no HPO before ablations"
log "Start: $(timestamp) | Sweep base: $BASE_EXP | Trials: $N_TRIALS"
log "Compute: LightGBM CPU, n_jobs=6, OMP_NUM_THREADS=$OMP_NUM_THREADS"
log "════════════════════════════════════════════════════════════"

free_kb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( free_kb < required_kb )); then
    log "✗ Insufficient disk space: at least ${MIN_FREE_GB} GiB free is required"
    log "  Current free space: $((free_kb / 1024 / 1024)) GiB"
    log "  Safe cleanup: rm -rf outputs/cache outputs/dry_runs"
    exit 3
fi
log "Disk pre-flight: $((free_kb / 1024 / 1024)) GiB free (minimum ${MIN_FREE_GB} GiB)"

run_step "Pre-flight quality checks" "${PYTHON_RUN[@]}" scripts/check.py

log ""
log "--- PHASE 1: Feature ablation model training ---"
for exp in "${EXPERIMENTS[@]}"; do
    train_experiment "$exp"
done

run_step "PHASE 2 — Local experiment leaderboard" \
    "${PYTHON_RUN[@]}" scripts/compare_experiments.py \
    --output outputs/leaderboard_local.csv

log ""
log "--- PHASE 3: OOF multiplier tuning and submissions ---"
for exp in "${POSTPROCESS_EXPERIMENTS[@]}"; do
    postprocess_experiment "$exp"
done

if [[ "$RUN_SWEEP" == "1" ]]; then
    if [[ ! " ${EXPERIMENTS[*]} " =~ [[:space:]]${BASE_EXP}[[:space:]] ]]; then
        log "✗ BASE_EXP=$BASE_EXP is not one of: ${EXPERIMENTS[*]}"
        exit 2
    fi
    run_step "PHASE 4 — LightGBM sweep ($BASE_EXP, $N_TRIALS trials)" \
        "${PYTHON_RUN[@]}" scripts/run_sweep.py \
        --base-exp "$BASE_EXP" --sweep configs/sweeps.yaml --n-trials "$N_TRIALS"
else
    log "[skip] PHASE 4 sweep disabled with RUN_SWEEP=$RUN_SWEEP"
fi

log ""
log "════════════════════════════════════════════════════════════"
log "PIPELINE COMPLETE: $(timestamp)"
log "Leaderboard: outputs/leaderboard_local.csv"
log "Submissions: outputs/experiments/{E002,E004,E006,E008}/"
log "Sweep outputs: outputs/experiments/SWEEP_*/"
log "Log: $PIPELINE_LOG"
log "════════════════════════════════════════════════════════════"
