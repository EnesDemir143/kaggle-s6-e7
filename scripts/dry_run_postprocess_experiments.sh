#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_RUN=(uv run --no-sync python)
DRY_ROOT="outputs/dry_runs/postprocess_experiments"
BEFORE_CACHE="$(mktemp)"
AFTER_CACHE="$(mktemp)"
BEFORE_PROD="$(mktemp)"
AFTER_PROD="$(mktemp)"
trap 'rm -f "$BEFORE_CACHE" "$AFTER_CACHE" "$BEFORE_PROD" "$AFTER_PROD"' EXIT

snapshot() {
    local path="$1"
    local target="$2"
    if [[ -d "$path" ]]; then
        find "$path" -type f -print0 | sort -z | xargs -0 stat -f '%N|%z|%m' > "$target"
    else
        : > "$target"
    fi
}

echo "════════════════════════════════════════════════════════════"
echo "E009-E013 POSTPROCESS DRY RUN"
echo "Source: outputs/experiments (read-only)"
echo "Output: $DRY_ROOT"
echo "════════════════════════════════════════════════════════════"

snapshot outputs/cache "$BEFORE_CACHE"
snapshot outputs/experiments "$BEFORE_PROD"

"${PYTHON_RUN[@]}" scripts/generate_postprocess_experiments.py \
    --source-root outputs/experiments \
    --output-root "$DRY_ROOT" \
    --force

snapshot outputs/cache "$AFTER_CACHE"
snapshot outputs/experiments "$AFTER_PROD"
cmp -s "$BEFORE_CACHE" "$AFTER_CACHE" || { echo "✗ Dry run changed outputs/cache"; exit 4; }
cmp -s "$BEFORE_PROD" "$AFTER_PROD" || { echo "✗ Dry run changed production experiments"; exit 5; }

echo "✓ Dry run complete; cache and production outputs are unchanged."
echo "Production command: bash scripts/run_postprocess_experiments.sh"
