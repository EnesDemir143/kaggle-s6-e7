#!/usr/bin/env python3
"""Execute every experiment path in bounded dry-run mode."""

import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    command = (sys.executable, *args)
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    for experiment in ("E001", "E002", "E003", "E004", "E005", "E006", "E008"):
        run("scripts/run_experiment.py", "--exp", experiment, "--dry-run", "--force")
    run("scripts/tune_multipliers.py", "--exp", "E002", "--dry-run")
    run("scripts/make_submission.py", "--exp", "E002", "--postprocess", "argmax", "--dry-run")
    run("scripts/make_submission.py", "--exp", "E002", "--postprocess", "multipliers", "--dry-run")
    run("scripts/compare_experiments.py", "--experiments-dir", "outputs/dry_runs", "--output", "outputs/dry_runs/leaderboard_local.csv")
    run("scripts/run_sweep.py", "--base-exp", "E002", "--n-trials", "2", "--dry-run", "--force")
    required = [Path("outputs/dry_runs") / exp / "metrics.json" for exp in ("E001", "E002", "E003", "E004", "E005", "E006", "E008")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Dry-run artifact validation failed")
    print("\nAll experiment dry-runs passed.")


if __name__ == "__main__":
    main()
