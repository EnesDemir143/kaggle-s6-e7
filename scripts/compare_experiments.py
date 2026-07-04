#!/usr/bin/env python3
"""Build a local leaderboard from experiment metric artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output", type=Path, default=Path("outputs/leaderboard_local.csv"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.experiments_dir.glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        fold = metrics.get("fold_summary", {})
        recalls = metrics["class_recall"]
        distribution = metrics["prediction_distribution"]
        rows.append({
            "experiment_name": path.parent.name,
            "mean_bal_acc": fold.get("balanced_accuracy", {}).get("mean", metrics["balanced_accuracy"]),
            "std_bal_acc": fold.get("balanced_accuracy", {}).get("std"),
            "at-risk_recall": recalls["at-risk"],
            "fit_recall": recalls["fit"],
            "unhealthy_recall": recalls["unhealthy"],
            "macro_f1": metrics["f1_macro"],
            "pred_at-risk_rate": distribution["at-risk"]["rate"],
            "pred_fit_rate": distribution["fit"]["rate"],
            "pred_unhealthy_rate": distribution["unhealthy"]["rate"],
            "best_iteration_mean": fold.get("best_iteration", {}).get("mean"),
            "features_count": fold.get("features_count", {}).get("mean"),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mean_bal_acc", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False) if not result.empty else "No completed experiments found")


if __name__ == "__main__":
    main()
