#!/usr/bin/env python3
"""Tune class decision multipliers using OOF probabilities only."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_s6_e7.config import CLASS_NAMES, ID_COL, TARGET_COL
from kaggle_s6_e7.postprocess import tune_class_multipliers
from kaggle_s6_e7.ensemble import search_multiplier_scales

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--metric", default="balanced_accuracy", choices=["balanced_accuracy"])
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        help="Optional deterministic fit/unhealthy scale grid around [1,1,1]",
    )
    args = parser.parse_args()
    source = (Path("outputs/dry_runs") if args.dry_run else args.source_dir) / args.exp
    log.info("Multiplier tuning: exp=%s source=%s", args.exp, source)
    proba = np.load(source / "oof_proba.npy")
    oof = pd.read_csv(source / "oof_pred.csv")
    trials = 100 if args.dry_run else 2000
    log.info("Loaded OOF: %d samples, %d classes | trials=%d", proba.shape[0], proba.shape[1], trials)
    if args.scales:
        multipliers, metrics = search_multiplier_scales(
            oof["y_true"].to_numpy(), proba, np.ones(3), args.scales
        )
    else:
        multipliers, metrics = tune_class_multipliers(
            oof["y_true"].to_numpy(), proba, random_trials=trials
        )
    mapping = dict(zip(CLASS_NAMES, map(float, multipliers), strict=True))
    (source / "best_multipliers.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (source / "metrics_tuned.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log.info("Multipliers saved: %s | bal_acc=%.4f | dist=%s",
             mapping, metrics["balanced_accuracy"], metrics.get("prediction_distribution", {}))
    test_proba = np.load(source / "test_proba.npy")
    submission = pd.read_csv(source / "submission_argmax.csv")[[ID_COL]]
    submission[TARGET_COL] = np.asarray(CLASS_NAMES)[np.argmax(test_proba * multipliers, axis=1)]
    submission.to_csv(source / "submission_tuned.csv", index=False)
    log.info("Tuned submission saved: %s", source / "submission_tuned.csv")
    print(json.dumps({"multipliers": mapping, **metrics}, indent=2))


if __name__ == "__main__":
    main()
