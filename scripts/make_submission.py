#!/usr/bin/env python3
"""Create a schema-safe submission from stored test probabilities."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_s6_e7.config import CLASS_NAMES, ID_COL, TARGET_COL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--postprocess", choices=["argmax", "multipliers"], required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/playground-series-s6e7/sample_submission.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = (Path("outputs/dry_runs") if args.dry_run else args.source_dir) / args.exp
    proba = np.load(source / "test_proba.npy")
    multipliers = np.ones(3)
    suffix = "argmax"
    if args.postprocess == "multipliers":
        mapping = json.loads((source / "best_multipliers.json").read_text())
        multipliers = np.array([mapping[label] for label in CLASS_NAMES])
        suffix = "tuned"
    sample = pd.read_csv(args.sample_submission)
    if args.dry_run:
        sample = sample.head(len(proba)).copy()
    if len(sample) != len(proba):
        raise ValueError("sample_submission and test probabilities have different row counts")
    result = sample[[ID_COL]].copy()
    result[TARGET_COL] = np.asarray(CLASS_NAMES)[np.argmax(proba * multipliers, axis=1)]
    output = source / f"submission_{args.exp}_{suffix}.csv"
    result.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
