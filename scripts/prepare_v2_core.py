#!/usr/bin/env python3
"""Fit P_MAIN_V2_CORE on train and persist model-ready feature frames."""

import argparse
import json
from pathlib import Path

import joblib

from kaggle_s6_e7.config import ID_COL, TARGET_COL
from kaggle_s6_e7.data import load_competition_data, validate_schema
from kaggle_s6_e7.preprocessing import V2CorePreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/E002/artifacts")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, test = load_competition_data()
    validate_schema(train, test)
    processor = V2CorePreprocessor()
    train_features = processor.fit_transform(train.drop(columns=[ID_COL, TARGET_COL]))
    test_features = processor.transform(test.drop(columns=ID_COL))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_features.to_pickle(args.output_dir / "train_features.pkl")
    test_features.to_pickle(args.output_dir / "test_features.pkl")
    train[[ID_COL, TARGET_COL]].to_pickle(args.output_dir / "train_keys.pkl")
    test[[ID_COL]].to_pickle(args.output_dir / "test_keys.pkl")
    joblib.dump(processor, args.output_dir / "preprocessor.joblib")
    manifest = {
        "pipeline": "P_MAIN_V2_CORE",
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "feature_count": train_features.shape[1],
        "categorical_features": processor.categorical_features_,
        "features": processor.feature_names_out_,
        "outlier_quantiles": [processor.lower_q, processor.upper_q],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
