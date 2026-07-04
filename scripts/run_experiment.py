#!/usr/bin/env python3
"""Run one configured cross-validated LightGBM experiment."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from kaggle_s6_e7.cache import FoldFeatureCache, load_cached_csv
from kaggle_s6_e7.config import TARGET_COL
from kaggle_s6_e7.data import validate_schema
from kaggle_s6_e7.experiment_config import load_yaml, resolve_experiment
from kaggle_s6_e7.training import run_cv_experiment

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/lgbm_base.yaml"))
    parser.add_argument("--train-path", type=Path, default=Path("data/playground-series-s6e7/train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/playground-series-s6e7/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stratified_sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= rows:
        return frame.reset_index(drop=True)
    fraction = rows / len(frame)
    return (
        frame.groupby(TARGET_COL, group_keys=False)
        .sample(frac=fraction, random_state=seed)
        .reset_index(drop=True)
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    log.info("Experiment: %s | dry_run=%s | no_cache=%s", args.exp, args.dry_run, args.no_cache)
    experiment = resolve_experiment(args.exp, args.config)
    if experiment.get("model") != "lightgbm":
        raise ValueError(f"{args.exp} is a postprocess-only experiment")
    log.info("Loading data: train=%s test=%s", args.train_path, args.test_path)
    train, test = load_cached_csv(args.train_path), load_cached_csv(args.test_path)
    log.info("Train rows: %d | Test rows: %d | Features: %d", len(train), len(test), len(train.columns) - 2)
    log.info("Validating schema")
    validate_schema(train, test)
    model_params = load_yaml(args.model_config)
    output_root = Path("outputs/dry_runs") if args.dry_run else args.output_dir
    output = output_root / args.exp
    if output.exists() and (output / "metrics.json").exists() and not args.force:
        raise FileExistsError(f"Completed output exists: {output}; pass --force to replace")
    if args.dry_run:
        log.info("Dry-run mode: subsampling to 3000 train / 1000 test rows, reducing n_estimators")
        train = stratified_sample(train, 3000, int(experiment["seed"]))
        test = test.head(1000).copy()
        model_params["n_estimators"] = 20
        model_params["early_stopping_rounds"] = 5
        model_params["min_child_samples"] = 20
    output.mkdir(parents=True, exist_ok=True)
    persisted_config = {**experiment, "model_params": model_params, "dry_run": args.dry_run}
    (output / "config.json").write_text(json.dumps(persisted_config, indent=2) + "\n")
    log.info("Starting 3-fold CV training → %s", output)
    metrics = run_cv_experiment(
        experiment_id=args.exp,
        experiment=experiment,
        model_params=model_params,
        train=train,
        test=test,
        output_dir=output,
        train_path=args.train_path,
        test_path=args.test_path,
        cache=FoldFeatureCache(enabled=not args.no_cache),
    )
    log.info("Artifacts saved to %s", output)
    print(json.dumps({"experiment": args.exp, "balanced_accuracy": metrics["balanced_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
