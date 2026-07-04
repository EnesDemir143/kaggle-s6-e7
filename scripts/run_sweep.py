#!/usr/bin/env python3
"""Run a seeded small random LightGBM parameter sweep."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kaggle_s6_e7.cache import FoldFeatureCache, load_cached_csv
from kaggle_s6_e7.data import validate_schema
from kaggle_s6_e7.experiment_config import load_yaml, resolve_experiment
from kaggle_s6_e7.training import run_cv_experiment
from run_experiment import stratified_sample

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-exp", required=True)
    parser.add_argument("--sweep", type=Path, default=Path("configs/sweeps.yaml"))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    log.info(
        "Sweep: base_exp=%s n_trials=%d dry_run=%s force=%s",
        args.base_exp, args.n_trials, args.dry_run, args.force,
    )
    sweep = load_yaml(args.sweep)["lgbm_small_sweep"]
    experiment = resolve_experiment(args.base_exp)
    base_params = load_yaml("configs/lgbm_base.yaml")
    train_path = Path("data/playground-series-s6e7/train.csv")
    test_path = Path("data/playground-series-s6e7/test.csv")
    log.info("Loading data")
    train, test = load_cached_csv(train_path), load_cached_csv(test_path)
    validate_schema(train, test)
    if args.dry_run:
        train = stratified_sample(train, 3000, int(experiment["seed"]))
        test = test.head(1000).copy()
    rng = np.random.default_rng(int(experiment["seed"]))
    trials = min(args.n_trials, 2) if args.dry_run else args.n_trials
    root = Path("outputs/dry_runs/sweeps" if args.dry_run else "outputs/experiments")
    search_space = sweep["search"]
    log.info("Search space: %s", {k: {"min": min(v), "max": max(v)} for k, v in search_space.items()})
    trial_bar = tqdm(range(trials), desc="Sweep", unit="trial")
    for trial in trial_bar:
        params = dict(base_params)
        selected = {key: rng.choice(values).item() for key, values in search_space.items()}
        params.update(selected)
        if args.dry_run:
            params.update(n_estimators=20, early_stopping_rounds=5, min_child_samples=20)
        trial_id = f"SWEEP_{trial:03d}"
        output = root / trial_id
        if (output / "metrics.json").exists() and not args.force:
            log.debug("Trial %s already complete, skipping", trial_id)
            continue
        output.mkdir(parents=True, exist_ok=True)
        trial_config = {**experiment, "experiment_id": trial_id, "base_exp": args.base_exp, "model_overrides": selected}
        (output / "config.json").write_text(json.dumps(trial_config, indent=2) + "\n")
        trial_bar.set_postfix_str(trial_id)
        log.info("Trial %s | params: %s", trial_id, selected)
        run_cv_experiment(
            experiment_id=trial_id,
            experiment=experiment,
            model_params=params,
            train=train,
            test=test,
            output_dir=output,
            train_path=train_path,
            test_path=test_path,
            cache=FoldFeatureCache(),
        )


if __name__ == "__main__":
    main()
