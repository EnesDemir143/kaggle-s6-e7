#!/usr/bin/env python3
"""Persist the deterministic fallback decision for E019 or E020."""

import argparse
import json
from pathlib import Path

import numpy as np

from kaggle_s6_e7.config import CLASS_NAMES
from kaggle_s6_e7.ensemble import apply_multipliers, disagreement_rate


def load_multipliers(directory: Path) -> np.ndarray:
    mapping = json.loads((directory / "best_multipliers.json").read_text())
    return np.array([mapping[label] for label in CLASS_NAMES])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["E019", "E020"], required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--oof-gap", type=float, default=0.0015)
    parser.add_argument("--min-disagreement", type=float, default=0.001)
    args = parser.parse_args()
    base = args.root / "E002"
    source = args.root / args.exp
    base_score = json.loads((base / "metrics_tuned.json").read_text())["balanced_accuracy"]
    source_score = json.loads((source / "metrics_tuned.json").read_text())["balanced_accuracy"]
    base_labels = apply_multipliers(np.load(base / "test_proba.npy"), load_multipliers(base))
    source_labels = apply_multipliers(
        np.load(source / "test_proba.npy"), load_multipliers(source)
    )
    disagreement = disagreement_rate(base_labels, source_labels)
    reasons = []
    if source_score < base_score - args.oof_gap:
        reasons.append("tuned_oof_below_threshold")
    if disagreement < args.min_disagreement:
        reasons.append("disagreement_below_threshold")
    payload = {
        "experiment": args.exp,
        "fallback_required": bool(reasons),
        "reasons": reasons,
        "base_tuned_oof": base_score,
        "source_tuned_oof": source_score,
        "test_disagreement": disagreement,
    }
    (source / "fallback_decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
