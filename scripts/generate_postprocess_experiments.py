#!/usr/bin/env python3
"""Generate E009-E013 submissions from stored OOF and test probabilities."""

import argparse
import logging
from pathlib import Path

from kaggle_s6_e7.candidate_experiments import run_candidate_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/postprocess_experiments.yaml"))
    parser.add_argument("--source-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    report = run_candidate_suite(args.config, args.source_root, args.output_root, force=args.force)
    print(report.to_string(index=False))
    print("\nCandidate generation complete; see the configured report_stem under the output root.")


if __name__ == "__main__":
    main()
