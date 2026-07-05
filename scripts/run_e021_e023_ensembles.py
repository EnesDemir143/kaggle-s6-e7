#!/usr/bin/env python3
"""Generate the fixed E021-E023 E002-centred ensemble candidates."""

import argparse
import json
import tempfile
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from kaggle_s6_e7.candidate_experiments import run_candidate_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/e021_e023_ensembles.yaml"))
    parser.add_argument("--source-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config
    selection_path = args.source_root / "e019_e020_selected_sources.json"
    if selection_path.is_file():
        selected = json.loads(selection_path.read_text())
        config = yaml.safe_load(args.config.read_text())
        for spec in config["experiments"].values():
            spec["sources"] = {
                selected.get(source, source): weight
                for source, weight in spec["sources"].items()
            }
        temporary = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(config, temporary)
        temporary.close()
        config_path = Path(temporary.name)
    report = run_candidate_suite(config_path, args.source_root, args.output_root, force=args.force)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
