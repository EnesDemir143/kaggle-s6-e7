#!/usr/bin/env python3
"""Select main or fallback probability sources after conditional training."""

import json
from pathlib import Path


def score(path: Path) -> float:
    return float(json.loads((path / "metrics_tuned.json").read_text())["balanced_accuracy"])


def main() -> None:
    root = Path("outputs/experiments")
    selected: dict[str, str] = {}
    for main, fallback in (("E019", "E019_ALT"), ("E020", "E020_ALT")):
        candidates = [main]
        if (root / fallback / "metrics_tuned.json").is_file():
            candidates.append(fallback)
        selected[main] = max(candidates, key=lambda name: score(root / name))
    output = root / "e019_e020_selected_sources.json"
    output.write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
