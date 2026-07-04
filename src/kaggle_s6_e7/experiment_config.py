"""Validated YAML configuration loading with recursive inheritance."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    content = yaml.safe_load(Path(path).read_text())
    if not isinstance(content, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return content


def resolve_experiment(
    experiment_id: str, path: str | Path = "configs/experiments.yaml"
) -> dict[str, Any]:
    experiments = load_yaml(path)

    def resolve(name: str, stack: tuple[str, ...]) -> dict[str, Any]:
        if name not in experiments:
            raise KeyError(f"Unknown experiment: {name}")
        if name in stack:
            raise ValueError(f"Experiment inheritance cycle: {' -> '.join((*stack, name))}")
        raw = experiments[name]
        if not isinstance(raw, dict):
            raise ValueError(f"Experiment {name} must be a mapping")
        parent = raw.get("inherit")
        own = {key: value for key, value in raw.items() if key != "inherit"}
        resolved = deep_merge(resolve(parent, (*stack, name)), own) if parent else own
        resolved["experiment_id"] = name
        return resolved

    return resolve(experiment_id, ())
