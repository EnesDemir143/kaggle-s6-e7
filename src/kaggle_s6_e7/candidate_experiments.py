"""Generate controlled E002-centered postprocess experiment candidates."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import balanced_accuracy_score

from .config import CLASS_NAMES, ID_COL, TARGET_COL
from .ensemble import (
    apply_multipliers,
    blend_probabilities,
    consensus_correction,
    disagreement_rate,
    eligibility_reasons,
    search_multiplier_scales,
)


@dataclass(frozen=True)
class ProbabilitySource:
    name: str
    oof_proba: np.ndarray
    test_proba: np.ndarray
    oof_ids: np.ndarray
    y_true: np.ndarray
    test_ids: np.ndarray
    multipliers: np.ndarray
    fold_assignments: np.ndarray | None


class NoEligibleCandidateError(ValueError):
    """Raised when a constrained search has no submission-safe candidate."""

    def __init__(self, evaluations: list[dict[str, Any]]) -> None:
        super().__init__("no multiplier candidate passed the configured selection filters")
        self.evaluations = evaluations


def _load_source(root: Path, name: str) -> ProbabilitySource:
    directory = root / name
    mapping = json.loads((directory / "label_mapping.json").read_text())
    expected = {label: index for index, label in enumerate(CLASS_NAMES)}
    if mapping != expected:
        raise ValueError(f"{name} class mapping differs from {expected}")
    oof = pd.read_csv(directory / "oof_pred.csv")
    submission_path = directory / "submission_argmax.csv"
    if not submission_path.is_file():
        submission_path = directory / f"submission_{name}_argmax.csv"
    submission = pd.read_csv(submission_path)
    multiplier_path = directory / "best_multipliers.json"
    multiplier_mapping = (
        json.loads(multiplier_path.read_text())
        if multiplier_path.is_file()
        else {label: 1.0 for label in CLASS_NAMES}
    )
    fold_path = directory / "fold_assignments.csv"
    fold_assignments = (
        pd.read_csv(fold_path)["fold_assignment"].to_numpy()
        if fold_path.is_file()
        else None
    )
    return ProbabilitySource(
        name=name,
        oof_proba=np.load(directory / "oof_proba.npy"),
        test_proba=np.load(directory / "test_proba.npy"),
        oof_ids=oof[ID_COL].to_numpy(),
        y_true=oof["y_true"].to_numpy(),
        test_ids=submission[ID_COL].to_numpy(),
        multipliers=np.array([multiplier_mapping[label] for label in CLASS_NAMES]),
        fold_assignments=fold_assignments,
    )


def _validate_alignment(sources: dict[str, ProbabilitySource]) -> None:
    reference = next(iter(sources.values()))
    for source in sources.values():
        if not np.array_equal(source.oof_ids, reference.oof_ids):
            raise ValueError(f"{source.name} OOF IDs are not aligned with {reference.name}")
        if not np.array_equal(source.y_true, reference.y_true):
            raise ValueError(f"{source.name} OOF labels are not aligned with {reference.name}")
        if not np.array_equal(source.test_ids, reference.test_ids):
            raise ValueError(f"{source.name} test IDs are not aligned with {reference.name}")
        if source.oof_proba.shape != reference.oof_proba.shape:
            raise ValueError(f"{source.name} OOF probabilities have a different shape")
        if source.test_proba.shape != reference.test_proba.shape:
            raise ValueError(f"{source.name} test probabilities have a different shape")
        if (
            source.fold_assignments is not None
            and reference.fold_assignments is not None
            and not np.array_equal(source.fold_assignments, reference.fold_assignments)
        ):
            raise ValueError(f"{source.name} fold assignments differ from {reference.name}")


def _distribution(labels: np.ndarray) -> dict[str, float]:
    return {label: float(np.mean(labels == label)) for label in CLASS_NAMES}


def _risk_summary(base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    transitions = {
        f"{source}->{target}": int(np.sum((base == source) & (candidate == target)))
        for source in CLASS_NAMES
        for target in CLASS_NAMES
        if source != target
    }
    changed_rows = int(np.sum(base != candidate))
    at_risk_to_minority = sum(
        transitions[f"at-risk->{target}"] for target in ("fit", "unhealthy")
    )
    minority_to_at_risk = sum(
        transitions[f"{source}->at-risk"] for source in ("fit", "unhealthy")
    )
    base_distribution = _distribution(base)
    candidate_distribution = _distribution(candidate)
    drift = {
        label: candidate_distribution[label] - base_distribution[label]
        for label in CLASS_NAMES
    }
    return {
        "changed_rows": changed_rows,
        "transition_matrix": transitions,
        "at_risk_to_minority": at_risk_to_minority,
        "minority_to_at_risk": minority_to_at_risk,
        "e018_like_one_way_push": (
            at_risk_to_minority > 150 and at_risk_to_minority > 3 * max(1, minority_to_at_risk)
        ),
        "base_distribution": base_distribution,
        "candidate_distribution": candidate_distribution,
        "distribution_drift": drift,
        "max_abs_distribution_drift": max(abs(value) for value in drift.values()),
    }


def _fold_metrics(reference: ProbabilitySource, labels: np.ndarray) -> list[dict[str, Any]]:
    if reference.fold_assignments is None:
        return []
    return [
        {
            "fold": int(fold),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    reference.y_true[reference.fold_assignments == fold],
                    labels[reference.fold_assignments == fold],
                )
            ),
        }
        for fold in sorted(np.unique(reference.fold_assignments))
    ]


def _confidence_analysis(
    reference: ProbabilitySource, base_labels: np.ndarray, candidate_labels: np.ndarray
) -> list[dict[str, Any]]:
    scores = reference.oof_proba * reference.multipliers
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    disagreement = base_labels != candidate_labels
    if not np.any(disagreement):
        return []
    thresholds = np.quantile(margins[disagreement], [0.25, 0.5, 0.75])
    buckets = np.digitize(margins, thresholds)
    rows: list[dict[str, Any]] = []
    for bucket in range(4):
        mask = disagreement & (buckets == bucket)
        if not np.any(mask):
            continue
        rows.append(
            {
                "margin_quartile": bucket + 1,
                "rows": int(np.sum(mask)),
                "mean_e002_margin": float(np.mean(margins[mask])),
                "e002_accuracy": float(np.mean(base_labels[mask] == reference.y_true[mask])),
                "candidate_accuracy": float(
                    np.mean(candidate_labels[mask] == reference.y_true[mask])
                ),
            }
        )
    return rows


def _weights_from_config(spec: dict[str, Any]) -> list[dict[str, float]]:
    if "weight_options" in spec:
        return list(spec["weight_options"])
    return [dict(spec["sources"])]


def _adjusted_scores(probabilities: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    return normalized * multipliers


def _alternative_probabilities(
    definition: dict[str, Any], sources: dict[str, ProbabilitySource]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "weights" in definition:
        names = list(definition["weights"])
        weights = list(definition["weights"].values())
        oof = blend_probabilities([sources[name].oof_proba for name in names], weights)
        test = blend_probabilities([sources[name].test_proba for name in names], weights)
    else:
        source = sources[definition["source"]]
        oof, test = source.oof_proba, source.test_proba
    multiplier_source = sources[definition["multiplier_source"]]
    multipliers = multiplier_source.multipliers * np.asarray(
        definition.get("multiplier_scales", [1.0, 1.0, 1.0])
    )
    return oof, test, multipliers


def _balanced_accuracy_after_direction_change(
    y_indices: np.ndarray,
    base_predictions: np.ndarray,
    mask: np.ndarray,
    from_index: int,
    to_index: int,
) -> float:
    totals = np.bincount(y_indices, minlength=len(CLASS_NAMES))
    correct = np.array(
        [np.sum((y_indices == index) & (base_predictions == index)) for index in range(len(CLASS_NAMES))]
    )
    correct[from_index] -= np.sum(mask & (y_indices == from_index))
    correct[to_index] += np.sum(mask & (y_indices == to_index))
    return float(np.mean(correct / totals))


def _run_selective_margin_correction(
    spec: dict[str, Any], sources: dict[str, ProbabilitySource]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base = sources[spec["base"]]
    base_oof_scores = _adjusted_scores(base.oof_proba, base.multipliers)
    base_test_scores = _adjusted_scores(base.test_proba, base.multipliers)
    base_oof_predictions = np.argmax(base_oof_scores, axis=1)
    base_test_predictions = np.argmax(base_test_scores, axis=1)
    base_oof_margin = np.max(base_oof_scores, axis=1) - np.partition(base_oof_scores, -2, axis=1)[:, -2]
    base_test_margin = np.max(base_test_scores, axis=1) - np.partition(base_test_scores, -2, axis=1)[:, -2]
    normalized_base_oof = base.oof_proba / base.oof_proba.sum(axis=1, keepdims=True)
    base_entropy = -np.sum(normalized_base_oof * np.log(np.clip(normalized_base_oof, 1e-15, 1.0)), axis=1)
    label_to_index = {label: index for index, label in enumerate(CLASS_NAMES)}
    y_indices = np.array([label_to_index[label] for label in base.y_true])
    base_counts = np.bincount(base_test_predictions, minlength=len(CLASS_NAMES))
    filters = spec["selection_filters"]
    evaluations: list[dict[str, Any]] = []
    eligible: list[tuple[float, np.ndarray, np.ndarray, dict[str, Any]]] = []

    for source_name, definition in spec["alternative_sources"].items():
        alt_oof, alt_test, alt_multipliers = _alternative_probabilities(definition, sources)
        alt_oof_scores = _adjusted_scores(alt_oof, alt_multipliers)
        alt_test_scores = _adjusted_scores(alt_test, alt_multipliers)
        alt_oof_predictions = np.argmax(alt_oof_scores, axis=1)
        alt_test_predictions = np.argmax(alt_test_scores, axis=1)
        alt_oof_margin = np.max(alt_oof_scores, axis=1) - np.partition(alt_oof_scores, -2, axis=1)[:, -2]
        alt_test_margin = np.max(alt_test_scores, axis=1) - np.partition(alt_test_scores, -2, axis=1)[:, -2]
        rows_oof = np.arange(len(base.y_true))
        rows_test = np.arange(len(base.test_ids))
        alt_gain_oof = (
            alt_oof_scores[rows_oof, alt_oof_predictions]
            - base_oof_scores[rows_oof, base_oof_predictions]
        )
        alt_gain_test = (
            alt_test_scores[rows_test, alt_test_predictions]
            - base_test_scores[rows_test, base_test_predictions]
        )
        disagreement_pool = alt_oof_predictions != base_oof_predictions
        base_margin_thresholds = {
            quantile: float(np.quantile(base_oof_margin[disagreement_pool], quantile))
            for quantile in spec["base_margin_quantiles"]
        }
        alt_gain_thresholds = {
            quantile: float(np.quantile(alt_gain_oof[disagreement_pool], quantile))
            for quantile in spec["alt_gain_quantiles"]
        }

        for from_label, to_label in spec["directions"]:
            from_index, to_index = label_to_index[from_label], label_to_index[to_label]
            direction_oof = (base_oof_predictions == from_index) & (alt_oof_predictions == to_index)
            direction_test = (base_test_predictions == from_index) & (alt_test_predictions == to_index)
            for margin_quantile in spec["base_margin_quantiles"]:
                for gain_quantile in spec["alt_gain_quantiles"]:
                    for min_alt_margin in spec["min_alt_margins"]:
                        margin_threshold = base_margin_thresholds[margin_quantile]
                        gain_threshold = alt_gain_thresholds[gain_quantile]
                        oof_mask = (
                            direction_oof
                            & (base_oof_margin <= margin_threshold)
                            & (alt_gain_oof >= gain_threshold)
                            & (alt_oof_margin >= min_alt_margin)
                        )
                        test_mask = (
                            direction_test
                            & (base_test_margin <= margin_threshold)
                            & (alt_gain_test >= gain_threshold)
                            & (alt_test_margin >= min_alt_margin)
                        )
                        score = _balanced_accuracy_after_direction_change(
                            y_indices, base_oof_predictions, oof_mask, from_index, to_index
                        )
                        changed_rows = int(np.sum(test_mask))
                        disagreement = changed_rows / len(base.test_ids)
                        test_counts = base_counts.copy()
                        test_counts[from_index] -= changed_rows
                        test_counts[to_index] += changed_rows
                        count_differences = np.abs(test_counts - base_counts)
                        passed = (
                            score >= filters["min_oof_score"]
                            and filters["disagreement_bounds"][0]
                            <= disagreement
                            <= filters["disagreement_bounds"][1]
                            and filters["changed_test_rows"][0]
                            <= changed_rows
                            <= filters["changed_test_rows"][1]
                            and all(
                                count_differences[index] <= filters["max_test_count_difference"][label]
                                for index, label in enumerate(CLASS_NAMES)
                            )
                        )
                        evaluation = {
                            "source": source_name,
                            "direction": f"{from_label}->{to_label}",
                            "base_margin_quantile": margin_quantile,
                            "alt_gain_quantile": gain_quantile,
                            "min_alt_margin": min_alt_margin,
                            "base_margin_threshold": margin_threshold,
                            "alt_gain_threshold": gain_threshold,
                            "oof_balanced_accuracy": score,
                            "oof_changed_rows": int(np.sum(oof_mask)),
                            "mean_oof_entropy_selected": (
                                float(np.mean(base_entropy[oof_mask])) if np.any(oof_mask) else None
                            ),
                            "changed_test_rows": changed_rows,
                            "disagreement_vs_E002": disagreement,
                            "test_count_differences": {
                                label: int(count_differences[index]) for index, label in enumerate(CLASS_NAMES)
                            },
                            "passed_filters": passed,
                        }
                        evaluations.append(evaluation)
                        if passed:
                            oof_predictions = base_oof_predictions.copy()
                            test_predictions = base_test_predictions.copy()
                            oof_predictions[oof_mask] = to_index
                            test_predictions[test_mask] = to_index
                            eligible.append((score, oof_predictions, test_predictions, evaluation))

    if not eligible:
        raise NoEligibleCandidateError(evaluations)
    score, oof_predictions, test_predictions, selected = max(eligible, key=lambda item: item[0])
    return np.asarray(CLASS_NAMES)[oof_predictions], np.asarray(CLASS_NAMES)[test_predictions], {
        "oof_balanced_accuracy": score,
        "selected_rule": selected,
        "search_candidate_count": len(evaluations),
        "eligible_candidate_count": len(eligible),
        "eligible_candidates": sorted(
            (evaluation for evaluation in evaluations if evaluation["passed_filters"]),
            key=lambda evaluation: evaluation["oof_balanced_accuracy"],
            reverse=True,
        ),
    }


def _run_probability_candidate(
    spec: dict[str, Any], sources: dict[str, ProbabilitySource]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    kind = spec["kind"]
    if kind == "source_tuned_filtered":
        source = sources[spec["source"]]
        base = sources[spec["multiplier_source"]]
        base_test_labels = apply_multipliers(source.test_proba, base.multipliers)
        filters = spec["selection_filters"]
        evaluations: list[dict[str, Any]] = []
        eligible: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
        for name, fit_scale, unhealthy_scale in spec["candidates"]:
            multipliers = base.multipliers * np.array([1.0, fit_scale, unhealthy_scale])
            oof_labels = apply_multipliers(source.oof_proba, multipliers)
            test_labels = apply_multipliers(source.test_proba, multipliers)
            score = float(balanced_accuracy_score(source.y_true, oof_labels))
            disagreement = disagreement_rate(base_test_labels, test_labels)
            passed = score >= filters["min_oof_score"] and (
                filters["disagreement_bounds"][0]
                <= disagreement
                <= filters["disagreement_bounds"][1]
            )
            evaluation = {
                "candidate": name,
                "fit_scale": fit_scale,
                "unhealthy_scale": unhealthy_scale,
                "oof_balanced_accuracy": score,
                "disagreement_vs_E002": disagreement,
                "changed_test_rows": int(np.sum(base_test_labels != test_labels)),
                "multipliers": multipliers.tolist(),
                "passed_filters": passed,
            }
            evaluations.append(evaluation)
            if passed:
                eligible.append((score, oof_labels, test_labels, multipliers, evaluation))
        if not eligible:
            raise NoEligibleCandidateError(evaluations)
        score, oof_labels, test_labels, multipliers, selected = max(eligible, key=lambda item: item[0])
        return oof_labels, test_labels, {
            "oof_balanced_accuracy": score,
            "multipliers": multipliers.tolist(),
            "selected_candidate": selected["candidate"],
            "candidate_evaluations": evaluations,
        }
    if kind == "blend_tuned":
        best: tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, float]] | None = None
        for weights in _weights_from_config(spec):
            names = list(weights)
            oof = blend_probabilities([sources[name].oof_proba for name in names], list(weights.values()))
            test = blend_probabilities([sources[name].test_proba for name in names], list(weights.values()))
            multiplier_source = sources[spec["multiplier_source"]]
            multipliers, metrics = search_multiplier_scales(
                multiplier_source.y_true,
                oof,
                multiplier_source.multipliers,
                spec.get("scales", []),
                scale_pairs=spec.get("scale_pairs"),
            )
            score = float(metrics["balanced_accuracy"])
            if best is None or score > best[0]:
                best = score, oof, test, multipliers, metrics, weights
        assert best is not None
        score, _oof, test, multipliers, metrics, weights = best
        return apply_multipliers(_oof, multipliers), apply_multipliers(test, multipliers), {
            "oof_balanced_accuracy": score,
            "multipliers": multipliers.tolist(),
            "weights": weights,
            "oof_metrics": metrics,
        }
    if kind == "source_tuned":
        source = sources[spec["source"]]
        multipliers, metrics = search_multiplier_scales(
            source.y_true, source.oof_proba, sources[spec["multiplier_source"]].multipliers, spec["scales"]
        )
        return apply_multipliers(source.oof_proba, multipliers), apply_multipliers(
            source.test_proba, multipliers
        ), {
            "oof_balanced_accuracy": float(metrics["balanced_accuracy"]),
            "multipliers": multipliers.tolist(),
            "oof_metrics": metrics,
        }
    if kind == "fixed_multiplier":
        source = sources[spec["source"]]
        multipliers = sources[spec["multiplier_source"]].multipliers * np.array(spec["scales"])
        oof_labels = apply_multipliers(source.oof_proba, multipliers)
        return oof_labels, apply_multipliers(source.test_proba, multipliers), {
            "oof_balanced_accuracy": float(balanced_accuracy_score(source.y_true, oof_labels)),
            "multipliers": multipliers.tolist(),
        }
    raise ValueError(f"Unsupported probability candidate kind: {kind}")


def _run_candidate(
    spec: dict[str, Any], sources: dict[str, ProbabilitySource]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if spec["kind"] == "selective_margin_correction":
        return _run_selective_margin_correction(spec, sources)
    if spec["kind"] != "consensus":
        return _run_probability_candidate(spec, sources)
    base = sources[spec["base"]]
    left, right = (sources[name] for name in spec["voters"])
    base_oof = apply_multipliers(base.oof_proba, base.multipliers)
    base_test = apply_multipliers(base.test_proba, base.multipliers)
    oof_labels = consensus_correction(
        base_oof,
        apply_multipliers(left.oof_proba, left.multipliers),
        apply_multipliers(right.oof_proba, right.multipliers),
    )
    test_labels = consensus_correction(
        base_test,
        apply_multipliers(left.test_proba, left.multipliers),
        apply_multipliers(right.test_proba, right.multipliers),
    )
    return oof_labels, test_labels, {
        "oof_balanced_accuracy": float(balanced_accuracy_score(base.y_true, oof_labels)),
        "consensus_sources": spec["voters"],
    }


def run_candidate_suite(
    config_path: Path, source_root: Path, output_root: Path, *, force: bool = False
) -> pd.DataFrame:
    """Generate every configured candidate plus machine-readable eligibility reports."""
    config = yaml.safe_load(config_path.read_text())
    required = {config["base_experiment"]}
    for spec in config["experiments"].values():
        required.update(spec.get("sources", {}))
        for option in spec.get("weight_options", []):
            required.update(option)
        required.update(spec.get("voters", []))
        for definition in spec.get("alternative_sources", {}).values():
            required.update(definition.get("weights", {}))
            for key in ("source", "multiplier_source"):
                if key in definition:
                    required.add(definition[key])
        for key in ("source", "multiplier_source", "base"):
            if key in spec:
                required.add(spec[key])
    sources = {name: _load_source(source_root, name) for name in sorted(required)}
    _validate_alignment(sources)
    reference = sources[config["base_experiment"]]
    base_oof_labels = apply_multipliers(reference.oof_proba, reference.multipliers)
    base_test_labels = apply_multipliers(reference.test_proba, reference.multipliers)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for experiment, spec in config["experiments"].items():
        output = output_root / spec["output_dir"]
        eligibility_path = output / "eligibility.json"
        if eligibility_path.is_file() and not force:
            rows.append(json.loads(eligibility_path.read_text())["report_row"])
            continue
        output.mkdir(parents=True, exist_ok=True)
        try:
            oof_labels, test_labels, metrics = _run_candidate(spec, sources)
        except NoEligibleCandidateError as error:
            reason = str(error)
            submission_path = output / spec["submission_name"]
            if submission_path.exists():
                submission_path.unlink()
            row = {
                "experiment": experiment,
                "output_dir": spec["output_dir"],
                "eligible": False,
                "oof_balanced_accuracy": None,
                "disagreement_vs_E002": None,
                **{f"test_share_{label}": None for label in CLASS_NAMES},
                "reasons": reason,
            }
            (output / "config.json").write_text(json.dumps(spec, indent=2) + "\n")
            (output / "metrics.json").write_text(
                json.dumps({"selected_candidate": None, "candidate_evaluations": error.evaluations}, indent=2)
                + "\n"
            )
            eligibility_path.write_text(
                json.dumps({"eligible": False, "reasons": [reason], "report_row": row}, indent=2) + "\n"
            )
            rows.append(row)
            continue
        distribution = _distribution(test_labels)
        disagreement = disagreement_rate(base_test_labels, test_labels)
        risk = _risk_summary(base_test_labels, test_labels)
        oof_risk = _risk_summary(base_oof_labels, oof_labels)
        fold_metrics = _fold_metrics(reference, oof_labels)
        base_fold_metrics = _fold_metrics(reference, base_oof_labels)
        fold_non_regressions = sum(
            candidate["balanced_accuracy"] >= base["balanced_accuracy"]
            for candidate, base in zip(fold_metrics, base_fold_metrics, strict=True)
        )
        eligibility = spec.get("eligibility", {})
        reasons = eligibility_reasons(
            distribution=distribution,
            distribution_bounds=eligibility.get("distribution_bounds"),
            disagreement=disagreement,
            disagreement_bounds=eligibility.get("disagreement_bounds"),
            oof_score=metrics["oof_balanced_accuracy"],
            min_oof_score=eligibility.get("min_oof_score"),
        )
        changed_bounds = eligibility.get("changed_rows")
        if changed_bounds and not changed_bounds[0] <= risk["changed_rows"] <= changed_bounds[1]:
            reasons.append(
                f"changed rows {risk['changed_rows']} outside [{changed_bounds[0]}, {changed_bounds[1]}]"
            )
        max_push = eligibility.get("max_at_risk_to_minority")
        if max_push is not None and risk["at_risk_to_minority"] > max_push:
            reasons.append(f"at-risk to minority {risk['at_risk_to_minority']} above {max_push}")
        max_reverse = eligibility.get("max_minority_to_at_risk")
        if max_reverse is not None and risk["minority_to_at_risk"] > max_reverse:
            reasons.append(f"minority to at-risk {risk['minority_to_at_risk']} above {max_reverse}")
        max_drift = eligibility.get("max_abs_distribution_drift")
        if max_drift is not None and risk["max_abs_distribution_drift"] > max_drift:
            reasons.append(
                f"distribution drift {risk['max_abs_distribution_drift']:.6f} above {max_drift:.6f}"
            )
        min_fold_non_regressions = eligibility.get("min_fold_non_regressions")
        if (
            min_fold_non_regressions is not None
            and fold_metrics
            and fold_non_regressions < min_fold_non_regressions
        ):
            reasons.append(
                f"fold non-regressions {fold_non_regressions} below {min_fold_non_regressions}"
            )
        row = {
            "experiment": experiment,
            "output_dir": spec["output_dir"],
            "eligible": not reasons,
            "oof_balanced_accuracy": metrics["oof_balanced_accuracy"],
            "disagreement_vs_E002": disagreement,
            **{f"test_share_{label}": distribution[label] for label in CLASS_NAMES},
            "reasons": " | ".join(reasons),
        }
        submission_path = output / spec["submission_name"]
        if reasons:
            if submission_path.exists():
                submission_path.unlink()
            (output / "rejection_reasons.json").write_text(
                json.dumps(reasons, indent=2) + "\n"
            )
        else:
            submission = pd.DataFrame({ID_COL: reference.test_ids, TARGET_COL: test_labels})
            submission.to_csv(submission_path, index=False)
        (output / "config.json").write_text(json.dumps(spec, indent=2) + "\n")
        (output / "risk_summary.json").write_text(json.dumps(risk, indent=2) + "\n")
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    **metrics,
                    "test_distribution": distribution,
                    "disagreement_vs_E002": disagreement,
                    "risk_summary": risk,
                    "oof_risk_summary": oof_risk,
                    "fold_metrics": fold_metrics,
                    "base_fold_metrics": base_fold_metrics,
                    "fold_non_regressions": fold_non_regressions,
                    "disagreement_confidence": _confidence_analysis(
                        reference, base_oof_labels, oof_labels
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        eligibility_payload = {"eligible": not reasons, "reasons": reasons, "report_row": row}
        eligibility_path.write_text(json.dumps(eligibility_payload, indent=2) + "\n")
        rows.append(row)
    report = pd.DataFrame(rows)
    report_stem = config.get("report_stem", "eligibility_report")
    report.to_csv(output_root / f"{report_stem}.csv", index=False)
    (output_root / f"{report_stem}.json").write_text(
        json.dumps(report.to_dict(orient="records"), indent=2) + "\n"
    )
    return report
