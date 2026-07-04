"""Reusable probability-ensemble and submission eligibility helpers."""

from itertools import product
from typing import Any, Sequence

import numpy as np

from .config import CLASS_NAMES
from .postprocess import multiplier_metrics


def _validated_probabilities(values: np.ndarray) -> np.ndarray:
    proba = np.asarray(values, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"probabilities must have shape (rows, {len(CLASS_NAMES)})")
    if not np.isfinite(proba).all():
        raise ValueError("probabilities must contain only finite values")
    row_sums = proba.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("probability rows must have positive sums")
    return proba / row_sums


def blend_probabilities(probabilities: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Return a row-normalized weighted probability blend."""
    if not probabilities or len(probabilities) != len(weights):
        raise ValueError("probability sources and weights must be non-empty and equally sized")
    arrays = [_validated_probabilities(value) for value in probabilities]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("probability sources must have the same shape")
    normalized_weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(normalized_weights).all() or np.any(normalized_weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if normalized_weights.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    normalized_weights /= normalized_weights.sum()
    blended = np.zeros_like(arrays[0], dtype=np.float64)
    for weight, value in zip(normalized_weights, arrays, strict=True):
        blended += weight * value
    return _validated_probabilities(blended)


def apply_multipliers(probabilities: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Convert probabilities to class labels after applying decision multipliers."""
    proba = _validated_probabilities(probabilities)
    factors = np.asarray(multipliers, dtype=np.float64)
    if factors.shape != (len(CLASS_NAMES),) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("multipliers must be a positive finite value per class")
    return np.asarray(CLASS_NAMES)[np.argmax(proba * factors, axis=1)]


def search_multiplier_scales(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    base_multipliers: np.ndarray,
    scales: Sequence[float],
    *,
    scale_pairs: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Search fit/unhealthy scales around a fixed base multiplier vector."""
    base = np.asarray(base_multipliers, dtype=np.float64)
    if base.shape != (len(CLASS_NAMES),):
        raise ValueError("base_multipliers must contain one value per class")
    best_multipliers = base.copy()
    best_metrics = multiplier_metrics(np.asarray(y_true), probabilities, best_multipliers)
    candidates = scale_pairs if scale_pairs is not None else product(scales, repeat=2)
    for pair in candidates:
        if len(pair) != 2:
            raise ValueError("each multiplier scale pair must contain fit and unhealthy scales")
        fit_scale, unhealthy_scale = pair
        candidate = base * np.array([1.0, fit_scale, unhealthy_scale])
        metrics = multiplier_metrics(np.asarray(y_true), probabilities, candidate)
        if metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]:
            best_multipliers, best_metrics = candidate, metrics
    return best_multipliers, best_metrics


def consensus_correction(base: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Replace base labels only where both alternative models agree against it."""
    base_labels, left_labels, right_labels = map(np.asarray, (base, left, right))
    if not (base_labels.shape == left_labels.shape == right_labels.shape):
        raise ValueError("consensus label arrays must have the same shape")
    corrected = base_labels.copy()
    mask = (left_labels == right_labels) & (left_labels != base_labels)
    corrected[mask] = left_labels[mask]
    return corrected


def disagreement_rate(first: np.ndarray, second: np.ndarray) -> float:
    first_labels, second_labels = np.asarray(first), np.asarray(second)
    if first_labels.shape != second_labels.shape:
        raise ValueError("label arrays must have the same shape")
    return float(np.mean(first_labels != second_labels))


def eligibility_reasons(
    *,
    distribution: dict[str, float],
    distribution_bounds: dict[str, Sequence[float]] | None = None,
    disagreement: float | None = None,
    disagreement_bounds: Sequence[float] | None = None,
    oof_score: float | None = None,
    min_oof_score: float | None = None,
) -> list[str]:
    """Return human-readable reasons why a generated candidate is not submission eligible."""
    reasons: list[str] = []
    for label, bounds in (distribution_bounds or {}).items():
        value = distribution[label]
        if not bounds[0] <= value <= bounds[1]:
            reasons.append(f"{label} distribution {value:.6f} outside [{bounds[0]:.6f}, {bounds[1]:.6f}]")
    if disagreement_bounds is not None and disagreement is not None:
        if not disagreement_bounds[0] <= disagreement <= disagreement_bounds[1]:
            reasons.append(
                f"disagreement {disagreement:.6f} outside "
                f"[{disagreement_bounds[0]:.6f}, {disagreement_bounds[1]:.6f}]"
            )
    if min_oof_score is not None and (oof_score is None or oof_score < min_oof_score):
        reasons.append(f"OOF balanced accuracy {oof_score!s} below {min_oof_score:.6f}")
    return reasons
