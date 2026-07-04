"""OOF-only class multiplier search."""

import logging
from itertools import product
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from tqdm import tqdm

from .config import CLASS_NAMES

log = logging.getLogger(__name__)


def multiplier_metrics(y_true: np.ndarray, proba: np.ndarray, multipliers: np.ndarray) -> dict[str, Any]:
    pred_idx = np.argmax(proba * multipliers, axis=1)
    y_idx = np.array([CLASS_NAMES.index(value) for value in y_true])
    recalls = recall_score(y_idx, pred_idx, labels=range(3), average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_idx, pred_idx)),
        "macro_f1": float(f1_score(y_idx, pred_idx, average="macro", zero_division=0)),
        "class_recall": dict(zip(CLASS_NAMES, map(float, recalls), strict=True)),
        "prediction_distribution": {
            label: float(np.mean(pred_idx == index)) for index, label in enumerate(CLASS_NAMES)
        },
    }


def tune_class_multipliers(
    y_true: np.ndarray, proba: np.ndarray, *, seed: int = 42, random_trials: int = 2000
) -> tuple[np.ndarray, dict[str, Any]]:
    best = np.ones(3)
    best_score = -1.0
    coarse = np.arange(0.8, 1.5001, 0.025)
    log.info("Coarse grid: %d × %d = %d candidates", len(coarse), len(coarse), len(coarse) ** 2)
    coarse_iter = tqdm(product(coarse, coarse), desc="Coarse grid", unit="trial", total=len(coarse) ** 2)
    for fit, unhealthy in coarse_iter:
        candidate = np.array([1.0, fit, unhealthy])
        score = multiplier_metrics(y_true, proba, candidate)["balanced_accuracy"]
        if score > best_score:
            best, best_score = candidate, score
    log.info("Coarse best: multipliers=%s bal_acc=%.4f", list(np.round(best, 4)), best_score)
    rng = np.random.default_rng(seed)
    refine_iter = tqdm(range(random_trials), desc="Random refinement", unit="trial")
    for _ in refine_iter:
        candidate = best.copy()
        candidate[1:] += rng.uniform(-0.05, 0.05, size=2)
        candidate = np.clip(candidate, 0.05, None)
        score = multiplier_metrics(y_true, proba, candidate)["balanced_accuracy"]
        if score > best_score:
            best, best_score = candidate, score
            refine_iter.set_postfix({"best": f"{best_score:.4f}"})
    best /= best.mean()
    log.info(
        "Tuned final: multipliers=%s bal_acc=%.4f | "
        "recall: at-risk=%.3f fit=%.3f unhealthy=%.3f",
        list(np.round(best, 4)),
        best_score,
        multiplier_metrics(y_true, proba, best)["class_recall"]["at-risk"],
        multiplier_metrics(y_true, proba, best)["class_recall"]["fit"],
        multiplier_metrics(y_true, proba, best)["class_recall"]["unhealthy"],
    )
    return best, multiplier_metrics(y_true, proba, best)
