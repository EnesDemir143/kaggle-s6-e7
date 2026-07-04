"""Multiclass OOF evaluation and ROC plotting utilities."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def predictions_from_probabilities(
    probabilities: NDArray[np.floating[Any]],
    class_labels: Sequence[str],
    class_multipliers: Sequence[float] | None = None,
) -> NDArray[np.str_]:
    """Convert probabilities to labels, optionally applying OOF-tuned multipliers."""
    proba = np.asarray(probabilities, dtype=float)
    labels = np.asarray(class_labels)
    if proba.ndim != 2 or proba.shape[1] != len(labels):
        raise ValueError("Probability columns must match class_labels")
    multipliers = (
        np.ones(len(labels))
        if class_multipliers is None
        else np.asarray(class_multipliers)
    )
    if multipliers.shape != (len(labels),) or np.any(multipliers <= 0):
        raise ValueError("class_multipliers must contain one positive value per class")
    return labels[np.argmax(proba * multipliers, axis=1)]


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: NDArray[np.floating[Any]],
    class_labels: Sequence[str],
) -> dict[str, Any]:
    """Return selection, imbalance-aware, probability, and per-class metrics."""
    labels = list(class_labels)
    proba = np.asarray(probabilities, dtype=float)
    if proba.shape != (len(y_true), len(labels)):
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    row_sums = proba.sum(axis=1)
    if np.any(proba < 0) or not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("probabilities must be non-negative and sum to one")
    proba = proba / row_sums[:, np.newaxis]
    y_binary = label_binarize(y_true, classes=labels)
    per_class_auc = roc_auc_score(y_binary, proba, average=None)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    counts = {label: int(np.sum(np.asarray(y_pred) == label)) for label in labels}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
        "roc_auc_ovr_macro": float(roc_auc_score(y_binary, proba, average="macro")),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(y_binary, proba, average="weighted")
        ),
        "class_recall": dict(zip(labels, map(float, recalls), strict=True)),
        "class_roc_auc": dict(zip(labels, map(float, per_class_auc), strict=True)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "prediction_distribution": {
            label: {"count": count, "rate": count / len(y_pred)}
            for label, count in counts.items()
        },
        "class_order": labels,
    }


def plot_multiclass_roc(
    y_true: Sequence[str],
    probabilities: NDArray[np.floating[Any]],
    class_labels: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Persist one-vs-rest ROC curves plus the micro-average curve."""
    import matplotlib.pyplot as plt

    labels = list(class_labels)
    y_binary = label_binarize(y_true, classes=labels)
    proba = np.asarray(probabilities, dtype=float)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_binary[:, index], proba[:, index])
        auc = roc_auc_score(y_binary[:, index], proba[:, index])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")
    micro_fpr, micro_tpr, _ = roc_curve(y_binary.ravel(), proba.ravel())
    micro_auc = roc_auc_score(y_binary, proba, average="micro")
    ax.plot(micro_fpr, micro_tpr, linestyle="--", label=f"micro (AUC={micro_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="OOF multiclass ROC (OvR)",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
