"""Cross-validated LightGBM experiment execution and artifact persistence."""

import json
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .cache import FoldFeatureCache, file_fingerprint, stable_hash
from .config import CLASS_NAMES, ID_COL, TARGET_COL
from .evaluation import (
    classification_metrics,
    plot_multiclass_roc,
    predictions_from_probabilities,
)
from .model_adapters import create_model_adapter
from .preprocessing import FoldPreprocessor

log = logging.getLogger(__name__)


def sample_weights(y: pd.Series, mode: str | None) -> np.ndarray | None:
    if mode is None:
        return None
    counts = y.value_counts()
    balanced = {label: len(y) / (len(CLASS_NAMES) * counts[label]) for label in CLASS_NAMES}
    if mode == "sqrt_balanced":
        balanced = {label: value**0.5 for label, value in balanced.items()}
    elif mode != "balanced":
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    return y.map(balanced).to_numpy(dtype=float)


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    proba = np.asarray(values, dtype=np.float64)
    return proba / proba.sum(axis=1, keepdims=True)


def run_cv_experiment(
    *,
    experiment_id: str,
    experiment: dict[str, Any],
    model_params: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    train_path: Path,
    test_path: Path,
    cache: FoldFeatureCache,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = train.drop(columns=[ID_COL, TARGET_COL])
    y = train[TARGET_COL]
    X_test = test.drop(columns=ID_COL)
    labels = {name: index for index, name in enumerate(CLASS_NAMES)}
    y_encoded = y.map(labels).to_numpy()
    splitter = StratifiedKFold(
        n_splits=int(experiment["n_splits"]), shuffle=True, random_state=int(experiment["seed"])
    )
    oof = np.zeros((len(train), len(CLASS_NAMES)), dtype=np.float32)
    oof_fold_proba = np.full((len(train), len(CLASS_NAMES), splitter.n_splits), np.nan, dtype=np.float32)
    test_fold = np.zeros((len(test), len(CLASS_NAMES), splitter.n_splits), dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    fold_valid_indices: list[np.ndarray] = []
    importances: list[pd.DataFrame] = []
    cache_keys: list[str] = []
    data_fingerprint = {"train": file_fingerprint(train_path), "test": file_fingerprint(test_path)}
    model_name = str(experiment.get("model", "lightgbm"))

    log.info(
        "Starting experiment %s | %d-fold CV | seed=%s | features=%s",
        experiment_id,
        splitter.n_splits,
        experiment.get("seed"),
        list(experiment.get("features", {}).keys()),
    )
    fold_bar = tqdm(
        enumerate(splitter.split(X, y_encoded)),
        desc=f"Folds ({experiment_id})",
        unit="fold",
        total=splitter.n_splits,
    )
    for fold, (train_idx, valid_idx) in fold_bar:
        cache_key = cache.key(
            data={**data_fingerprint, "train_indices": stable_hash(train_idx.tolist())},
            fold=fold,
            config={"features": experiment["features"], "model": model_name},
        )
        cache_keys.append(cache_key)
        cached = cache.load(cache_key)
        if cached is None:
            log.debug("Cache MISS: fold %d — preprocessing", fold)
            processor = FoldPreprocessor(experiment["features"])
            X_train = processor.fit_transform(X.iloc[train_idx])
            X_valid = processor.transform(X.iloc[valid_idx])
            X_fold_test = processor.transform(X_test)
            cache.save(cache_key, X_train, X_valid, X_fold_test)
        else:
            X_train, X_valid, X_fold_test = cached
        model = create_model_adapter(model_name, model_params)
        model.fit(
            X_train,
            y_encoded[train_idx],
            X_valid,
            y_encoded[valid_idx],
            sample_weights(y.iloc[train_idx], experiment["training"].get("class_weight_mode")),
        )
        training_history.append(
            {"fold": fold, "best_iteration": model.best_iteration()}
        )
        fold_valid_indices.append(valid_idx.copy())
        valid_proba = normalize_probabilities(model.predict_proba(X_valid))
        oof_fold_proba[valid_idx, :, fold] = valid_proba
        oof[valid_idx] = valid_proba
        test_fold[:, :, fold] = normalize_probabilities(model.predict_proba(X_fold_test))
        valid_pred = predictions_from_probabilities(valid_proba, CLASS_NAMES)
        record = classification_metrics(
            y.iloc[valid_idx], valid_pred.tolist(), valid_proba, CLASS_NAMES
        )
        fold_bar.set_postfix({"bal_acc": f"{record['balanced_accuracy']:.4f}"})
        log.info(
            "Fold %d/%d | features=%d | best_iter=%d | bal_acc=%.4f | "
            "recall: at-risk=%.3f fit=%.3f unhealthy=%.3f",
            fold + 1,
            splitter.n_splits,
            X_train.shape[1],
            model.best_iteration(),
            record["balanced_accuracy"],
            record["class_recall"]["at-risk"],
            record["class_recall"]["fit"],
            record["class_recall"]["unhealthy"],
        )
        # Streaming JSONL log — one line per fold for post-hoc / crash recovery
        progress_line = {
            "event": "fold_complete",
            "experiment_id": experiment_id,
            "fold": fold,
            "features_count": X_train.shape[1],
            "best_iteration": model.best_iteration(),
            "balanced_accuracy": record["balanced_accuracy"],
            "macro_f1": record["f1_macro"],
            "class_recall": record["class_recall"],
        }
        (output_dir / "progress.jsonl").open("a").write(json.dumps(progress_line) + "\n")
        fold_records.append({
            "fold": fold,
            "balanced_accuracy": record["balanced_accuracy"],
            "macro_f1": record["f1_macro"],
            "at-risk_recall": record["class_recall"]["at-risk"],
            "fit_recall": record["class_recall"]["fit"],
            "unhealthy_recall": record["class_recall"]["unhealthy"],
            "best_iteration": model.best_iteration(),
            "features_count": X_train.shape[1],
        })
        fold_importance = model.feature_importance(X_train.columns.tolist())
        fold_importance["fold"] = fold
        importances.append(fold_importance)
        model.save(output_dir / f"model_fold{fold}")

    oof = normalize_probabilities(oof).astype(np.float32)
    test_proba = normalize_probabilities(test_fold.mean(axis=2)).astype(np.float32)
    oof_pred = predictions_from_probabilities(oof, CLASS_NAMES)
    metrics = classification_metrics(y, oof_pred.tolist(), oof, CLASS_NAMES)
    fold_frame = pd.DataFrame(fold_records)
    log.info(
        "Experiment %s complete | mean_bal_acc=%.4f ± %.4f | "
        "recall: at-risk=%.3f fit=%.3f unhealthy=%.3f | macro_f1=%.4f",
        experiment_id,
        metrics["balanced_accuracy"],
        fold_frame["balanced_accuracy"].std(ddof=1),
        metrics["class_recall"]["at-risk"],
        metrics["class_recall"]["fit"],
        metrics["class_recall"]["unhealthy"],
        metrics["f1_macro"],
    )
    # Streaming JSONL — experiment completion record
    summary_line = {
        "event": "experiment_complete",
        "experiment_id": experiment_id,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "balanced_accuracy_std": float(fold_frame["balanced_accuracy"].std(ddof=1)),
        "macro_f1": metrics["f1_macro"],
        "class_recall": metrics["class_recall"],
        "prediction_distribution": metrics["prediction_distribution"],
    }
    (output_dir / "progress.jsonl").open("a").write(json.dumps(summary_line) + "\n")
    metrics["experiment_id"] = experiment_id
    metrics["fold_summary"] = {
        col: {"mean": float(fold_frame[col].mean()), "std": float(fold_frame[col].std(ddof=1))}
        for col in fold_frame.columns if col != "fold"
    }
    np.save(output_dir / "oof_proba.npy", oof)
    np.save(output_dir / "test_proba.npy", test_proba)
    pd.DataFrame({ID_COL: train[ID_COL], "y_true": y, "y_pred": oof_pred}).to_csv(
        output_dir / "oof_pred.csv", index=False
    )
    pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: predictions_from_probabilities(test_proba, CLASS_NAMES)}).to_csv(
        output_dir / "submission_argmax.csv", index=False
    )
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    importance = pd.concat(importances, ignore_index=True)
    importance_summary = (
        importance.groupby("feature", as_index=False)[["importance_gain", "importance_split"]]
        .mean()
        .assign(fold="mean")
    )
    importance = pd.concat([importance, importance_summary], ignore_index=True)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    pd.DataFrame(confusion_matrix(y, oof_pred, labels=CLASS_NAMES), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(y, oof_pred, labels=CLASS_NAMES, zero_division=0)
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "label_mapping.json").write_text(json.dumps(labels, indent=2) + "\n")
    manifest = {
        "experiment_id": experiment_id,
        "python": platform.python_version(),
        "model": model_name,
        "model_library_version": model.library_version,
        "input_mode": model.input_mode,
        "categorical_columns": model.categorical_columns,
        "categorical_feature_count": len(model.categorical_columns),
        "train_rows": len(train),
        "test_rows": len(test),
        "cache_keys": cache_keys,
        "data_fingerprint": data_fingerprint,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- Extended artifacts ---

    # Per-fold test probabilities (n_test × n_classes × n_folds) for blending
    np.save(output_dir / "test_proba_fold.npy", test_fold.astype(np.float32))

    # Per-fold OOF probabilities (n_train × n_classes × n_folds, sparse NaN)
    # Each (:, :, fold) slice has NaN except the held-out validation rows
    np.save(output_dir / "oof_proba_fold.npy", oof_fold_proba)

    # Fold assignment per row (which fold held this row out as validation)
    fold_assign = np.full(len(train), -1, dtype=np.int8)
    for fold_idx, vidx in enumerate(fold_valid_indices):
        fold_assign[vidx] = fold_idx
    pd.DataFrame({"fold_assignment": fold_assign}).to_csv(
        output_dir / "fold_assignments.csv", index=False
    )

    # Training loss history (per-fold train/valid multi_logloss per iteration)
    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2) + "\n"
    )

    # OOF multiclass ROC curves (visual diagnostic)
    try:
        plot_multiclass_roc(
            y_true=y.tolist(),
            probabilities=oof,
            class_labels=CLASS_NAMES,
            output_path=output_dir / "roc_curves.png",
        )
        log.debug("ROC curves saved to %s", output_dir / "roc_curves.png")
    except Exception:
        log.warning("ROC plot failed (matplotlib may be missing)", exc_info=True)

    log.info("Extra artifacts: test_proba_fold, oof_proba_fold, fold_assignments, training_history, roc_curves")
    return metrics
