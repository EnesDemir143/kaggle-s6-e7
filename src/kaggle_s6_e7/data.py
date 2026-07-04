"""Data loading and schema validation."""

from pathlib import Path

import pandas as pd

from .config import EXPECTED_TARGETS, ID_COL, TARGET_COL, TEST_PATH, TRAIN_PATH


def load_competition_data(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def validate_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
) -> None:
    """Raise a descriptive error when the competition schema is inconsistent."""
    if target_col not in train:
        raise ValueError(f"Train is missing target column: {target_col}")
    if target_col in test:
        raise ValueError(f"Test must not contain target column: {target_col}")
    if id_col not in train or id_col not in test:
        raise ValueError(f"Both datasets must contain id column: {id_col}")
    if set(train.columns) - {target_col} != set(test.columns):
        raise ValueError("Train feature columns do not match test columns")
    targets = set(train[target_col].dropna().unique())
    if targets != EXPECTED_TARGETS:
        raise ValueError(f"Unexpected target labels: {sorted(targets)}")
    if train[id_col].duplicated().any() or test[id_col].duplicated().any():
        raise ValueError("Duplicate ids detected")


def infer_feature_columns(
    train: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
) -> tuple[list[str], list[str]]:
    """Return categorical and numeric feature names in source-column order."""
    features = train.drop(columns=[id_col, target_col], errors="ignore")
    cat_cols = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    num_cols = features.select_dtypes(include="number").columns.tolist()
    return cat_cols, num_cols
