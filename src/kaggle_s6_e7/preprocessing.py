"""Fold-safe implementation of the P_MAIN_V2_CORE preprocessing contract."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .features import (
    add_categorical_interactions,
    add_missing_features,
    add_outlier_flags,
    add_ratio_features,
    fit_outlier_bounds,
)

RAW_NUMERIC_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
RAW_CATEGORICAL_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
RATIO_COLS = [
    "exercise_per_bmi",
    "calorie_per_step",
    "calorie_per_exercise_min",
    "step_per_exercise_min",
    "water_per_bmi",
    "steps_per_sleep_hour",
]
INTERACTION_COLS = ["stress_sleep_quality", "activity_diet", "smoking_activity"]


class V2CorePreprocessor(TransformerMixin, BaseEstimator):
    """Create V2-Core features while learning every statistic in ``fit`` only.

    The output remains a pandas frame so CatBoost/LightGBM can consume categorical
    columns natively. Use :func:`build_one_hot_encoder` for sklearn/XGBoost models.
    """

    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: pd.DataFrame, y: object = None) -> "V2CorePreprocessor":
        del y
        self._validate_input(X)
        self.numeric_medians_ = X[RAW_NUMERIC_COLS].median().to_dict()
        prepared = self._base_features(X)
        outlier_cols = [*RAW_NUMERIC_COLS, *RATIO_COLS]
        self.outlier_bounds_ = fit_outlier_bounds(
            prepared, outlier_cols, self.lower_q, self.upper_q
        )
        flagged = add_outlier_flags(prepared, self.outlier_bounds_)
        candidate_flags = [
            f"{col}_outlier_{side}" for col in outlier_cols for side in ("low", "high")
        ]
        self.active_outlier_flags_ = [
            col for col in candidate_flags if flagged[col].nunique(dropna=False) > 1
        ]
        transformed = self._select_output(flagged)
        self.feature_names_out_ = transformed.columns.tolist()
        self.categorical_features_ = [*RAW_CATEGORICAL_COLS, *INTERACTION_COLS]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_input(X)
        prepared = self._base_features(X)
        flagged = add_outlier_flags(prepared, self.outlier_bounds_)
        return self._select_output(flagged).reindex(columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        del input_features
        self._require_fitted()
        return np.asarray(self.feature_names_out_, dtype=object)

    def _base_features(self, X: pd.DataFrame) -> pd.DataFrame:
        features = X[[*RAW_NUMERIC_COLS, *RAW_CATEGORICAL_COLS]].copy()
        features = add_missing_features(
            features, [*RAW_NUMERIC_COLS, *RAW_CATEGORICAL_COLS]
        )
        features[RAW_NUMERIC_COLS] = features[RAW_NUMERIC_COLS].fillna(
            self.numeric_medians_
        )
        features[RAW_CATEGORICAL_COLS] = features[RAW_CATEGORICAL_COLS].fillna(
            "missing"
        )
        features = add_ratio_features(features)
        features = add_categorical_interactions(features, include_gender_activity=False)
        return features

    def _select_output(self, flagged: pd.DataFrame) -> pd.DataFrame:
        flagged = flagged.copy()
        if self.active_outlier_flags_:
            flagged["outlier_count"] = (
                flagged[self.active_outlier_flags_].sum(axis=1).astype("int16")
            )
        else:
            flagged["outlier_count"] = np.int16(0)
        base = [
            *RAW_NUMERIC_COLS,
            *RAW_CATEGORICAL_COLS,
            *(
                f"{col}_is_missing"
                for col in [*RAW_NUMERIC_COLS, *RAW_CATEGORICAL_COLS]
            ),
            "missing_count",
            *RATIO_COLS,
            *INTERACTION_COLS,
            *self.active_outlier_flags_,
            "outlier_count",
        ]
        return flagged.loc[:, base].copy()

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        missing = set([*RAW_NUMERIC_COLS, *RAW_CATEGORICAL_COLS]) - set(X.columns)
        if missing:
            raise ValueError(f"V2-Core input is missing columns: {sorted(missing)}")

    def _require_fitted(self) -> None:
        if not hasattr(self, "numeric_medians_"):
            raise RuntimeError("V2CorePreprocessor must be fitted before transform")


def build_one_hot_encoder(
    categorical_cols: Sequence[str], numeric_cols: Sequence[str]
) -> ColumnTransformer:
    """Build model encoding with explicit unknown-category handling."""
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(categorical_cols),
            ),
            ("numeric", "passthrough", list(numeric_cols)),
        ],
        verbose_feature_names_out=False,
    )
