"""Fold-safe implementation of the P_MAIN_V2_CORE preprocessing contract."""

from collections.abc import Sequence
from typing import cast

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
    add_rule_features,
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
LOG_RATIO_COLS = [f"log_{col}" for col in RATIO_COLS]


class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Configurable, fold-fitted preprocessing for E001-E008."""

    def __init__(self, feature_config: dict[str, object]) -> None:
        self.feature_config = feature_config

    def fit(self, X: pd.DataFrame, y: object = None) -> "FoldPreprocessor":
        del y
        V2CorePreprocessor._validate_input(X)
        self.numeric_medians_ = X[RAW_NUMERIC_COLS].median().to_dict()
        prepared = self._prepare_before_outliers(X)
        outlier_cols = [*RAW_NUMERIC_COLS]
        if self._enabled("ratios"):
            outlier_cols.extend(RATIO_COLS)
        self.outlier_bounds_ = (
            fit_outlier_bounds(prepared, outlier_cols, 0.005, 0.995)
            if self._enabled("outlier_flags")
            else {}
        )
        if self._enabled("clipping"):
            lower = cast(float, self.feature_config.get("clip_lower_q", 0.001))
            upper = cast(float, self.feature_config.get("clip_upper_q", 0.999))
            self.clip_bounds_ = fit_outlier_bounds(prepared, outlier_cols, lower, upper)
        else:
            self.clip_bounds_ = {}
        categorical = [*RAW_CATEGORICAL_COLS]
        if self._enabled("categorical_interactions"):
            categorical.extend(INTERACTION_COLS)
        if self._enabled("gender_activity"):
            categorical.append("gender_activity")
        self.categorical_features_ = categorical
        transformed = self._finish(prepared)
        self.category_levels_ = {
            col: sorted(set(transformed[col].astype(str))) + ["__UNKNOWN__"]
            for col in categorical
        }
        transformed = self._cast_categories(transformed)
        self.feature_names_out_ = transformed.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        V2CorePreprocessor._validate_input(X)
        transformed = self._finish(self._prepare_before_outliers(X))
        transformed = transformed.reindex(columns=self.feature_names_out_)
        return self._cast_categories(transformed)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        del input_features
        self._require_fitted()
        return np.asarray(self.feature_names_out_, dtype=object)

    def _enabled(self, key: str) -> bool:
        return bool(self.feature_config.get(key, False))

    def _prepare_before_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        columns = [*RAW_NUMERIC_COLS, *RAW_CATEGORICAL_COLS]
        result = X[columns].copy()
        if self._enabled("missing_flags"):
            missing = result[columns].isna()
            for col in columns:
                result[f"{col}_is_missing"] = missing[col].astype("int8")
            if self._enabled("missing_count"):
                result["missing_count"] = missing.sum(axis=1).astype("int16")
        result[RAW_NUMERIC_COLS] = result[RAW_NUMERIC_COLS].fillna(
            self.numeric_medians_
        )
        result[RAW_CATEGORICAL_COLS] = result[RAW_CATEGORICAL_COLS].fillna("missing")
        if self._enabled("ratios"):
            result = add_ratio_features(result)
        if self._enabled("categorical_interactions"):
            result = add_categorical_interactions(result, include_gender_activity=False)
        if self._enabled("gender_activity"):
            result["gender_activity"] = (
                result["gender"].astype(str)
                + "__"
                + result["physical_activity_level"].astype(str)
            )
        if self._enabled("rule_flags"):
            result = add_rule_features(result)
        return result

    def _finish(self, prepared: pd.DataFrame) -> pd.DataFrame:
        result = prepared.copy()
        if self.outlier_bounds_:
            result = add_outlier_flags(result, self.outlier_bounds_)
            if not self._enabled("outlier_count"):
                result = result.drop(columns="outlier_count")
        for col, (low, high) in self.clip_bounds_.items():
            result[col] = result[col].clip(low, high)
        if self._enabled("log_ratio"):
            for source, output in zip(RATIO_COLS, LOG_RATIO_COLS, strict=True):
                result[output] = np.log1p(result[source].clip(lower=0))
        return result

    def _cast_categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for col, levels in self.category_levels_.items():
            values = result[col].astype(str)
            result[col] = values.where(values.isin(levels), "__UNKNOWN__")
            result[col] = pd.Categorical(result[col], categories=levels)
        return result

    def _require_fitted(self) -> None:
        if not hasattr(self, "numeric_medians_"):
            raise RuntimeError("FoldPreprocessor must be fitted before transform")


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
