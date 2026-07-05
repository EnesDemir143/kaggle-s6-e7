"""Small model-family adapters used by the shared cross-validation runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class ModelAdapter(Protocol):
    library_version: str
    input_mode: str
    categorical_columns: list[str]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def best_iteration(self) -> int: ...
    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame: ...
    def save(self, path: Path) -> None: ...


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that should be passed as native categorical features."""
    return [
        column
        for column in frame.columns
        if isinstance(frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]


class LightGBMAdapter:
    def __init__(self, params: dict[str, Any]) -> None:
        import lightgbm as lgb

        self._lgb = lgb
        self._early_stopping = int(params.pop("early_stopping_rounds"))
        self.model = lgb.LGBMClassifier(**params)
        self.library_version = lgb.__version__
        self.input_mode = "native_categorical"
        self.categorical_columns: list[str] = []

    def fit(self, X_train, y_train, X_valid, y_valid, sample_weight) -> None:
        self.categorical_columns = categorical_columns(X_train)
        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            categorical_feature="auto",
            callbacks=[
                self._lgb.early_stopping(
                    self._early_stopping, first_metric_only=True, verbose=False
                )
            ],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.model.predict_proba(X, num_iteration=self.model.best_iteration_)
        )

    def best_iteration(self) -> int:
        return int(self.model.best_iteration_)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance_gain": self.model.booster_.feature_importance("gain"),
                "importance_split": self.model.booster_.feature_importance("split"),
            }
        )

    def save(self, path: Path) -> None:
        self.model.booster_.save_model(path.with_suffix(".txt"))


class XGBoostAdapter:
    def __init__(self, params: dict[str, Any]) -> None:
        import xgboost as xgb

        self._xgb = xgb
        self._early_stopping = int(params.pop("early_stopping_rounds"))
        self.model = xgb.XGBClassifier(
            **params, early_stopping_rounds=self._early_stopping
        )
        self.library_version = xgb.__version__
        self.input_mode = "one_hot_sparse"
        self.categorical_columns: list[str] = []
        self.encoder: ColumnTransformer | None = None
        self.encoded_feature_names: list[str] = []

    def _transform(self, X: pd.DataFrame) -> sparse.spmatrix:
        assert self.encoder is not None
        return sparse.csr_matrix(self.encoder.transform(X))

    def fit(self, X_train, y_train, X_valid, y_valid, sample_weight) -> None:
        self.categorical_columns = categorical_columns(X_train)
        numeric = [c for c in X_train.columns if c not in self.categorical_columns]
        self.encoder = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    self.categorical_columns,
                ),
                ("numeric", "passthrough", numeric),
            ],
            verbose_feature_names_out=False,
        )
        X_train_encoded = sparse.csr_matrix(self.encoder.fit_transform(X_train))
        X_valid_encoded = self._transform(X_valid)
        self.encoded_feature_names = self.encoder.get_feature_names_out().tolist()
        self.model.fit(
            X_train_encoded,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid_encoded, y_valid)],
            verbose=False,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._transform(X)))

    def best_iteration(self) -> int:
        value = getattr(self.model, "best_iteration", None)
        if value is not None:
            return int(value + 1)
        estimators = self.model.n_estimators
        return int(estimators if estimators is not None else 0)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        del feature_names
        values = np.asarray(self.model.feature_importances_)
        return pd.DataFrame(
            {
                "feature": self.encoded_feature_names,
                "importance_gain": values,
                "importance_split": np.zeros_like(values),
            }
        )

    def save(self, path: Path) -> None:
        self.model.save_model(path.with_suffix(".json"))


class CatBoostAdapter:
    def __init__(self, params: dict[str, Any]) -> None:
        import catboost

        self._catboost = catboost
        self.model = catboost.CatBoostClassifier(**params)
        self.library_version = catboost.__version__
        self.input_mode = "numeric_only"
        self.categorical_columns: list[str] = []

    @staticmethod
    def _catboost_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = frame.copy()
        for column in columns:
            result[column] = result[column].astype(str)
        return result

    def fit(self, X_train, y_train, X_valid, y_valid, sample_weight) -> None:
        self.categorical_columns = categorical_columns(X_train)
        self.input_mode = (
            "native_categorical" if self.categorical_columns else "numeric_only"
        )
        train = self._catboost_frame(X_train, self.categorical_columns)
        valid = self._catboost_frame(X_valid, self.categorical_columns)
        kwargs: dict[str, Any] = {
            "X": train,
            "y": y_train,
            "sample_weight": sample_weight,
            "eval_set": (valid, y_valid),
            "verbose": False,
        }
        if self.categorical_columns:
            kwargs["cat_features"] = self.categorical_columns
        self.model.fit(**kwargs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._catboost_frame(X, self.categorical_columns)
        return np.asarray(self.model.predict_proba(frame))

    def best_iteration(self) -> int:
        value = self.model.get_best_iteration()
        if value is not None and value >= 0:
            return int(value + 1)
        return int(self.model.tree_count_ or 0)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        values = np.asarray(self.model.get_feature_importance())
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance_gain": values,
                "importance_split": np.zeros_like(values),
            }
        )

    def save(self, path: Path) -> None:
        self.model.save_model(str(path.with_suffix(".cbm")))


def create_model_adapter(model: str, params: dict[str, Any]) -> ModelAdapter:
    """Construct a supported model adapter without mutating caller parameters."""
    copied = dict(params)
    if model == "lightgbm":
        return LightGBMAdapter(copied)
    if model == "xgboost":
        return XGBoostAdapter(copied)
    if model == "catboost":
        return CatBoostAdapter(copied)
    raise ValueError(f"Unsupported model: {model}")
