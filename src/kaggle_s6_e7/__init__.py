"""Reusable EDA and feature-engineering helpers."""

from .data import infer_feature_columns, load_competition_data, validate_schema
from .features import (
    add_categorical_interactions,
    add_missing_features,
    add_outlier_flags,
    add_ratio_features,
    add_rule_features,
    fit_outlier_bounds,
)
from .preprocessing import V2CorePreprocessor, build_one_hot_encoder

__all__ = [
    "add_categorical_interactions",
    "add_missing_features",
    "add_outlier_flags",
    "add_ratio_features",
    "add_rule_features",
    "fit_outlier_bounds",
    "infer_feature_columns",
    "load_competition_data",
    "validate_schema",
    "V2CorePreprocessor",
    "build_one_hot_encoder",
]
