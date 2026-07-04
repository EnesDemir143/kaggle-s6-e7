"""Candidate feature builders used by analysis notebooks."""

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.add(1)).replace([np.inf, -np.inf], np.nan)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-motivated ratios without changing the input frame."""
    result = df.copy()
    definitions = {
        "calorie_per_step": ("calorie_expenditure", "step_count"),
        "calorie_per_exercise_min": ("calorie_expenditure", "exercise_duration"),
        "step_per_exercise_min": ("step_count", "exercise_duration"),
        "water_per_bmi": ("water_intake", "bmi"),
        "exercise_per_bmi": ("exercise_duration", "bmi"),
        "steps_per_sleep_hour": ("step_count", "sleep_duration"),
    }
    for output, (numerator, denominator) in definitions.items():
        result[output] = _safe_ratio(result[numerator], result[denominator])
    return result


def add_missing_features(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Add per-column missing flags and row-level missing count."""
    result, cols = df.copy(), list(cols)
    for col in cols:
        result[f"{col}_is_missing"] = result[col].isna().astype("int8")
    result["missing_count"] = result[cols].isna().sum(axis=1).astype("int16")
    return result


def add_categorical_interactions(
    df: pd.DataFrame, *, include_gender_activity: bool = True
) -> pd.DataFrame:
    """Add explicit lifestyle-category interactions."""
    result = df.copy()
    pairs = {
        "stress_sleep_quality": ("stress_level", "sleep_quality"),
        "activity_diet": ("physical_activity_level", "diet_type"),
        "smoking_activity": ("smoking_alcohol", "physical_activity_level"),
        "gender_activity": ("gender", "physical_activity_level"),
    }
    if not include_gender_activity:
        pairs.pop("gender_activity")
    for output, (left, right) in pairs.items():
        result[output] = (
            result[left].fillna("missing").astype(str)
            + "__"
            + result[right].fillna("missing").astype(str)
        )
    return result


def add_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add review-only threshold candidates, not medical diagnoses."""
    result = df.copy()
    definitions = {
        "low_sleep_flag": result.sleep_duration < 6,
        "high_sleep_flag": result.sleep_duration > 9,
        "high_bmi_flag": result.bmi >= 30,
        "low_bmi_flag": result.bmi < 18.5,
        "high_heart_rate_flag": result.heart_rate > 100,
        "low_heart_rate_flag": result.heart_rate < 60,
        "low_steps_flag": result.step_count < 3000,
        "high_steps_flag": result.step_count > 12000,
    }
    for col, mask in definitions.items():
        result[col] = mask.astype("int8")
    return result


def fit_outlier_bounds(
    train: pd.DataFrame,
    cols: Iterable[str],
    lower_q: float = 0.005,
    upper_q: float = 0.995,
) -> dict[str, tuple[float, float]]:
    """Learn quantile bounds from train only."""
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("Expected 0 <= lower_q < upper_q <= 1")
    return {col: tuple(train[col].quantile([lower_q, upper_q])) for col in cols}


def add_outlier_flags(
    df: pd.DataFrame, bounds: Mapping[str, tuple[float, float]]
) -> pd.DataFrame:
    """Apply fitted bounds and add row-level outlier count."""
    result, flag_cols = df.copy(), []
    for col, (low, high) in bounds.items():
        low_col, high_col = f"{col}_outlier_low", f"{col}_outlier_high"
        result[low_col] = (result[col] < low).astype("int8")
        result[high_col] = (result[col] > high).astype("int8")
        flag_cols.extend([low_col, high_col])
    result["outlier_count"] = result[flag_cols].sum(axis=1).astype("int16")
    return result
