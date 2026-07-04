"""Reusable EDA table builders."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def missing_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Return aligned missing counts and rates."""
    summary = pd.DataFrame({
        "train_missing_count": train.isna().sum(),
        "train_missing_rate": train.isna().mean(),
        "test_missing_count": test.isna().sum(),
        "test_missing_rate": test.isna().mean(),
    }).fillna(0)
    summary["missing_rate_delta"] = summary.test_missing_rate - summary.train_missing_rate
    return summary.sort_values("train_missing_rate", ascending=False)


def target_summary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Return target counts and proportions."""
    counts = df[target_col].value_counts(dropna=False)
    return pd.DataFrame({"count": counts, "rate": counts / len(df)})


def missingness_target_table(df: pd.DataFrame, col: str, target_col: str) -> pd.DataFrame:
    """Compare class proportions for missing and observed rows."""
    return pd.crosstab(df[col].isna(), df[target_col], normalize="index").rename_axis("is_missing")


def numeric_shift_summary(train: pd.DataFrame, test: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Summarize numeric location, spread, and KS distance."""
    records = []
    for col in cols:
        tr, te = train[col].dropna(), test[col].dropna()
        statistic, pvalue = ks_2samp(tr, te)
        records.append({"feature": col, "train_mean": tr.mean(), "test_mean": te.mean(),
                        "train_median": tr.median(), "test_median": te.median(),
                        "train_std": tr.std(), "test_std": te.std(),
                        "ks_statistic": statistic, "ks_pvalue": pvalue})
    return pd.DataFrame(records).set_index("feature").sort_values("ks_statistic", ascending=False)


def categorical_shift_summary(train: pd.DataFrame, test: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Return total-variation distance and unseen categories."""
    records = []
    for col in cols:
        tr = train[col].fillna("__MISSING__").astype(str)
        te = test[col].fillna("__MISSING__").astype(str)
        categories = sorted(set(tr) | set(te))
        tr_freq = tr.value_counts(normalize=True).reindex(categories, fill_value=0)
        te_freq = te.value_counts(normalize=True).reindex(categories, fill_value=0)
        records.append({"feature": col,
                        "total_variation_distance": 0.5 * np.abs(tr_freq - te_freq).sum(),
                        "train_unique": tr.nunique(), "test_unique": te.nunique(),
                        "test_only_categories": sorted(set(te) - set(tr))})
    return pd.DataFrame(records).set_index("feature").sort_values("total_variation_distance", ascending=False)


def outlier_summary(df: pd.DataFrame, cols: Iterable[str], lower_q: float = 0.005,
                    upper_q: float = 0.995) -> pd.DataFrame:
    """Calculate IQR and quantile outlier rates without deleting rows."""
    records = []
    for col in cols:
        values = df[col]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_low, iqr_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        quantile_low, quantile_high = values.quantile([lower_q, upper_q])
        records.append({"feature": col, "iqr_low": iqr_low, "iqr_high": iqr_high,
                        "iqr_outlier_rate": ((values < iqr_low) | (values > iqr_high)).mean(),
                        "quantile_low": quantile_low, "quantile_high": quantile_high,
                        "quantile_outlier_rate": ((values < quantile_low) | (values > quantile_high)).mean()})
    return pd.DataFrame(records).set_index("feature")
